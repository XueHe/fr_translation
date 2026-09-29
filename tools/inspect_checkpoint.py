#!/usr/bin/env python3
"""Read-only validation and summary for a resumable translation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_journal(path: Path, shard: int) -> tuple[int, int, float]:
    expected_start = 0
    batches = 0
    inference_seconds = 0.0
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            try:
                envelope = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc
            if envelope.get("schema_version") != 1 or envelope.get("shard") != shard:
                raise RuntimeError(f"Schema/shard mismatch at {path}:{line_number}")
            if envelope.get("input_start") != expected_start:
                raise RuntimeError(
                    f"Non-contiguous journal at {path}:{line_number}: "
                    f"expected {expected_start}, got {envelope.get('input_start')}"
                )
            records = envelope.get("records", [])
            expected_end = expected_start + len(records)
            if not records or envelope.get("input_end") != expected_end:
                raise RuntimeError(f"Invalid batch envelope at {path}:{line_number}")
            expected_start = expected_end
            batches += 1
            inference_seconds += float(envelope.get("batch_seconds", 0.0))
    return expected_start, batches, inference_seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--skip-input-hashes", action="store_true")
    args = parser.parse_args()

    root = args.checkpoint.resolve()
    manifest = json.loads((root / "input_manifest.json").read_text(encoding="utf-8"))
    print(
        f"eligible_source_rows={manifest['eligible_source_rows']:,} "
        f"unique_terms={manifest['unique_terms']:,}"
    )
    for shard, target in enumerate(manifest["shard_counts"]):
        input_path = root / f"terms_shard_{shard:02d}.jsonl"
        if not args.skip_input_hashes:
            actual = sha256_file(input_path)
            expected = manifest["shard_sha256"][shard]
            if actual != expected:
                raise RuntimeError(
                    f"Input hash mismatch for shard {shard}: expected {expected}, got {actual}"
                )
        journal = root / f"run_shard_{shard:02d}" / "raw_responses.jsonl"
        offset, batches, seconds = inspect_journal(journal, shard)
        print(
            f"shard={shard} offset={offset:,}/{target:,} batches={batches:,} "
            f"inference_hours={seconds / 3600:.2f}"
        )


if __name__ == "__main__":
    main()

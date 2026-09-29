#!/usr/bin/env python3
"""Quarantine and truncate a batch journal at an exact record boundary."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def locate_boundary(path: Path, shard: int, target: int) -> tuple[int, int, int]:
    expected_start = 0
    boundary = None
    total_lines = 0
    with path.open("rb") as handle:
        if target == 0:
            boundary = 0
        for line_number, raw in enumerate(handle, 1):
            total_lines = line_number
            envelope = json.loads(raw)
            if envelope.get("schema_version") != 1 or envelope.get("shard") != shard:
                raise RuntimeError(f"Schema/shard mismatch at line {line_number}")
            if envelope.get("input_start") != expected_start:
                raise RuntimeError(f"Non-contiguous journal at line {line_number}")
            records = envelope.get("records", [])
            expected_end = expected_start + len(records)
            if not records or envelope.get("input_end") != expected_end:
                raise RuntimeError(f"Invalid batch envelope at line {line_number}")
            expected_start = expected_end
            if expected_start == target:
                boundary = handle.tell()
            elif expected_start > target and boundary is None:
                raise RuntimeError(f"Offset {target} is not a batch boundary")
    if boundary is None:
        raise RuntimeError(f"Offset {target} was not found; journal ends at {expected_start}")
    return boundary, expected_start, total_lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("journal", type=Path)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    path = args.journal.resolve()
    boundary, current, lines = locate_boundary(path, args.shard, args.offset)
    removed = current - args.offset
    print(
        f"journal={path} shard={args.shard} current={current} target={args.offset} "
        f"remove_records={removed} batches={lines}"
    )
    if removed == 0:
        print("No rollback is needed.")
        return
    if not args.apply:
        print("Dry run only. Re-run with --apply to quarantine the tail and truncate.")
        return

    stamp = time.strftime("%Y%m%dT%H%M%S")
    quarantine = path.with_name(f"{path.name}.quarantine.{stamp}.offset-{args.offset}")
    with path.open("rb+") as source, quarantine.open("xb") as target:
        source.seek(boundary)
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            target.write(chunk)
        target.flush()
        os.fsync(target.fileno())
        source.truncate(boundary)
        source.flush()
        os.fsync(source.fileno())
    print(f"Rollback complete. Quarantined tail: {quarantine}")


if __name__ == "__main__":
    main()

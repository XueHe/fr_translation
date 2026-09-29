#!/usr/bin/env python3
"""Resumable full-corpus UMLS translation with Llama 3.3 70B.

The prepare step freezes eligible source rows and globally deduplicates normalized
English terms. Each inference shard writes an append-only, fsync'd batch journal.
The journal is the source of truth for resume and retains the raw generated token
IDs plus decoded model output before cleanup.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import platform
import sys
import time
from itertools import islice
from pathlib import Path


MODEL_REPO = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_REVISION = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
SCHEMA_VERSION = 1
PROMPT = (
    "Translate the following English biomedical term into French. "
    "Preserve its complete meaning, including anatomical site, laterality, negation, "
    "specimen, measurement properties, units, species, and qualifiers. "
    "Use established French biomedical terminology. Preserve identifiers, gene symbols, "
    "formulas, and scientific names when appropriate. Do not add information or expand "
    "an ambiguous abbreviation without evidence. Return only one translated term, "
    "without explanations or alternatives.\n\nEnglish term: {term}"
)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def term_id(term: str) -> str:
    return hashlib.sha256(term.encode("utf-8")).hexdigest()


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_manifest = output / "input_manifest.json"
    if final_manifest.exists() and not args.force:
        print(f"Frozen input already exists: {final_manifest}", flush=True)
        return

    pilot_manifest = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    allowed = set(pilot_manifest["source_allowlist"])
    shard_temps = [output / f"terms_shard_{i:02d}.jsonl.partial" for i in range(args.shards)]
    shard_finals = [output / f"terms_shard_{i:02d}.jsonl" for i in range(args.shards)]
    mapping_temp = output / "eligible_source_rows.jsonl.gz.partial"
    mapping_final = output / "eligible_source_rows.jsonl.gz"
    for path in shard_temps + shard_finals + [mapping_temp, mapping_final]:
        if path.exists():
            path.unlink()

    seen_terms: set[str] = set()
    id_to_term: dict[str, str] = {}
    source_digest = hashlib.sha256()
    eligible_rows = 0
    shard_counts = [0] * args.shards
    started = time.monotonic()
    shard_handles = [path.open("w", encoding="utf-8", buffering=1024 * 1024) for path in shard_temps]
    try:
        with gzip.open(mapping_temp, "wt", encoding="utf-8", compresslevel=4) as mapping:
            with args.mrconso.open("rb") as source:
                for source_row, raw in enumerate(source, 1):
                    source_digest.update(raw)
                    fields = raw.decode("utf-8").rstrip("\r\n").split("|")
                    if (
                        len(fields) < 18
                        or fields[1] != "ENG"
                        or fields[16] != "N"
                        or fields[11] not in allowed
                        or not fields[14].strip()
                        or not (len(fields[0]) == 8 and fields[0][0] == "C" and fields[0][1:].isdigit())
                    ):
                        if args.limit_source_rows and source_row >= args.limit_source_rows:
                            break
                        continue
                    english = " ".join(fields[14].split())
                    identifier = term_id(english)
                    prior = id_to_term.get(identifier)
                    if prior is not None and prior != english:
                        raise RuntimeError(f"SHA-256 collision between {prior!r} and {english!r}")
                    id_to_term[identifier] = english
                    eligible_rows += 1
                    mapping.write(json.dumps({
                        "source_row": source_row,
                        "term_id": identifier,
                        "cui": fields[0],
                        "lat": fields[1],
                        "aui": fields[7],
                        "sab": fields[11],
                        "tty": fields[12],
                        "code": fields[13],
                        "english": english,
                        "suppress": fields[16],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if english not in seen_terms:
                        seen_terms.add(english)
                        shard = int(identifier[:16], 16) % args.shards
                        shard_handles[shard].write(json.dumps({
                            "term_id": identifier,
                            "english": english,
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                        shard_counts[shard] += 1
                    if eligible_rows % 100000 == 0:
                        elapsed = max(time.monotonic() - started, 1e-9)
                        print(
                            f"prepare eligible={eligible_rows:,} unique={len(seen_terms):,} "
                            f"rate={eligible_rows / elapsed:,.0f} rows/s",
                            flush=True,
                        )
                    if args.limit_source_rows and source_row >= args.limit_source_rows:
                        break
    finally:
        for handle in shard_handles:
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

    os.replace(mapping_temp, mapping_final)
    for temp, final in zip(shard_temps, shard_finals):
        os.replace(temp, final)
    source_sha256 = source_digest.hexdigest()
    expected_source_sha256 = pilot_manifest.get("mrconso_sha256")
    if not args.limit_source_rows and source_sha256 != expected_source_sha256:
        raise RuntimeError(
            f"MRCONSO checksum mismatch: expected {expected_source_sha256}, got {source_sha256}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mrconso": str(args.mrconso.resolve()),
        "mrconso_sha256": source_sha256,
        "pilot_manifest": str(args.pilot_manifest.resolve()),
        "source_allowlist": sorted(allowed),
        "filters": ["LAT=ENG", "SUPPRESS=N", "nonempty STR", "valid CUI", "historical source allowlist"],
        "normalization": "Collapse all Unicode whitespace runs in STR to one ASCII space and strip ends.",
        "deduplication": "Translate each globally unique normalized English STR exactly once; retain all eligible source rows in mapping.",
        "eligible_source_rows": eligible_rows,
        "unique_terms": len(seen_terms),
        "shards": args.shards,
        "shard_counts": shard_counts,
        "shard_sha256": [sha256_file(path) for path in shard_finals],
        "mapping_file": mapping_final.name,
        "mapping_sha256": sha256_file(mapping_final),
        "limited_source_rows": args.limit_source_rows,
    }
    atomic_json(final_manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def recover_journal(path: Path, shard: int) -> tuple[int, int, int, float]:
    """Validate complete JSONL envelopes and truncate only an incomplete tail."""
    if not path.exists():
        return 0, 0, 0, 0.0
    expected_start = 0
    batches = records = 0
    inference_seconds = 0.0
    valid_end = 0
    with path.open("rb+") as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                envelope = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                handle.seek(line_start)
                handle.truncate()
                print(f"Truncated incomplete journal tail at byte {line_start}", flush=True)
                break
            if envelope.get("schema_version") != SCHEMA_VERSION or envelope.get("shard") != shard:
                raise RuntimeError(f"Journal schema/shard mismatch at byte {line_start}")
            if envelope.get("input_start") != expected_start:
                raise RuntimeError(
                    f"Non-contiguous journal at byte {line_start}: "
                    f"expected {expected_start}, got {envelope.get('input_start')}"
                )
            batch_records = envelope.get("records", [])
            expected_end = expected_start + len(batch_records)
            if envelope.get("input_end") != expected_end or not batch_records:
                raise RuntimeError(f"Invalid batch envelope at byte {line_start}")
            expected_start = expected_end
            records += len(batch_records)
            batches += 1
            inference_seconds += float(envelope.get("batch_seconds", 0.0))
            valid_end = handle.tell()
        handle.seek(0, os.SEEK_END)
        if handle.tell() != valid_end:
            handle.truncate(valid_end)
    return expected_start, batches, records, inference_seconds


def iter_input(path: Path, skip: int):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index < skip:
                continue
            if line.strip():
                yield json.loads(line)


def batched(iterator, size: int):
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def run_shard(args: argparse.Namespace) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = args.output.resolve()
    manifest = json.loads((output / "input_manifest.json").read_text(encoding="utf-8"))
    input_path = output / f"terms_shard_{args.shard:02d}.jsonl"
    expected_input_sha = manifest["shard_sha256"][args.shard]
    if sha256_file(input_path) != expected_input_sha:
        raise RuntimeError(f"Frozen input checksum mismatch: {input_path}")
    shard_dir = output / f"run_shard_{args.shard:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (shard_dir / "worker.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Shard {args.shard} already has an active worker") from exc

    journal = shard_dir / "raw_responses.jsonl"
    offset, prior_batches, prior_records, prior_seconds = recover_journal(journal, args.shard)
    target = manifest["shard_counts"][args.shard]
    if offset == target:
        print(f"Shard {args.shard} already complete ({target:,} records)", flush=True)
        return
    if offset > target:
        raise RuntimeError(f"Journal has {offset} records but target is {target}")

    visible = torch.cuda.device_count()
    if visible != args.expected_gpus:
        raise RuntimeError(f"Expected {args.expected_gpus} visible GPUs, found {visible}")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    model_path = args.model_path.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="balanced",
        max_memory={i: args.max_memory_per_gpu for i in range(visible)},
        low_cpu_mem_usage=True,
    )
    model.eval()
    first_device = next(model.parameters()).device
    load_seconds = time.monotonic() - load_started
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_REPO,
        "revision": MODEL_REVISION,
        "model_path": str(model_path),
        "model_path_manifest": json.loads((model_path / "download_manifest.json").read_text()),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "input": str(input_path),
        "input_sha256": expected_input_sha,
        "shard": args.shard,
        "target": target,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
        "dtype": "bfloat16",
        "visible_gpus": visible,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(visible)],
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "load_seconds_this_start": load_seconds,
        "resume_offset": offset,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(shard_dir / "run_config.json", run_config)
    print(
        f"Loaded shard={args.shard} in {load_seconds:.1f}s; "
        f"resume={offset:,}/{target:,}",
        flush=True,
    )

    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    current = offset
    session_records = 0
    session_seconds = 0.0
    input_iterator = iter_input(input_path, offset)
    if args.max_session_records:
        input_iterator = islice(input_iterator, args.max_session_records)
    with journal.open("ab", buffering=0) as handle:
        for batch_index, batch in enumerate(batched(input_iterator, args.batch_size), prior_batches):
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT.format(term=row["english"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(
                rendered,
                padding=True,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            if encoded.input_ids.shape[1] > args.max_input_tokens:
                raise RuntimeError(
                    f"Input length {encoded.input_ids.shape[1]} exceeds limit {args.max_input_tokens}"
                )
            encoded = {key: value.to(first_device) for key, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            generated_rows = output_ids[:, prompt_length:].tolist()
            records = []
            for source, rendered_prompt, ids in zip(batch, rendered, generated_rows):
                eos_position = next((i for i, token in enumerate(ids) if token in eos_ids), None)
                token_count = eos_position + 1 if eos_position is not None else len(ids)
                raw_output = tokenizer.decode(
                    ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                raw_with_special = tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                cleaned = raw_output.strip()
                records.append({
                    "term_id": source["term_id"],
                    "english": source["english"],
                    "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
                    "generated_token_ids": ids,
                    "raw_output": raw_output,
                    "raw_output_with_special_tokens": raw_with_special,
                    "french": cleaned,
                    "generated_tokens_through_eos": token_count,
                    "finish_reason": "eos" if eos_position is not None else "length",
                    "empty": not cleaned,
                    "unchanged": cleaned.casefold() == source["english"].casefold(),
                })
            envelope = {
                "schema_version": SCHEMA_VERSION,
                "shard": args.shard,
                "batch_index": batch_index,
                "input_start": current,
                "input_end": current + len(records),
                "batch_seconds": elapsed,
                "batch_records": len(records),
                "committed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "records": records,
            }
            payload = (json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(payload)
            os.fsync(handle.fileno())
            current += len(records)
            session_records += len(records)
            session_seconds += elapsed
            cumulative_seconds = prior_seconds + session_seconds
            status = {
                "status": "running" if current < target else "complete",
                "shard": args.shard,
                "completed": current,
                "target": target,
                "fraction": current / target,
                "batches": batch_index + 1,
                "session_terms_per_second": session_records / max(session_seconds, 1e-9),
                "journal_terms_per_second": current / max(cumulative_seconds, 1e-9),
                "inference_seconds": cumulative_seconds,
                "load_seconds_this_start": load_seconds,
                "peak_gpu_gib": [
                    torch.cuda.max_memory_allocated(i) / 2**30 for i in range(visible)
                ],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            atomic_json(shard_dir / "status.json", status)
            print(
                f"shard={args.shard} {current:,}/{target:,} "
                f"session_rate={status['session_terms_per_second']:.3f}/s",
                flush=True,
            )
    if current == target:
        print(f"Shard {args.shard} complete: {current:,} records", flush=True)
    else:
        print(
            f"Shard {args.shard} session stopped cleanly at {current:,}/{target:,}",
            flush=True,
        )


def status(args: argparse.Namespace) -> None:
    manifest_path = args.output / "input_manifest.json"
    if not manifest_path.exists():
        print("Input preparation is not complete")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(
        f"eligible_source_rows={manifest['eligible_source_rows']:,} "
        f"unique_terms={manifest['unique_terms']:,}"
    )
    for shard in range(manifest["shards"]):
        path = args.output / f"run_shard_{shard:02d}" / "status.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            print(
                f"shard={shard} {item['status']} {item['completed']:,}/{item['target']:,} "
                f"rate={item['session_terms_per_second']:.3f}/s updated={item['updated_at']}"
            )
        else:
            print(f"shard={shard} not_started target={manifest['shard_counts'][shard]:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--mrconso", type=Path, required=True)
    prepare_parser.add_argument("--pilot-manifest", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    prepare_parser.add_argument("--shards", type=int, default=2)
    prepare_parser.add_argument("--limit-source-rows", type=int)
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run-shard")
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--model-path", type=Path, required=True)
    run_parser.add_argument("--shard", type=int, required=True)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--max-new-tokens", type=int, default=128)
    run_parser.add_argument("--max-input-tokens", type=int, default=2048)
    run_parser.add_argument("--expected-gpus", type=int, default=2)
    run_parser.add_argument("--max-memory-per-gpu", default="90GiB")
    run_parser.add_argument("--max-session-records", type=int)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    {"prepare": prepare, "run-shard": run_shard, "status": status}[args.command](args)


if __name__ == "__main__":
    main()

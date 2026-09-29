# fr_translation

Resumable English-to-French biomedical term translation with
`meta-llama/Llama-3.3-70B-Instruct`.

The runner translates each globally unique normalized English term once, splits
the work into two deterministic shards, and appends one `fsync`-committed JSON
envelope per batch. A restarted worker validates the journal and resumes at the
next uncommitted input record.

## Repository contents

- `scripts/translation_llama_full.py`: preparation, inference, and status CLI.
- `slurm/translate_4gpu.sbatch`: one-node, four-GPU SLURM launcher. It runs two
  independent two-GPU replicas, one per shard.
- `tools/inspect_checkpoint.py`: read-only validation of input hashes and journal
  continuity.
- `tools/rollback_journal.py`: auditable rollback to a trusted batch boundary.
- `handoff/checkpoint_state_20260929.json`: current and trusted resume offsets.

## Data and model are not in Git

The checkpoint directory is about 1.1 GB and contains UMLS-derived terms. UMLS
content must be transferred privately and handled under the recipient's UMLS
license. Do not upload the checkpoint, `MRCONSO.RRF`, generated journals, or model
weights to a public GitHub repository.

The required private checkpoint layout is:

```text
llama3_3_70b_full_20260928/
  input_manifest.json
  eligible_source_rows.jsonl.gz
  terms_shard_00.jsonl
  terms_shard_01.jsonl
  run_shard_00/raw_responses.jsonl
  run_shard_01/raw_responses.jsonl
```

Transfer it directly to Jean-Zay, for example from a host that can reach both
systems:

```bash
rsync -aP --partial --append-verify \
  cng6:/data/projects/deepaslr/sapalibert/results/llama3_3_70b_full_20260928/ \
  USER@jean-zay.idris.fr:/path/to/private/llama3_3_70b_full_20260928/
```

Replace `USER` and `/path/to/private`. If direct host-to-host access is not
available, stage the directory on an approved private storage system. Verify the
checkpoint after transfer:

```bash
python tools/inspect_checkpoint.py /path/to/private/llama3_3_70b_full_20260928
```

## Important checkpoint correction

GPU 1 on the original host reported uncorrectable ECC errors and a pending row
remap. Shard 0 produced 136 records after that condition was known. Before
resuming on healthy hardware, quarantine and roll shard 0 back from `261592` to
the last trusted batch boundary, `261456`:

```bash
python tools/rollback_journal.py \
  /path/to/private/llama3_3_70b_full_20260928/run_shard_00/raw_responses.jsonl \
  --shard 0 --offset 261456

# Review the plan, then apply it. The removed bytes are preserved separately.
python tools/rollback_journal.py \
  /path/to/private/llama3_3_70b_full_20260928/run_shard_00/raw_responses.jsonl \
  --shard 0 --offset 261456 --apply
```

Shard 1 used GPUs 2 and 3 and has a trusted resume offset of `260072`.

Run the read-only inspector again after rollback. Expected offsets are then
`261456` and `260072`.

## Environment

Use Python 3.12 with a CUDA-compatible PyTorch installation, then install:

```bash
python -m pip install -r requirements.txt
```

The original run used Python 3.12.3, PyTorch 2.13.0+cu129, Transformers 5.17.0,
BF16, greedy decoding, batch size 8, and a fixed model revision:
`6f6073b423013f6a7d4d9f39144961bfbfbc386b`.

The local model directory must contain the complete gated Llama model and a
`download_manifest.json`. A template is provided at
`config/download_manifest.example.json`; update `downloaded_at` after obtaining
the exact pinned revision through an authorized Hugging Face account.

## Resume on Jean-Zay

Edit the site-specific `#SBATCH` account/partition directives if required, then
submit from the repository root:

```bash
export OUTPUT_DIR=/path/to/private/llama3_3_70b_full_20260928
export MODEL_DIR=/path/to/private/llama3_3_70b
export PYTHON=/path/to/venv/bin/python

sbatch --export=ALL slurm/translate_4gpu.sbatch
```

The job requests four GPUs on one node and launches:

- shard 0 on visible GPUs `0,1`;
- shard 1 on visible GPUs `2,3`.

The script traps `SIGTERM`/`SIGINT`, allowing SLURM cancellation or wall-time
expiry without losing completed batches. Re-submit the same command to resume.

Check progress with:

```bash
$PYTHON scripts/translation_llama_full.py status --output "$OUTPUT_DIR"
```

## Safety checks

Before every resume:

1. Run `tools/inspect_checkpoint.py` and confirm journal continuity.
2. Confirm all allocated GPUs are healthy and have no pending row remap.
3. Confirm `input_manifest.json` and both term shard hashes match.
4. Keep the checkpoint on private, backed-up storage.

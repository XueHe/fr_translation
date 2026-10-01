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

The local model directory must contain the complete gated Llama model. An actual
`download_manifest.json` can optionally be supplied via `MODEL_MANIFEST`; otherwise
the runner checks the model directory for it. Missing provenance is recorded as
null. `config/download_manifest.example.json` is a template, not proof of the
revision of a shared model. Confirm the shared model revision before combining
its results with the original run. Metadata alone does not verify model weights.

## Resume on Jean-Zay

Copy `.env.example` to `.env` and fill in the absolute paths for this machine.
The private `.env` is ignored by Git. It uses trusted Bash assignment syntax;
quote paths with spaces. Both Python and the scheduler load this configuration.
CLI arguments override Python configuration values. Edit the site-specific
`#SBATCH` account/partition directives, then submit from the repository root:

```bash
cp .env.example .env
# Edit .env before submitting.
sbatch --export=ALL slurm/translate_4gpu.sbatch
```

For submission from another directory, provide the absolute configuration path:

```bash
ENV_FILE=/absolute/path/to/fr_translation/.env sbatch --export=ALL /absolute/path/to/fr_translation/slurm/translate_4gpu.sbatch
```

Set `MAX_SESSION_RECORDS=100` for a trial of at most 100 new terms per shard.
Empty it for full continuation. Leave `GPU_DEVICES` empty under SLURM: the job
preserves scheduler-provided `CUDA_VISIBLE_DEVICES`, including GPU UUIDs.
If the site sets GPU visibility only inside `srun`, launch the script within
that site's GPU job step. It fails clearly rather than selecting unallocated GPUs.
Outside SLURM, `GPU_DEVICES` can explicitly select the four devices.

The job requests four GPUs on one node and launches:

- shard 0 on the first two allocated devices;
- shard 1 on the remaining two allocated devices.

The script traps `SIGTERM`/`SIGINT`, allowing SLURM cancellation or wall-time
expiry without losing completed batches. Re-submit the same command to resume.

Check progress with:

```bash
python scripts/translation_llama_full.py --env-file /absolute/path/to/fr_translation/.env status
```

## Safety checks

Before every resume:

1. Run `tools/inspect_checkpoint.py` and confirm journal continuity.
2. Confirm all allocated GPUs are healthy and have no pending row remap.
3. Confirm `input_manifest.json` and both term shard hashes match.
4. Keep the checkpoint on private, backed-up storage.

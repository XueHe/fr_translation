#!/usr/bin/env bash
#SBATCH --job-name=fr-translate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

SCRIPT_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# SLURM may execute a copied script; submit from the repository or set ENV_FILE.
ENV_FILE="${ENV_FILE:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_DIR}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
elif [[ -n "${ENV_FILE_REQUIRED:-}" ]]; then
  echo "Missing configuration: $ENV_FILE" >&2
  exit 1
fi

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the private checkpoint directory}"
: "${MODEL_DIR:?Set MODEL_DIR to the pinned Llama model directory}"

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$SCRIPT_REPO_DIR}}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-76GiB}"
STAGE_MODEL="${STAGE_MODEL:-1}"
for path in "$REPO_DIR" "$OUTPUT_DIR" "$MODEL_DIR"; do
  if [[ "$path" != /* ]]; then
    echo "Use absolute directory paths in .env: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$REPO_DIR/scripts/translation_llama_full.py" ]]; then
  echo "Invalid REPO_DIR: $REPO_DIR" >&2
  exit 1
fi

if [[ ! -f "$OUTPUT_DIR/input_manifest.json" ]]; then
  echo "Missing checkpoint manifest: $OUTPUT_DIR/input_manifest.json" >&2
  exit 1
fi

RUN_MODEL_DIR="$MODEL_DIR"
if [[ "$STAGE_MODEL" == "1" ]]; then
  LOCAL_ROOT="${JOBSCRATCH:-${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}}"
  RUN_MODEL_DIR="${MODEL_STAGE_DIR:-$LOCAL_ROOT/llama3_3_70b}"
  mkdir -p "$RUN_MODEL_DIR"
  rsync -a --delete "$MODEL_DIR/" "$RUN_MODEL_DIR/"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ -n "${SLURM_JOB_ID:-}" && -n "${GPU_DEVICES:-}" ]]; then
  echo "Leave GPU_DEVICES empty under SLURM; use scheduler allocation." >&2
  exit 1
fi
allocated_devices="${GPU_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$allocated_devices" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "CUDA_VISIBLE_DEVICES is missing. Launch inside the site's GPU job environment (srun if required)." >&2
    exit 1
  fi
  allocated_devices=0,1,2,3
fi
IFS=',' read -r -a gpu_ids <<<"$allocated_devices"
if ((${#gpu_ids[@]} != 4)); then
  echo "Expected four allocated GPUs, got CUDA_VISIBLE_DEVICES=$allocated_devices" >&2
  exit 1
fi
pair_0="${gpu_ids[0]},${gpu_ids[1]}"
pair_1="${gpu_ids[2]},${gpu_ids[3]}"

pids=()
stop_workers() {
  trap - TERM INT EXIT
  if ((${#pids[@]})); then
    kill -TERM "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap stop_workers TERM INT EXIT

run_shard() {
  local shard="$1"
  local devices="$2"
  local extra_args=()
  if [[ -n "${MAX_SESSION_RECORDS:-}" ]]; then
    extra_args+=(--max-session-records "$MAX_SESSION_RECORDS")
  fi
  exec env CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
    "$REPO_DIR/scripts/translation_llama_full.py" run-shard \
    --output "$OUTPUT_DIR" \
    --model-path "$RUN_MODEL_DIR" \
    --shard "$shard" \
    --batch-size "$BATCH_SIZE" \
    --expected-gpus 2 \
    --max-memory-per-gpu "$MAX_MEMORY_PER_GPU" "${extra_args[@]}"
}

run_shard 0 "$pair_0" &
pids+=("$!")
sleep "${WORKER_START_DELAY:-45}"
run_shard 1 "$pair_1" &
pids+=("$!")

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done
pids=()
trap - TERM INT EXIT
exit "$rc"

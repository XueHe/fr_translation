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

: "${OUTPUT_DIR:?Set OUTPUT_DIR to the private checkpoint directory}"
: "${MODEL_DIR:?Set MODEL_DIR to the pinned Llama model directory}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-76GiB}"
STAGE_MODEL="${STAGE_MODEL:-1}"

if [[ ! -f "$OUTPUT_DIR/input_manifest.json" ]]; then
  echo "Missing checkpoint manifest: $OUTPUT_DIR/input_manifest.json" >&2
  exit 1
fi

if [[ ! -f "$MODEL_DIR/download_manifest.json" ]]; then
  echo "Missing model provenance file: $MODEL_DIR/download_manifest.json" >&2
  exit 1
fi

RUN_MODEL_DIR="$MODEL_DIR"
if [[ "$STAGE_MODEL" == "1" ]]; then
  LOCAL_ROOT="${JOBSCRATCH:-${TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}}"
  RUN_MODEL_DIR="$LOCAL_ROOT/llama3_3_70b"
  mkdir -p "$RUN_MODEL_DIR"
  rsync -a --delete "$MODEL_DIR/" "$RUN_MODEL_DIR/"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

allocated_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
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
  exec env CUDA_VISIBLE_DEVICES="$devices" "$PYTHON" \
    "$REPO_DIR/scripts/translation_llama_full.py" run-shard \
    --output "$OUTPUT_DIR" \
    --model-path "$RUN_MODEL_DIR" \
    --shard "$shard" \
    --batch-size "$BATCH_SIZE" \
    --expected-gpus 2 \
    --max-memory-per-gpu "$MAX_MEMORY_PER_GPU"
}

run_shard 0 "$pair_0" &
pids+=("$!")
sleep 45
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

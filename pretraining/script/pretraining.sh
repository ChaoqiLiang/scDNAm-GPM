#!/bin/bash
# Pretrain scDNAm-GPM from scratch.
#
# Usage (from the repository root):
#   bash pretraining/script/pretraining.sh
#
# Overridables:
#   NPROC_PER_NODE   GPUs per node (default: auto-detect, fallback 1)
#   MODEL_CONFIG     model config json   (default: pretraining/config/model/6mer_8mambalayers_model_config.json)
#   TRAINING_ARGS    training args json  (default: pretraining/config/pretraining/training_args_fp16.json)
#   MASTER_PORT      torchrun port       (default: 16866)

set -e

MODEL_CONFIG="${MODEL_CONFIG:-pretraining/config/model/6mer_8mambalayers_model_config.json}"
TRAINING_ARGS="${TRAINING_ARGS:-pretraining/config/pretraining/training_args_fp16.json}"
MASTER_PORT="${MASTER_PORT:-16866}"

if [[ -z "${NPROC_PER_NODE}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l)"
  else
    NPROC_PER_NODE=1
  fi
fi

# Per-rank Triton cache base (pretraining.py creates one subdir per rank).
# Set this on clusters whose shared filesystem is slow, e.g. a local /tmp.
export TRITON_CACHE_BASE="${TRITON_CACHE_BASE:-$(pwd)/pretraining/triton_cache}"

[[ -f "${MODEL_CONFIG}" ]]  || { echo "missing model config: ${MODEL_CONFIG}"; exit 1; }
[[ -f "${TRAINING_ARGS}" ]] || { echo "missing training args: ${TRAINING_ARGS}"; exit 1; }

echo "[LAUNCH] GPUs: ${NPROC_PER_NODE}, model: ${MODEL_CONFIG}, args: ${TRAINING_ARGS}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  pretraining/pretraining.py \
    --model_config_path "${MODEL_CONFIG}" \
    --training_args_path "${TRAINING_ARGS}"

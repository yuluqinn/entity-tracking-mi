#!/bin/bash
# Sequential fallback for GPUs in exclusive-process mode: waits until the `local` fleet's log
# shows JOB_END, then runs the listed fleets one after another on the same GPU.
#   bash scripts/correctonly_merged/run_interactive_seq.sh global mention   [TAG MODEL_TYPE MODEL_PATH N_LAYERS ENV via env vars]
set -u
TAG=${TAG:-codellama13b}; MODEL_TYPE=${MODEL_TYPE:-CodeLlama-13b-hf}; MODEL_PATH=${MODEL_PATH:-codellama/CodeLlama-13b-hf}; N_LAYERS=${N_LAYERS:-40}; ENV=${ENV:-nnsight_env}
module load miniconda 2>/dev/null; module load cuda/11.8 2>/dev/null
conda activate /projectnb/tin-lab/yuluq/$ENV
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
export WANDB_PROJECT=entity-tracking-probing
export TAG MODEL_TYPE MODEL_PATH N_LAYERS
LOCAL_LOG=logs/interactive_co_local_${TAG}.log
if [ -z "${NOWAIT:-}" ] && [ -f "$LOCAL_LOG" ] && ! grep -q JOB_END "$LOCAL_LOG"; then
  echo "waiting for the local fleet ($LOCAL_LOG) to finish ..."
  until grep -q JOB_END "$LOCAL_LOG"; do sleep 20; done
fi
for PROBE in "$@"; do
  export PROBE WANDB_TAGS=${TAG}-correctOnly-merged,${PROBE}
  echo "=== $PROBE start $(date)"
  bash <(sed -n '/^DATA=/,$p' scripts/correctonly_merged/train.qsub) > logs/interactive_co_${PROBE}_${TAG}.log 2>&1
  echo "=== $PROBE end $(date) (exit $?)"
done
echo "ALL DONE $(date)"

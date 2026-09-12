#!/bin/bash
# Run the three correctOnly-merged probe fleets CONCURRENTLY on one GPU inside an interactive
# session (qrsh). Same commands/log stamps as train.qsub, so the extractor works unchanged.
#   cd /projectnb/tin-lab/yuluq/entity-tracking-code/entity-tracking-mi
#   bash scripts/correctonly_merged/run_interactive.sh [TAG MODEL_TYPE MODEL_PATH N_LAYERS ENV]
set -u
TAG=${1:-codellama13b}; MODEL_TYPE=${2:-CodeLlama-13b-hf}; MODEL_PATH=${3:-codellama/CodeLlama-13b-hf}; N_LAYERS=${4:-40}; ENV=${5:-nnsight_env}
module load miniconda 2>/dev/null; module load cuda/11.8 2>/dev/null
conda activate /projectnb/tin-lab/yuluq/$ENV
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
export WANDB_PROJECT=entity-tracking-probing
mkdir -p logs
nvidia-smi --query-gpu=name --format=csv,noheader
for PROBE in local global mention; do
  (
    export WANDB_TAGS=${TAG}-correctOnly-merged,${PROBE}
    export TAG MODEL_TYPE MODEL_PATH N_LAYERS PROBE
    # reuse the batch script body (skip the #$ header lines and the module/conda setup it repeats)
    bash <(sed -n '/^DATA=/,$p' scripts/correctonly_merged/train.qsub)
  ) > logs/interactive_co_${PROBE}_${TAG}.log 2>&1 &
  echo "started $PROBE -> logs/interactive_co_${PROBE}_${TAG}.log (pid $!)"
done
wait
echo "ALL DONE $(date)"

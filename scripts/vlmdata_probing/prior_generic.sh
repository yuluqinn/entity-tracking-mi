#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_$JOB_NAME.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === Generic vlm-data prior-state pipeline (cache or train), parameterized ===
# -v vars: MODEL_TYPE, MODEL_PATH, N_HS, DATASET (chat-wrapped dir), SHORT,
#          MODE=save|train, PRIOR_STATE (train only, empty = local),
#          LAYER_START/LAYER_END (train only)
# Env is qwen3vl_env (all three VLMs); cache_prefix always used.

module load miniconda
module load cuda/11.8
conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=vlmdata-vlm3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
ROOT=$(pwd)/probe_experiments

if [ "$MODE" = "save" ]; then
  python probe_experiments/train_probe.py \
    --model_type $MODEL_TYPE --model_path $MODEL_PATH \
    --dataset_path $ROOT/../data/$DATASET \
    --layer $N_HS --epo 64 \
    --binary_probe --exclude_empty --condition_on the \
    --cache_prefix_field cache_prefix \
    --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/binary_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/$SHORT/exclude_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/vlm-data/objects.csv
else
  EXTRA=""
  if [ -n "$PRIOR_STATE" ]; then EXTRA="--num_prior_state $PRIOR_STATE"; fi
  for layer in $(seq ${LAYER_START:-1} ${LAYER_END:-$N_HS}); do
    python probe_experiments/train_probe.py \
      --model_type $MODEL_TYPE --model_path $MODEL_PATH \
      --dataset_path $ROOT/../data/$DATASET \
      --layer $layer --epo 64 \
      --binary_probe --exclude_empty --condition_on the \
      --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/binary_the \
      --load_model_representation \
      --model_representation_path probe_experiments/representations/$SHORT/exclude_empty_conditioned_on_the \
      --dataset_subset \
      --object_vocabulary_file data/vlm-data/objects.csv \
      $EXTRA
  done
fi

#!/bin/bash
# Mention probe with --standardize_inputs on the correctOnly-merged split, FINAL LAYER only, CPU, wandb-logged.
#   bash scripts/correctonly_merged/mention_std_final.sh <tag> <MODEL_TYPE> <MODEL_PATH> <layer> [env]
TAG=$1; MODEL_TYPE=$2; MODEL_PATH=$3; LAYER=$4; ENV=${5:-qwen3vl_env}
module load miniconda 2>/dev/null; conda activate /projectnb/tin-lab/yuluq/$ENV
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/" WANDB_PROJECT=entity-tracking-probing WANDB_TAGS=${TAG}-correctOnly-merged,mention,standardized
export CUDA_VISIBLE_DEVICES=""
echo "START $TAG layer $LAYER $(date +%s)"
python probe_experiments/train_probe.py \
  --model_type $MODEL_TYPE --model_path $MODEL_PATH \
  --dataset_path $(pwd)/data/orig-correctOnly-$TAG --dataset_subset \
  --layer $LAYER --epo 64 --mention --binary_probe --exclude_empty --condition_on the \
  --checkpoint_root probe_experiments/probe_checkpoints/$TAG-correctOnly-std/mention_the \
  --load_model_representation --model_representation_path probe_experiments/representations/$TAG-correctOnly/exclude_empty_conditioned_on_the \
  --object_vocabulary_file data/objects/llama_friendly_objects.csv --standardize_inputs
echo "END $TAG layer $LAYER $(date +%s) exit $?"

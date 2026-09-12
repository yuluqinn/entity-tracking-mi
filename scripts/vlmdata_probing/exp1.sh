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

# === Experiment-1 probes (global / mention) on vlm-data, parameterized ===
# Required -v vars:
#   ENVDIR   conda env (nnsight_env for CodeLlama, qwen3vl_env for Qwen)
#   MODEL_TYPE, MODEL_PATH, N_HS (40 CodeLlama / 37 Qwen)
#   DATASET  data dir under data/ (vlm-data-noinstr | vlm-data-instr | *-qwenchat)
#   SHORT    checkpoint/representation prefix (e.g. codellama-13b-vlmdata-noinstr)
#   MODE     save_global | global | mention
#   CACHEFIELD  optional: cache_prefix (instr CodeLlama; all Qwen)
# Caches: global uses include_empty_conditioned_on_the (built by MODE=save_global);
# mention reuses the existing exclude_empty cache.

module load miniconda
module load cuda/11.8
conda activate /projectnb/tin-lab/yuluq/$ENVDIR
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=vlmdata-exp1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
ROOT=$(pwd)/probe_experiments

CF_ARGS=""
if [ -n "$CACHEFIELD" ]; then CF_ARGS="--cache_prefix_field $CACHEFIELD"; fi

case "$MODE" in
  save_global)
    python probe_experiments/train_probe.py \
        --model_type $MODEL_TYPE --model_path $MODEL_PATH \
        --dataset_path $ROOT/../data/$DATASET \
        --layer $N_HS --epo 64 \
        --condition_on the --box_label_base 1 $CF_ARGS \
        --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/global_the \
        --save_model_representation \
        --model_representation_path probe_experiments/representations/$SHORT/include_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/vlm-data/objects.csv
    ;;
  global)
    for layer in $(seq 1 $N_HS); do
      python probe_experiments/train_probe.py \
        --model_type $MODEL_TYPE --model_path $MODEL_PATH \
        --dataset_path $ROOT/../data/$DATASET \
        --layer $layer --epo 64 \
        --condition_on the --box_label_base 1 \
        --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/global_the \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/$SHORT/include_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/vlm-data/objects.csv
    done
    ;;
  mention)
    for layer in $(seq 1 $N_HS); do
      python probe_experiments/train_probe.py \
        --model_type $MODEL_TYPE --model_path $MODEL_PATH \
        --dataset_path $ROOT/../data/$DATASET \
        --layer $layer --epo 64 \
        --mention --binary_probe --exclude_empty \
        --condition_on the --box_label_base 1 \
        --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/mention_the \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/$SHORT/exclude_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/vlm-data/objects.csv
    done
    ;;
esac

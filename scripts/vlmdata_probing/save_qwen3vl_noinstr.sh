#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_vlmdata_qwen3vl_noinstr_save.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === Qwen3-VL-8B on the user's vlm-data (no-instruction, CHAT-WRAPPED) ===
# Model input = cache_prefix (the exact chat template the vlm-entity-tracking
# behavioral runs used, byte-checked vs their meta.json); labels parse the
# canonical prefix/sentence fields. 37 hidden states, exclude_empty split.

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=vlmdata-noinstr-qwenchat
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

python probe_experiments/train_probe.py \
    --model_type Qwen3-VL-8B-Instruct \
    --dataset_path $ROOT/../data/vlm-data-noinstr-qwenchat \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --layer 37 \
    --epo 64 \
    --binary_probe \
    --exclude_empty \
    --condition_on the \
    --cache_prefix_field cache_prefix \
    --checkpoint_root probe_experiments/probe_checkpoints/qwen3-vl-8b-vlmdata-noinstr/binary_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/qwen3-vl-8b-vlmdata-noinstr/exclude_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/vlm-data/objects.csv \

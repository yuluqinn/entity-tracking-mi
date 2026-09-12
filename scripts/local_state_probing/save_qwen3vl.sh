#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_local_qwen3vl.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # fixes GLIBCXX_3.4.30 import error

export WANDB_PROJECT=entity-tracking-probing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# caching dirs (HF token is read from ~/.cache/huggingface/token)
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

# NOTE: --layer must be <= 37 for Qwen3-VL (36 decoder layers + embeddings);
# caching stores all 37 layers regardless.
python probe_experiments/train_probe.py \
    --model_type Qwen3-VL-8B-Instruct \
    --exp_name binary \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path Qwen/Qwen3-VL-8B-Instruct \
    --layer 36 \
    --epo 64 \
    --binary_probe \
    --exclude_empty \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/qwen3-vl-8b/binary_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/qwen3-vl-8b/exclude_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \

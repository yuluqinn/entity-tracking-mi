#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S  # torch 2.7.1+cu126 has no sm_120 kernels: avoid RTXP6000 (Blackwell) nodes
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_replication_codellama13b_save_global.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === CodeLlama-13B REPLICATION (paper baseline) — global-probe cache ===
# Caches last-token representations (all 41 layers) for the include_empty split.

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # fixes GLIBCXX_3.4.30 import error

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=codellama13b-replication
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

python probe_experiments/train_probe.py \
    --model_type CodeLlama-13b-hf \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path codellama/CodeLlama-13b-hf \
    --layer 40 \
    --epo 64 \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/codellama-13b-replication/global_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/codellama-13b-replication/include_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \

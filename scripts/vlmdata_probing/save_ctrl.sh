#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_vlmdata_codellama_noinstr_save.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === CodeLlama-13B on the user's vlm-data (no-instruction condition) ===
# Caches last-token representations (all 41 hidden states) for the exclude_empty
# split of data/${DATASET} (30/30 scene split, 60-object vocabulary).
# New representation/checkpoint paths: does not touch any existing results.

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=orig-scene-controls
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

python probe_experiments/train_probe.py \
    --model_type CodeLlama-13b-hf \
    --dataset_path $ROOT/../data/${DATASET} \
    --model_path codellama/CodeLlama-13b-hf \
    --layer 40 \
    --epo 64 \
    --binary_probe \
    --exclude_empty \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/${SHORT}/binary_the \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/${SHORT}/exclude_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \

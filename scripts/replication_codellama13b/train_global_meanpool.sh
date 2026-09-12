#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S  # torch 2.7.1+cu126 has no sm_120 kernels: avoid RTXP6000 (Blackwell) nodes
#$ -o logs/$JOB_ID_replication_codellama13b_train_global_meanpool.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === CodeLlama-13B MEAN-POOL EXPERIMENT — global state probes, layers 1-40 ===
# Trains on the mean-over-context representations; everything else identical to the
# last-token global replication (hyperparameters, splits, metrics).

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # fixes GLIBCXX_3.4.30 import error

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=codellama13b-meanpool
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments
for layer in {1..40}
do
    python probe_experiments/train_probe.py \
        --model_type CodeLlama-13b-hf \
        --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
        --model_path codellama/CodeLlama-13b-hf \
        --layer $layer \
        --epo 64 \
        --condition_on the \
        --pooling mean_context \
        --checkpoint_root probe_experiments/probe_checkpoints/codellama-13b-replication/global_the_meanpool \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/codellama-13b-replication/include_empty_meanpool_context \
        --dataset_subset \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \

done

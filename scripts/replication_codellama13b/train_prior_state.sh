#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=6:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S  # torch 2.7.1+cu126 has no sm_120 kernels: avoid RTXP6000 (Blackwell) nodes
#$ -o logs/$JOB_ID.$TASK_ID_replication_codellama13b_prior_state.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com
#$ -t 1-7

# === CodeLlama-13B PRIOR-STATE PROBES (paper's non_triv_avg_accuracy experiment) ===
# --num_prior_state -k: relabel each example with the query box's contents k-1 local
# operations BEFORE the final state (labels back-tracked by reversing ops), probing
# whether the last-token representation still encodes past states.
# Reuses the local-probe cache — no model forward passes.
# Fixes vs the released qsub: adds --object_vocabulary_file (its omission is a hard
# KeyError), real cache/checkpoint paths, drops the no-op -1 array element.

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # fixes GLIBCXX_3.4.30 import error

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=codellama13b-prior-state
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

index=$(($SGE_TASK_ID-1))
PRIOR_STATES=(-8 -7 -6 -5 -4 -3 -2)
PRIOR_STATE=${PRIOR_STATES[$index]}
ROOT=$(pwd)/probe_experiments

for LAYER in {1..40}
do
    python probe_experiments/train_probe.py \
        --model_type CodeLlama-13b-hf \
        --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
        --model_path codellama/CodeLlama-13b-hf \
        --layer $LAYER \
        --num_prior_state $PRIOR_STATE \
        --epo 64 \
        --binary_probe \
        --exclude_empty \
        --condition_on the \
        --checkpoint_root probe_experiments/probe_checkpoints/codellama-13b-replication/binary_the \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/codellama-13b-replication/exclude_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \

done

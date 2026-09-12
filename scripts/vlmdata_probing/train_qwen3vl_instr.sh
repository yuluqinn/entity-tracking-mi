#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -o logs/$JOB_ID_$JOB_NAME.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# === Qwen3-VL-8B probe training on data/vlm-data-instr-qwenchat ===
# Optional -v vars: PRIOR_STATE=-k, LAYER_START/LAYER_END (default 1..37)

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=vlmdata-instr-qwenchat
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

EXTRA_ARGS=""
if [ -n "$PRIOR_STATE" ]; then EXTRA_ARGS="$EXTRA_ARGS --num_prior_state $PRIOR_STATE"; fi

for layer in $(seq ${LAYER_START:-1} ${LAYER_END:-37})
do
    python probe_experiments/train_probe.py \
        --model_type Qwen3-VL-8B-Instruct \
        --dataset_path $ROOT/../data/vlm-data-instr-qwenchat \
        --model_path Qwen/Qwen3-VL-8B-Instruct \
        --layer $layer \
        --epo 64 \
        --binary_probe \
        --exclude_empty \
        --condition_on the \
        --checkpoint_root probe_experiments/probe_checkpoints/qwen3-vl-8b-vlmdata-instr/binary_the \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/qwen3-vl-8b-vlmdata-instr/exclude_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/vlm-data/objects.csv \
        $EXTRA_ARGS
done

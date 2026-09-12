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

# === CodeLlama-13B probe training on data/${DATASET} ===
# Optional -v vars:
#   PRIOR_STATE=-k   prior-state probing (omit for the current-state local probe)
#   LAYER_START / LAYER_END   layer range (default 1..40)
#   MAX_TRAIN=n      cap training examples (learning-curve runs)
#   CKPT_SUFFIX=x    extra checkpoint-root suffix (keeps learning-curve runs separate)

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export WANDB_TAGS=orig-scene-controls
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

EXTRA_ARGS=""
if [ -n "$PRIOR_STATE" ]; then EXTRA_ARGS="$EXTRA_ARGS --num_prior_state $PRIOR_STATE"; fi
if [ -n "$MAX_TRAIN" ]; then EXTRA_ARGS="$EXTRA_ARGS --max_train_data $MAX_TRAIN"; fi
if [ -n "$MENTION" ]; then EXTRA_ARGS="$EXTRA_ARGS --mention"; fi

for layer in $(seq ${LAYER_START:-1} ${LAYER_END:-40})
do
    python probe_experiments/train_probe.py \
        --model_type CodeLlama-13b-hf \
        --dataset_path $ROOT/../data/${DATASET} \
        --model_path codellama/CodeLlama-13b-hf \
        --layer $layer \
        --epo 64 \
        --binary_probe \
        --exclude_empty \
        --condition_on the \
        --checkpoint_root probe_experiments/probe_checkpoints/${SHORT}/${CKPT:-binary_the} \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/${SHORT}/exclude_empty_conditioned_on_the \
        --dataset_subset \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \
        $EXTRA_ARGS
done

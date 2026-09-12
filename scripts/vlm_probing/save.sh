#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_$JOB_NAME.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com

# Parameterized representation-caching job for text-only VLM probing.
# Required -v vars: MODEL_TYPE (key in _INPUT_DIMENSIONS/_VLM_REGISTRY),
#   MODEL_PATH (HF repo), SHORT (output dir name), COND (local|global),
#   SAVE_LAYER (any valid layer index, e.g. num_decoder_layers)
# Example:
#   qsub -N save_local_internvl35 -v MODEL_TYPE=InternVL3_5-8B,MODEL_PATH=OpenGVLab/InternVL3_5-8B,SHORT=internvl3_5-8b,COND=local,SAVE_LAYER=36 scripts/vlm_probing/save.sh

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

if [ "$COND" = "local" ]; then
    COND_ARGS="--binary_probe --exclude_empty"
    REP_DIR="exclude_empty_conditioned_on_the"
    CKPT="binary_the"
else
    COND_ARGS=""
    REP_DIR="include_empty_conditioned_on_the"
    CKPT="global_the"
fi

# optional -v POOLING=mean_context: mean over context tokens instead of last token
if [ -n "$POOLING" ] && [ "$POOLING" != "last" ]; then
    COND_ARGS="$COND_ARGS --pooling $POOLING"
    REP_DIR="${REP_DIR%_conditioned_on_the}_meanpool_context"
    CKPT="${CKPT}_meanpool"
fi

python probe_experiments/train_probe.py \
    --model_type $MODEL_TYPE \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --model_path $MODEL_PATH \
    --layer $SAVE_LAYER \
    --epo 64 \
    $COND_ARGS \
    --condition_on the \
    --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/$CKPT \
    --save_model_representation \
    --model_representation_path probe_experiments/representations/$SHORT/$REP_DIR \
    --dataset_subset \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv \

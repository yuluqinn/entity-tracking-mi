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

# Parameterized probe-training job (loads cached representations; no model load).
# Required -v vars: MODEL_TYPE, MODEL_PATH, SHORT, COND (local|global|mention),
#   N_HS (number of hidden states = decoder layers + 1; loop trains layers 1..N_HS)
# Example:
#   qsub -N train_mention_internvl35 -hold_jid <savejob> -v MODEL_TYPE=InternVL3_5-8B,MODEL_PATH=OpenGVLab/InternVL3_5-8B,SHORT=internvl3_5-8b,COND=mention,N_HS=37 scripts/vlm_probing/train.sh

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments

case "$COND" in
  local)
    COND_ARGS="--binary_probe --exclude_empty"
    REP_DIR="exclude_empty_conditioned_on_the"; CKPT="binary_the" ;;
  mention)
    COND_ARGS="--mention --binary_probe --exclude_empty"
    REP_DIR="exclude_empty_conditioned_on_the"; CKPT="mention_the" ;;
  *)
    COND_ARGS=""
    REP_DIR="include_empty_conditioned_on_the"; CKPT="global_the" ;;
esac

# optional -v POOLING=mean_context: mean over context tokens instead of last token
if [ -n "$POOLING" ] && [ "$POOLING" != "last" ]; then
    COND_ARGS="$COND_ARGS --pooling $POOLING"
    REP_DIR="${REP_DIR%_conditioned_on_the}_meanpool_context"
    CKPT="${CKPT}_meanpool"
fi

# optional -v PRIOR_STATE=-k: prior-state probing (labels back-tracked k-1 local ops);
# checkpoint folder gets an automatic _prior_state_-k suffix inside train_probe.py
if [ -n "$PRIOR_STATE" ]; then
    COND_ARGS="$COND_ARGS --num_prior_state $PRIOR_STATE"
fi

for layer in $(seq 1 $N_HS)
do
    python probe_experiments/train_probe.py \
        --model_type $MODEL_TYPE \
        --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
        --model_path $MODEL_PATH \
        --layer $layer \
        --epo 64 \
        $COND_ARGS \
        --condition_on the \
        --checkpoint_root probe_experiments/probe_checkpoints/$SHORT/$CKPT \
        --load_model_representation \
        --model_representation_path probe_experiments/representations/$SHORT/$REP_DIR \
        --dataset_subset \
        --object_vocabulary_file data/objects/llama_friendly_objects.csv \

done

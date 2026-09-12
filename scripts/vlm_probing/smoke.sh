#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=2:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_$JOB_NAME.log
#$ -j y

# Parameterized smoke test: cache 70/70 examples (all layers, last token), train one
# mid-layer local probe from the cache, assert cache shapes.
# Required -v vars: MODEL_TYPE, MODEL_PATH, SHORT, N_HS, HIDDEN, MID_LAYER

module load miniconda
module load cuda/11.8

conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export WANDB_PROJECT=entity-tracking-probing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"

ROOT=$(pwd)/probe_experiments
SMOKE_DIR=$SHORT-smoke

COMMON_ARGS="--model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --dataset_path $ROOT/../data/boxes_altAlways_default_maxop12_5k \
    --binary_probe --exclude_empty --condition_on the \
    --max_train_data 70 --max_test_data 70 \
    --checkpoint_root probe_experiments/probe_checkpoints/$SMOKE_DIR/binary_the \
    --model_representation_path probe_experiments/representations/$SMOKE_DIR/exclude_empty_conditioned_on_the \
    --dataset_subset \
    --object_vocabulary_file data/objects/llama_friendly_objects.csv"

echo "=== SMOKE $MODEL_TYPE 1/2: cache 70/70 ==="
python probe_experiments/train_probe.py $COMMON_ARGS --layer $MID_LAYER --epo 4 --save_model_representation || exit 1

echo "=== SMOKE $MODEL_TYPE 2/2: train mid-layer probe ==="
python probe_experiments/train_probe.py $COMMON_ARGS --layer $MID_LAYER --epo 4 --load_model_representation || exit 1

echo "=== verifying cache shapes ==="
python - <<EOF
import pickle
p = "probe_experiments/representations/$SMOKE_DIR/exclude_empty_conditioned_on_the/representations_train_subset.p"
rep = pickle.load(open(p, "rb"))
n_layers, shape = len(rep[0]), tuple(rep[0][0].shape)
print(f"examples: {len(rep)}; layers: {n_layers}; shape: {shape}; dtype: {rep[0][0].dtype}")
assert n_layers == $N_HS, f"expected $N_HS hidden states, got {n_layers}"
assert shape[-1] == $HIDDEN, f"expected hidden $HIDDEN, got {shape}"
print("SMOKE TEST PASSED: $MODEL_TYPE")
EOF

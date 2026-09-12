# Shared env block for the CodeLlama-13B path-patching replication. Source from the repo root.
module load miniconda
module load cuda/11.8
conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH  # fixes GLIBCXX_3.4.30 import error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
export HF_HUB_CACHE="/projectnb/tin-lab/yuluq/huggingface_cache/"   # where models--codellama--CodeLlama-13b-hf lives
export TOKENIZERS_PARALLELISM=false
ROOT=$(pwd)
MODEL="codellama/CodeLlama-13b-hf"
MODEL_NAME="codellama-13b"
DATAFILE="${ROOT}/data/boxes_altAlways_1put_moreObj_noEmpty/train-t5.jsonl"
# PREC=8bit -> --load_in_8bit (paper setting); PREC=bf16 -> --dtype bfloat16
PREC=${PREC:-8bit}
if [ "$PREC" = "8bit" ]; then PREC_FLAGS="--load_in_8bit"; else PREC_FLAGS="--dtype bfloat16"; fi
# TARGET=put -> put object (last object) ; TARGET=desc -> description object (all but last)
TARGET=${TARGET:-put}
if [ "$TARGET" = "put" ]; then OBJ_INDEX="-1"; SUBDIR="logp_lastObjOnly"; else OBJ_INDEX=":-1"; SUBDIR="logp_notLastObj"; fi
# Common data-filter args shared by discovery / eval / cross-eval
DATA_ARGS="--datafile ${DATAFILE} --ops_order put --query_ops_order put --counterfactual rand_obj_rand_query_id --num_query_object 2 --sort_query_objects"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "PREC=$PREC TARGET=$TARGET SUBDIR=$SUBDIR OBJ_INDEX=$OBJ_INDEX"

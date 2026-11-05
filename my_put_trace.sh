#!/bin/bash
# Put-circuit path patching for CodeLlama-13b (200 samples).
# Mirrors scripts/path_patching/run_path_patching_1put_lastObjOnly_codellama13b.qsub
# with the corrected script name (run_path_patching.py).
#
# Usage (inside a GPU session with the conda env activated):
#   bash my_put_trace.sh
# Re-running with the same output_dir resumes from the first unfinished group.

MODEL=codellama/CodeLlama-13b-hf
MODEL_NAME=codellama-13b
ROOT=/projectnb/tin-lab/yuluq/entity-tracking-code/entity-tracking-mi-fresh/entity-tracking-mi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${ROOT}/nnsight_patching_experiments" || exit 1

python run_path_patching.py \
  --model $MODEL \
  --datafile "${ROOT}/data/boxes_altAlways_1put_moreObj_noEmpty/train-t5.jsonl" \
  --batch_size 4 \
  --num_samples 200 \
  --ops_order "put" \
  --query_ops_order "put" \
  --output_dir "${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/logp_lastObjOnly" \
  --counterfactual "rand_obj_rand_query_id" \
  --success_filter true \
  --sort_query_objects \
  --num_query_object 2 \
  --score_source logp \
  --use_object_index "-1" \
  --load_in_8bit \
  2>&1 | tee "${ROOT}/put_trace.log"

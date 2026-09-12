#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 8
#$ -l gpus=1
#$ -l h_rt=1:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_diag.log
#$ -j y
# === Null-patch diagnostic: does the freeze machinery reproduce the clean baseline? qsub -v PREC=8bit|bf16 ===
source scripts/path_patching_codellama13b/common_env.sh
cd nnsight_patching_experiments
python diag_null_patch.py --model $MODEL $DATA_ARGS --batch_size 4 --num_samples ${NUM_SAMPLES:-8} \
  --success_filter true --score_source logp --use_object_index "$OBJ_INDEX" $PREC_FLAGS
echo "exit=$?"

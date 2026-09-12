#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 8
#$ -l gpus=1
#$ -l h_rt=48:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_discover.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com
# === Path patching circuit discovery (groups A->D) on CodeLlama-13B, 1put data, n=200 ===
# qsub -v TARGET=put,PREC=8bit scripts/path_patching_codellama13b/run_path_patching.sh
# qsub -v TARGET=desc,PREC=8bit scripts/path_patching_codellama13b/run_path_patching.sh
# Fixed version of scripts/path_patching/run_path_patching_1put_{lastObjOnly,notLastObj}_codellama13b.qsub
# (which call a nonexistent run_patching.py and have unfilled env placeholders).
# Per-group results are cached as pp_group{A..D}.npy, so resubmitting resumes after a wall-time kill.
source scripts/path_patching_codellama13b/common_env.sh
OUT="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/${SUBDIR}"
cd nnsight_patching_experiments
START=$(date +%s)
python run_path_patching.py --model $MODEL $DATA_ARGS \
  --batch_size ${BATCH_SIZE:-4} --num_samples 200 \
  --output_dir "$OUT" --success_filter true \
  --score_source logp --use_object_index "$OBJ_INDEX" $PREC_FLAGS
echo "exit=$? elapsed=$(( $(date +%s) - START ))s"
ls -la "$OUT"/n200/

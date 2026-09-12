#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 8
#$ -l gpus=1
#$ -l h_rt=4:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_smoke.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com
# === Smoke test + timing for path patching on CodeLlama-13B ===
# qsub -v PREC=8bit scripts/path_patching_codellama13b/smoke.sh ; qsub -v PREC=bf16 ...
# Runs all four groups on 8 examples; the tqdm rate in the log gives seconds/trace per precision.
source scripts/path_patching_codellama13b/common_env.sh
OUT="${ROOT}/outputs/nnsight_patch_1put_smoke/${MODEL_NAME}-${PREC}/${SUBDIR}"
cd nnsight_patching_experiments
START=$(date +%s)
python run_path_patching.py --model $MODEL $DATA_ARGS \
  --batch_size 4 --num_samples ${NUM_SAMPLES:-8} \
  --output_dir "$OUT" --success_filter true \
  --score_source logp --use_object_index "$OBJ_INDEX" $PREC_FLAGS
echo "exit=$? elapsed=$(( $(date +%s) - START ))s"
ls -la "$OUT"/n*/

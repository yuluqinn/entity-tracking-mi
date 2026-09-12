#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=1:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_dump.log
#$ -j y
# Dump the exact n=200 filtered path-patching dataset (same flags as run_path_patching.sh) to a CSV.
source scripts/path_patching_codellama13b/common_env.sh
cd nnsight_patching_experiments
python dump_pp_data.py --model $MODEL $DATA_ARGS --batch_size 4 --num_samples 200 --success_filter true \
  --score_source logp --use_object_index "$OBJ_INDEX" $PREC_FLAGS \
  --output_dir "${ROOT}/outputs/tmp_dump" --csv_out "${ROOT}/scratch/pp_codellama13b_dataset_n200.csv"
echo "exit=$?"

#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 4
#$ -l gpus=1
#$ -l h_rt=1:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_drop.log
#$ -j y
# Ablation: circuit with group C / D / both removed. qsub -v TARGET=put|desc,PREC=bf16
source scripts/path_patching_codellama13b/common_env.sh
CIRCUIT="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/${SUBDIR}/n200"
MEAN_CACHE="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/mean_activations_${PREC}"
cd nnsight_patching_experiments
python eval_circuits_drop_group.py --model $MODEL $DATA_ARGS --batch_size 4 --num_samples 100 \
  --circuit_root_path "$CIRCUIT" --mean_activation_cache_path "$MEAN_CACHE" $PREC_FLAGS
echo "exit=$?"

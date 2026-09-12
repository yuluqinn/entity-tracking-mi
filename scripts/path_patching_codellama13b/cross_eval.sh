#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 8
#$ -l gpus=1
#$ -l h_rt=24:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_cross.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com
# === PUT <-> DESCRIBE cross-group patching: swap one group (A..D) between the two circuits, both directions ===
# qsub -v PREC=8bit scripts/path_patching_codellama13b/cross_eval.sh
# Mirrors scripts/path_patching/run_circuit_cross_eval_1put_codellama13b.qsub with a shared mean-activation cache.
source scripts/path_patching_codellama13b/common_env.sh
PUT_CIRCUIT="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/logp_lastObjOnly/n200"
DESC_CIRCUIT="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/logp_notLastObj/n200"
MEAN_CACHE="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/mean_activations_${PREC}"
mkdir -p "$MEAN_CACHE"
cd nnsight_patching_experiments
for PATCH_GROUP in A B C D; do
  for DIRECTION in put_base desc_base; do
    if [ "$DIRECTION" = "put_base" ]; then C1=$PUT_CIRCUIT; C2=$DESC_CIRCUIT; else C1=$DESC_CIRCUIT; C2=$PUT_CIRCUIT; fi
    echo "===== CROSS base=${DIRECTION} patch_group=${PATCH_GROUP} ====="
    python eval_circuits_cross_group.py --model $MODEL $DATA_ARGS \
      --batch_size 4 --num_samples 100 \
      --circuit1_root_path "$C1" --circuit2_root_path "$C2" \
      --mean_activation_cache_path "$MEAN_CACHE" \
      --patch_group $PATCH_GROUP \
      --n_more_heads_per_group ${N_MORE:-0} $PREC_FLAGS
    echo "exit=$?"
  done
done

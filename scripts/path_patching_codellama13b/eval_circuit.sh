#!/bin/bash -l
#$ -P tin-lab
#$ -pe omp 8
#$ -l gpus=1
#$ -l h_rt=12:00:00
#$ -l gpu_c=8.0
#$ -l gpu_type=L40S
#$ -l gpu_memory=40G
#$ -o logs/$JOB_ID_pp_eval.log
#$ -j y
#$ -m e
#$ -M yuluqinn24@gmail.com
# === Circuit faithfulness eval (mean-ablation) for the PUT (TARGET=put) or DESCRIBE (TARGET=desc) circuit ===
# qsub -v TARGET=put,PREC=8bit scripts/path_patching_codellama13b/eval_circuit.sh
# Prints Model / Circuit / Random-circuit performance and Faithfulness to the log (parsed by scratch/pp_codellama13b_analysis.py).
# Mean activations are cached once in a dir shared with cross_eval.sh.
source scripts/path_patching_codellama13b/common_env.sh
CIRCUIT="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/${SUBDIR}/n200"
MEAN_CACHE="${ROOT}/outputs/nnsight_patch_1put/${MODEL_NAME}/mean_activations_${PREC}"
mkdir -p "$MEAN_CACHE"
cd nnsight_patching_experiments
python eval_circuits.py --model $MODEL $DATA_ARGS \
  --batch_size 4 --num_samples 100 \
  --circuit_root_path "$CIRCUIT" \
  --mean_activation_cache_path "$MEAN_CACHE" \
  --n_more_heads_per_group ${N_MORE:-0} $PREC_FLAGS
echo "exit=$?"

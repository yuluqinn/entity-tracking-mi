#!/bin/bash
# Full correctOnly-merged pipeline for one VLM, sequentially on ONE GPU (interactive session):
#   bf16 0-shot inference on the 5,236 train-origin rows -> build (merge/behavioral/filter/split/cache re-index/verify)
#   -> local -> global -> mention probe fleets.  Every step is resumable (inference appends; build is idempotent;
#   probe layers with tensorboard.txt are skipped), so re-running the same command after an interruption is safe.
#   cd /projectnb/tin-lab/yuluq/entity-tracking-code/entity-tracking-mi
#   bash scripts/correctonly_merged/run_vlm_pipeline.sh qwen3vl|internvl35|molmo2|llava   [more models...]
set -u
module load miniconda 2>/dev/null; module load cuda/11.8 2>/dev/null
conda activate /projectnb/tin-lab/yuluq/qwen3vl_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="/projectnb/tin-lab/yuluq/transformer_cache/"
export WANDB_PROJECT=entity-tracking-probing
OUT_DIR=probe_experiments/results/boxes_altAlways_default_maxop12_5k/baseline_inference
TRAIN_ROWS=data/boxes_altAlways_default_maxop12_5k/train-subsample-states-gpt.jsonl
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

for M in "$@"; do
  case $M in
    qwen3vl)    TAG=qwen3vl;    MODEL_TYPE=Qwen3-VL-8B-Instruct;           MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct;                 SHORT=qwen3-vl-8b;         N_LAYERS=37 ;;
    internvl35) TAG=internvl35; MODEL_TYPE=InternVL3_5-8B;                 MODEL_PATH=OpenGVLab/InternVL3_5-8B;                  SHORT=internvl3_5-8b;      N_LAYERS=37 ;;
    molmo2)     TAG=molmo2;     MODEL_TYPE=Molmo2-8B;                      MODEL_PATH=allenai/Molmo2-8B;                         SHORT=molmo2-8b;           N_LAYERS=37 ;;
    llava)      TAG=llava;      MODEL_TYPE=llava-onevision-qwen2-7b-ov-hf; MODEL_PATH=llava-hf/llava-onevision-qwen2-7b-ov-hf;   SHORT=llava-onevision-7b;  N_LAYERS=29 ;;
    *) echo "unknown model key $M"; exit 1 ;;
  esac
  INF=$(basename $MODEL_PATH); LOG=logs/interactive_co_pipeline_${TAG}.log
  export TAG MODEL_TYPE MODEL_PATH N_LAYERS
  echo "PIPELINE_START $M $(date +%s) $(date)" | tee -a $LOG

  # 1. behavioral inference on train-origin rows (skips if already complete)
  INF_FILE=$OUT_DIR/inference_${INF}_train-subsample-states-gpt.jsonl
  if [ -f "$INF_FILE" ] && [ "$(wc -l < "$INF_FILE")" -ge 5236 ]; then
    echo "inference already complete" | tee -a $LOG
  else
    echo "INFER_START $(date +%s)" | tee -a $LOG
    python probe_experiments/inference.py --model_dir $MODEL_PATH --data_path $TRAIN_ROWS \
      --object_vocabulary_file data/objects/llama_friendly_objects.csv --output_dir $OUT_DIR --batch_size 16 \
      >> logs/interactive_co_infer_${TAG}.log 2>&1 || { echo "INFER_FAIL" | tee -a $LOG; exit 1; }
    echo "INFER_END $(date +%s)" | tee -a $LOG
  fi

  # 2. build filtered dataset + re-indexed caches
  if [ -f scratch/correctonly_merged/build_${TAG}.json ]; then
    echo "build already complete" | tee -a $LOG
  else
    echo "BUILD_START $(date +%s)" | tee -a $LOG
    python scratch/correctonly_merged/build_dataset.py $TAG $SHORT $INF >> logs/interactive_co_build_${TAG}.log 2>&1 || { echo "BUILD_FAIL" | tee -a $LOG; exit 1; }
    echo "BUILD_END $(date +%s)" | tee -a $LOG
  fi

  # 3. probes, one fleet at a time (GPU is exclusive-process)
  for PROBE in local global mention; do
    export PROBE WANDB_TAGS=${TAG}-correctOnly-merged,${PROBE}
    echo "FLEET_START $PROBE $(date +%s)" | tee -a $LOG
    bash <(sed -n '/^DATA=/,$p' scripts/correctonly_merged/train.qsub) > logs/interactive_co_${PROBE}_${TAG}.log 2>&1 \
      || { echo "FLEET_FAIL $PROBE" | tee -a $LOG; exit 1; }
    echo "FLEET_END $PROBE $(date +%s)" | tee -a $LOG
  done
  echo "PIPELINE_END $M $(date +%s) $(date)" | tee -a $LOG
done
echo "ALL DONE $(date)"

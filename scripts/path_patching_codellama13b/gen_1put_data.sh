#!/bin/bash -l
# === Generate the 1put dataset used by the PUT / DESCRIBE path-patching experiment ===
# Identical to dataset_generation/boxes_AltForm_1put_moreObj.sh except the output dir is
# named as the path-patching qsubs expect (boxes_altAlways_1put_moreObj_noEmpty).
# CPU only, no GPU needed. Run from the repo root:  bash scripts/path_patching_codellama13b/gen_1put_data.sh
set -euo pipefail
module load miniconda
conda activate /projectnb/tin-lab/yuluq/nnsight_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd "$(dirname "$0")/../../dataset_generation"
python generate_boxes_data_modified.py \
    --object_vocabulary_file "../data/objects/llama_friendly_objects.csv" \
    --output_dir "../data/boxes_altAlways_1put_moreObj_noEmpty" \
    --alternative_forms "always" \
    --num_samples 50000 \
    --allowed_operations "put" \
    --num_operations 1 \
    --fix_object_count_per_phrase 1 \
    --expected_num_items_per_box 2 \
    --max_items_per_box 4

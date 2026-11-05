# Reproducibility notes: CodeLlama-13b put / describe circuit replication

This branch (`replication-deterministic`) replicates the path-patching results for the
`Put` and `Describe` circuits on CodeLlama-13b and adds a five-seed robustness study.
All commands below are run from the repository root unless stated otherwise.

## Code

- Base: upstream commit `21b1304` (PootieT/entity-tracking-mi, main).
- Fixes needed to run the public code as shipped:
  - `nnsight_patching_experiments/patch_utils.py`: removed a stale `stupid_pad` import
    (the function does not exist in `utils.py`).
  - `scripts/path_patching/run_path_patching_1put_lastObjOnly_codellama13b.qsub`:
    `run_patching.py` -> `run_path_patching.py`.
- Deterministic data generation (commit `<fill in after committing>`):
  `dataset_generation/generate_boxes_data_modified.py` sampled objects from Python
  `set`s via `list(set)`. Python randomizes string hashing per process, so the set
  iteration order, and therefore the sampled objects, differed on every run even with
  `--seed` fixed. All sampling and serialisation sites now use `sorted(set)`.
  With this change the same `--seed` produces byte-identical output across processes
  (verified for the 1put configuration and the default multi-operation configuration).
  Datasets generated before this fix cannot be regenerated.

## Environment

- `environment.yaml` in this repository (Python 3.10, torch 2.7.1, transformers 4.53.2,
  nnsight 0.5.12, bitsandbytes 0.47.0).
- Model: `codellama/CodeLlama-13b-hf`, loaded in 8-bit (`--load_in_8bit`).
- Hardware for the seed study: one 48 GB GPU per run (NVIDIA L40S requested; the run is
  bound by single-core CPU speed, not by the GPU).

## Datasets for the five-seed study

Generated from `dataset_generation/` with:

```bash
python generate_boxes_data_modified.py \
    --object_vocabulary_file ../data/objects/llama_friendly_objects.csv \
    --output_dir ../data/boxes_altAlways_1put_moreObj_noEmpty_seed${SEED} \
    --seed ${SEED} \
    --alternative_forms always --num_samples 50000 \
    --allowed_operations put --num_operations 1 \
    --fix_object_count_per_phrase 1 \
    --expected_num_items_per_box 2 --max_items_per_box 4
```

Seeds were fixed in advance as 1..5. Each file has 700,000 records (50,000 worlds x
7 boxes x 2 query points), one object per box, one Put per world, no empty boxes, and
no sentence overlap between seeds.

| Seed | Directory (under `data/`)                    | md5sum of `train-t5.jsonl`         |
|-----:|----------------------------------------------|------------------------------------|
| 1    | `boxes_altAlways_1put_moreObj_noEmpty_seed1` | `6444555730349eee7c61628a616cce7e` |
| 2    | `boxes_altAlways_1put_moreObj_noEmpty_seed2` | `285313675777a8d8e1f9213c7362f383` |
| 3    | `boxes_altAlways_1put_moreObj_noEmpty_seed3` | `2cd3ee5b2e69ed8587f91b3bcaed40ae` |
| 4    | `boxes_altAlways_1put_moreObj_noEmpty_seed4` | `1b73a19829a0f75242a1c635e6746296` |
| 5    | `boxes_altAlways_1put_moreObj_noEmpty_seed5` | `afeac6eea5ea45f2b04af5f9fc22d4be` |

Verify a regenerated file with `md5sum data/<dir>/train-t5.jsonl`.

## Path patching (circuit discovery)

One array job per circuit; SGE task id = data seed. The patching-side seed
(`--seed 42`, default) is fixed for all runs; only the data seed varies.

```bash
mkdir -p logs
qsub scripts/path_patching/run_path_patching_1put_seeds_codellama13b.qsub put
qsub scripts/path_patching/run_path_patching_1put_seeds_codellama13b.qsub describe
```

- `put`      -> `--use_object_index "-1"`  (score on the put object)      -> `logp_lastObjOnly`
- `describe` -> `--use_object_index ":-1"` (score on the description object) -> `logp_notLastObj`

Outputs: `outputs/nnsight_patch_1put/codellama-13b/seed${SEED}/<subdir>/n200/pp_group{A,B,C,D}.npy`
(one score per attention head, rows = layers, columns = heads; negative = head matters).
Groups are discovered sequentially (B depends on A's knee-selected heads, etc.). A killed
run resumes from the first group without a saved `.npy` when resubmitted unchanged.

## Evaluation (per seed and circuit)

Faithfulness of the knee-selected circuit, from `nnsight_patching_experiments/`:

```bash
python eval_circuits.py --model codellama/CodeLlama-13b-hf \
  --datafile ../data/boxes_altAlways_1put_moreObj_noEmpty_seed${SEED}/train-t5.jsonl \
  --batch_size 4 --num_samples 100 --ops_order put --query_ops_order put \
  --counterfactual rand_obj_rand_query_id --num_query_object 2 --sort_query_objects \
  --circuit_root_path ../outputs/nnsight_patch_1put/codellama-13b/seed${SEED}/<subdir>/n200 \
  --mean_activation_cache_path ../outputs/nnsight_patch_1put/codellama-13b/seed${SEED}/<subdir>/n200 \
  --load_in_8bit
```

Cross-group patching (put <-> describe) follows
`scripts/path_patching/run_circuit_cross_eval_1put_codellama13b.qsub` with the per-seed paths.
Results are printed to stdout only; keep the logs.

## Earlier runs kept for reference (not part of the seed family)

- `outputs/nnsight_patch_1put/codellama-13b/{logp_lastObjOnly,logp_notLastObj}/n200`:
  200-sample traces on `data/boxes_AltForm_1put_moreObj` (generated 2026-09-09 with the
  pre-fix generator; not regenerable). Public code plus the two run fixes above.
- The August 2026 run in the older working copy (branch `old-replication-snapshot`) used
  additional local edits (`use_cache=False` on the clean pass, nnsight 0.7) and a different
  pre-fix dataset. Its scores carry a small constant offset relative to the public code.

# CodeLlama-13B PUT / DESCRIBE circuit replication (path patching)

Replicates `nnsight_patching_experiments/README.md` "Path Patching to find Put Circuit" +
faithfulness + cross-patching for `codellama/CodeLlama-13b-hf`, i.e. the paper figure
`assets/put_description_circuit_and_result.png`.

## Pipeline (run everything from the repo root)

| step | script | notes |
|---|---|---|
| 1 | `bash scripts/path_patching_codellama13b/gen_1put_data.sh` | CPU, ~1 min. Writes `data/boxes_altAlways_1put_moreObj_noEmpty/train-t5.jsonl` (700k rows; 50k scenes x 2 states x 7 boxes) |
| 2 | `qsub -v PREC=8bit smoke.sh` / `qsub -v PREC=bf16 smoke.sh` | 8 examples, all 4 groups; timing + nnsight-0.7 compatibility check |
| 3 | `qsub -v TARGET=put,PREC=8bit run_path_patching.sh` and `TARGET=desc` | circuit discovery, n=200, writes `outputs/nnsight_patch_1put/codellama-13b/{logp_lastObjOnly,logp_notLastObj}/n200/pp_group{A..D}.npy` |
| 4 | `qsub -v TARGET=put,PREC=8bit eval_circuit.sh` and `TARGET=desc` | faithfulness (mean-ablation), n=100, stdout in `logs/<job>_pp_eval.log` |
| 5 | `qsub -v PREC=8bit cross_eval.sh` | swap each group A..D between the two circuits, both directions |
| 6 | `python scratch/pp_codellama13b_analysis.py` | head lists + knee info, overlap coefficients, parses logs, figure |

Fixes relative to the released `scripts/path_patching/*codellama13b*.qsub`: correct script name
(`run_path_patching.py`, not `run_patching.py`), filled-in env, single shared mean-activation cache,
`--dtype` flag added to `patch_utils.py` for the bf16 option, stale `stupid_pad` import -> `force_pad`.

## Precision decision (smoke test, 2026-08-25, L40S, batch 4, prompt len 70 tok)

| PREC | jobs | group-A rate | s/trace | projected n=200 per circuit |
|---|---|---|---|---|
| 8bit (bnb int8, fp16 compute) | 7308643 | 2.87 heads/s | ~0.17 | A ~3.9 h, B/C/D <= 7.7 h each -> <= ~27 h |
| bf16 | 7308645 | 4.71 heads/s | ~0.11 | A ~2.4 h, B/C/D <= 4.7 h each -> <= ~15 h |

(group A = 1 trace per batch; B/C/D = 2 traces per batch; B/C/D sender layers only run up to the
max receiver layer, so their real cost is lower than the bound.)
Initial decision was 8-bit (paper setting); the user then chose **bf16** for speed (2026-08-25 13:20). The 8-bit chain
(7308894-98) was killed before any group finished and the whole chain resubmitted in bf16: discovery 7309003 (PUT) /
7309004 (DESC), eval 7309005 / 7309006, cross 7309007. Deviation from the paper: bf16 weights instead of bnb int8
(changes which examples pass the success filter slightly; earlier 0-shot runs: ~1 pt behavioral difference).

## Baseline-offset bug found and fixed (2026-08-25)

Smoke runs showed the *bulk* of all 1600 sender heads shifted together (median score bf16 -0.005..-0.010,
8-bit +0.005 (A) / +0.029 (B, C)), i.e. a systematic offset, not per-head signal. `diag_null_patch.py`
(`scripts/path_patching_codellama13b/diag.sh`, jobs 7308878 8-bit / 7308879 bf16) isolated it: every patched
pass uses `model.trace(..., use_cache=False)` while the clean baseline in `cache_logit_and_hidden` used the
default `use_cache=True`; the two HF attention paths differ numerically by up to 0.15 nats/example (mean +0.033,
8-bit). A null patch through the freeze machinery reproduced the baseline to 2e-4 once the setting matched.
Fix: `use_cache=False` in `cache_logit_and_hidden` (run_path_patching.py).
Verification (post-fix null patch, n=8): 8-bit job 7308893 `b/c/d` mean diff +0.00005, max 0.0009 nats
(score std 0.00025); bf16 job 7308879 max 0.006 nats (diag-only float32-vs-bf16 log_softmax residual). All discovery outputs produced before
the fix were deleted; chain resubmitted as jobs 7308893 (verification diag), 7308894/7308895 (discovery),
7308896/7308897 (eval), 7308898 (cross).

## nnsight 0.7 compatibility fix for the eval scripts (2026-08-26)
`eval_circuits.py` / `eval_circuits_cross_group.py` crashed with `module 'nnsight' has no attribute 'apply'` (job 7309005):
nnsight>=0.6 executes the trace body on real tensors, so `nnsight.apply(fn, ...)` became a plain call. Added `_nn_apply`
shim in `utils.py` (3 call sites: mean-activation caching x2, `mean_ablate` in `eval_circuit_performance`). Eval chain
resubmitted: 7315799 (eval PUT) -> 7315800 (eval DESC) -> 7315801 (cross). From the failed run: full-model accuracy on the
PUT eval set = 0.94 (argmax-any); argmax by label index [desc 0.60, put 0.34].

## Discovery results (bf16, n=200; jobs 7309003 PUT 12.3 h / 7309004 DESC 9.9 h)
Knee-selected group sizes PUT A/B/C/D = 47/41/35/44, DESC = 40/27/23/15. Overlap coefficient A 0.30, B 0.15, C 0.04, D 0.27
(paper: ~0.25/0.25/0.07/~0). DESC has crisp signal at every group (B (15,7) -0.20, C (12,20) -0.08, D (12,20) -0.11);
PUT's C and D are essentially noise (strongest -0.011 / -0.004 vs a +-0.002 noise band) -> PUT C/D head lists are not
meaningful; the put object appears to be read by A/B without the C->D positional chain. Heatmaps:
`scratch/pp_codellama13b_heatmap_{put,desc}.png`.

## Results (bf16, eval n=100; jobs 7315799 eval PUT, 7315800 eval DESC, 7315801 cross)

Full-model accuracy on the eval set (argmax = either label): 0.94 (by label index: desc 0.60, put 0.34).

| circuit | full circuit acc / faithfulness | random circuit | LOO A | LOO B | LOO C | LOO D |
|---|---|---|---|---|---|---|
| PUT  | 0.84 / **0.89** | 0.00 | 0.30 | 1.01 | 0.70 | 0.90 |
| DESC | 0.47 / **0.50** | 0.00 | 0.40 | 0.14 | 0.61 | 0.57 |

("LOO X" = faithfulness of the circuit with group X replaced by the other circuit's group X, re-cut to the same size.)
Overlap coefficient per group (desc vs put knee lists): A 0.30, B 0.15, C 0.04, D 0.27 (paper ~0.25/0.25/0.07/~0).

Reading: PUT circuit faithful (0.89, paper ~0.9) and put-specific (ablated circuit predicts put obj 0.84, desc obj 0.00);
DESC circuit ~0.5 (paper ~0.5). Groups A/B/C/D descend in layers as in the entity-binding model for DESC; for PUT only
A/B carry signal (C/D patch scores at noise level) and LOO D = 0.90 confirms D is irrelevant for the put target.
B is interchangeable in the PUT direction (1.01); the reverse direction (0.14) is confounded by donor re-sizing (PUT's
top-27 B heads replace DESC's 27, while DESC's top-41 replace PUT's 41). DESC<-PUT-C (0.61) flips predictions toward the
put object (desc 0.19 / put 0.38): the "either label" faithfulness metric counts that as correct, so per-label numbers
in `scratch/pp_codellama13b_results.json` should be reported alongside. n=100 -> binomial SE ~0.05.
Figure: `scratch/pp_codellama13b_circuit_results.png`; heatmaps `scratch/pp_codellama13b_heatmap_{put,desc}.png`.

## Follow-up: dropping groups C / D (2026-08-26, jobs 7318504 PUT / 7318505 DESC, `drop_group.sh` -> `eval_circuits_drop_group.py`)
Same mean-ablation eval, with group C, group D or both removed (nothing kept at those positions).

| kept | PUT acc / faith (desc/put) | DESC acc / faith (desc/put) |
|---|---|---|
| A,B,C,D | 0.84 / 0.89 (0.00/0.84) | 0.47 / 0.50 (0.39/0.08) |
| A,B,C (no D) | 0.84 / 0.89 (0.00/0.84) | 0.48 / 0.51 (0.43/0.05) |
| A,B,D (no C) | 0.63 / 0.67 (0.03/0.60) | 0.27 / 0.29 (0.22/0.05) |
| A,B only | 0.62 / 0.66 (0.03/0.59) | 0.21 / 0.22 (0.18/0.03) |

Findings: (1) group D is dispensable under mean ablation in BOTH circuits (even DESC, whose D had a crisp path-patching
signal); (2) the C position (query box id) matters for both (-0.2 when dropped), so PUT's C stage is real but distributed
over many individually weak heads (single-head patching under-detects it; the knee list of 35 contains 9 heads > 5 sigma).
Caveat: no control with 35 random heads kept at the C position; A+B alone still give PUT 0.66 faithfulness.
Noise analysis: `scratch/pp_codellama13b_knee_curves.png`, `scratch/pp_kneedle_walkthrough.png`; Kneedle on Gaussian noise
shaped like PUT-D returns cuts of 9-59 heads (real PUT-D cut: 44). Dataset CSV: `scratch/pp_codellama13b_dataset_n200.csv`.

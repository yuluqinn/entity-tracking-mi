"""Per-example circuit evaluation on a fixed evaluation set (CodeLlama-13B, single-PUT task).

For a fixed set of examples and one fixed mean-activation table, run several conditions
(full model, sanity checks, discovered circuits, size-matched circuits, random circuits) with
the exact mean-ablation code path of eval_circuits.py (utils.mean_ablate inside model.trace),
and record for every example and condition the prediction-level outcome for BOTH target
objects (DESCRIPTION object = label 0, PUT object = label 1): log probability, vocabulary rank,
rank among the objects mentioned in the prompt, rank among the 100 vocabulary objects, argmax
and top-2 membership, the top-5 tokens, and the KL divergence from the full model.

Example sets (both drawn from the same data file with the same loader as the circuit scripts):
  fixed : first --n_fixed first-token-correct candidates with candidate rank >= --fixed_start_rank
          (first-token correctness is read from the behavioral map jsonl of this data file)
  repro : the first --n_repro candidates, unfiltered (what eval_circuits.py evaluated)

Run from the repository root (see scratch/run_per_example_eval_codellama13b.qsub).
"""
import argparse
import csv
import json
import operator
import os
import re
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import nnsight
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "nnsight_patching_experiments"))
from utils import (  # noqa: E402
    compute_topk_components_knee, fix_random_seed, get_circuit, get_mean_activations, get_module,
    get_multi_object_phrase_spans, get_random_circuit, mean_ablate, pad_batch_collate_fn, rsetattr,
)
from patch_utils import build_parser, get_model_and_dataset, post_arg_parse_fix  # noqa: E402

KNEE_KW = {"S": 1.0, "curve": "convex", "direction": "decreasing"}
GROUPS = ["A", "B", "C", "D"]


def add_args(parser):
    parser.add_argument("--map_jsonl", required=True, help="behavioral map jsonl for this data file (first-token flags per candidate)")
    parser.add_argument("--mean_activation_cache_path", required=True, help="directory holding mean_activations.pkl (shared by all conditions)")
    parser.add_argument("--circuit", action="append", default=[], help="name=path_to_n200_dir; repeatable")
    parser.add_argument("--size_match", action="append", default=[], help="name=scores_circuit:sizes_circuit (cut the first circuit's scores at the second's knee sizes); repeatable")
    parser.add_argument("--random_like", action="append", default=[], help="circuit_name:count (random circuits with that circuit's effective group sizes); repeatable")
    parser.add_argument("--keep_nonvital", action="append", default=[], help="circuit_name: same heads at the 4 vital positions, but non-vital positions left intact (ablate_non_vital_pos=False); repeatable")
    parser.add_argument("--topk", action="append", default=[], help="name=scores_circuit:kA,kB,kC,kD (top-k most-negative heads per group from that circuit's score files; a very large k = all negative-score heads); repeatable")
    parser.add_argument("--fixed_start_rank", type=int, default=1000)
    parser.add_argument("--n_fixed", type=int, default=150)
    parser.add_argument("--n_repro", type=int, default=100)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--object_vocabulary_file", default=str(REPO / "data" / "objects" / "llama_friendly_objects.csv"))
    return parser


# ----------------------------------------------------------------------------- circuits
def knee_heads(root):
    """Knee-selected heads per group, exactly as utils.get_circuit computes them (before A∩B removal)."""
    out = {}
    for g in GROUPS:
        arr = torch.tensor(np.load(f"{root}/pp_group{g}.npy"))
        out[g] = [tuple(h) for h in compute_topk_components_knee(arr, largest=False, knee_kwargs=KNEE_KW)]
    return out


def topk_heads(root, sizes):
    """Most-negative-k heads per group from the same score files (same ordering the knee uses)."""
    out = {}
    for g in GROUPS:
        flat = np.load(f"{root}/pp_group{g}.npy")
        n_heads = flat.shape[1]
        flat = flat.ravel()
        order = np.argsort(flat)  # most negative first
        order = [i for i in order if flat[i] < 0][: sizes[g]]
        out[g] = [(int(i // n_heads), int(i % n_heads)) for i in order]
    return out


def build_components(model, heads):
    """Mirror utils.get_circuit: A∪B at the last position (0), C at query-box position (2), D at previous query-box positions (-1)."""
    comps = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}
    A = [h for h in heads["A"] if h not in set(heads["B"])]
    for pos, hs in ((0, A + list(heads["B"])), (2, heads["C"]), (-1, heads["D"])):
        for layer_idx, head in hs:
            comps[pos][get_module(model, layer_idx)].append(head)
    eff = {"A": len(A), "B": len(heads["B"]), "C": len(heads["C"]), "D": len(heads["D"])}
    return comps, eff


def all_heads_components(model):
    comps = {0: defaultdict(list), 2: defaultdict(list), -1: defaultdict(list)}
    for layer_idx in range(model.config.num_hidden_layers):
        for pos in comps:
            comps[pos][get_module(model, layer_idx)] = list(range(model.config.num_attention_heads))
    return comps


def components_equal(c1, c2):
    for pos in (0, 2, -1):
        k = set(c1[pos].keys()) | set(c2[pos].keys())
        for layer in k:
            if sorted(set(c1[pos].get(layer, []))) != sorted(set(c2[pos].get(layer, []))):
                return False
    return True


# ----------------------------------------------------------------------------- evaluation
def forward_logits(model, input_ids, modules, components, mean_activations, last_pos, ablate_non_vital_pos):
    """Last-layer logits under mean ablation; components=None means no intervention (plain trace)."""
    with model.trace(input_ids) as tracer:
        if components is not None:
            object_spans_batch = get_multi_object_phrase_spans(input_ids, model.tokenizer)
            for layer in modules:
                getter = operator.attrgetter(layer)
                original_input = getter(model).input.clone()
                ablated_input = nnsight.apply(
                    mean_ablate, original_input, layer, model, components, mean_activations,
                    input_ids, last_pos, ablate_non_vital_pos, object_spans_batch,
                )
                rsetattr(model, f"{layer}.input", ablated_input)
        logits = model.lm_head.output.save()
    return logits


def describe_prediction(logprobs, ex, tokenizer, vocab_ids, full_logprobs=None):
    """logprobs: [vocab] log-softmax at the last position for one example."""
    desc_id, put_id = ex["desc_id"], ex["put_id"]
    order = torch.argsort(logprobs, descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(order), device=order.device)
    top5 = order[:5].tolist()
    ctx_ids = ex["context_ids"]  # 8 objects mentioned in the prompt (7 initial + put object)
    ctx_lp = logprobs[ctx_ids]
    voc_lp = logprobs[vocab_ids]

    def ctx_rank(tid):  # 1-based rank among prompt objects
        return int((ctx_lp > logprobs[tid]).sum().item()) + 1

    def voc_rank(tid):  # 1-based rank among the 100 vocabulary objects
        return int((voc_lp > logprobs[tid]).sum().item()) + 1

    am = top5[0]
    if am == desc_id:
        am_type = "description"
    elif am == put_id:
        am_type = "put"
    elif am in ctx_ids:
        am_type = "other_context_object"
    elif am in set(vocab_ids):
        am_type = "other_vocab_object"
    else:
        am_type = "non_object"
    rec = {
        "argmax_token": tokenizer.decode([am]), "argmax_type": am_type,
        "first_token_correct": int(am in (desc_id, put_id)),
        "desc_logprob": float(logprobs[desc_id]), "put_logprob": float(logprobs[put_id]),
        "desc_vocab_rank": int(rank[desc_id]) + 1, "put_vocab_rank": int(rank[put_id]) + 1,
        "desc_context_rank": ctx_rank(desc_id), "put_context_rank": ctx_rank(put_id),
        "desc_object_rank": voc_rank(desc_id), "put_object_rank": voc_rank(put_id),
        "desc_in_top2": int(desc_id in order[:2].tolist()), "put_in_top2": int(put_id in order[:2].tolist()),
        "desc_minus_put_logprob": float(logprobs[desc_id] - logprobs[put_id]),
        "top5_tokens": [tokenizer.decode([t]) for t in top5], "top5_logprobs": [float(logprobs[t]) for t in top5],
        "context_logprobs": {ex["context_objects"][i]: float(ctx_lp[i]) for i in range(len(ctx_ids))},
        "entropy": float(-(logprobs.exp() * logprobs).sum()),
    }
    if full_logprobs is not None:
        rec["kl_full_to_cond"] = float((full_logprobs.exp() * (full_logprobs - logprobs)).sum())
        rec["desc_logprob_delta"] = rec["desc_logprob"] - float(full_logprobs[desc_id])
        rec["put_logprob_delta"] = rec["put_logprob"] - float(full_logprobs[put_id])
    return rec


def run_condition(name, model, loader, examples, modules, components, mean_activations, ablate_non_vital_pos,
                  tokenizer, vocab_ids, full_lp_store):
    """Returns list of per-example records (in loader order) and fills/uses full_lp_store (set-name -> list of logprob tensors)."""
    t0 = time.time()
    recs = []
    i = 0
    with torch.no_grad():
        for inp in loader:
            input_ids = inp["base_tokens"].to(model.device)
            last_pos = inp["base_last_token_indices"]
            logits = forward_logits(model, input_ids, modules, components, mean_activations, last_pos, ablate_non_vital_pos)
            for bi in range(input_ids.shape[0]):
                ex = examples[i]
                lp = torch.log_softmax(logits[bi][int(last_pos[bi])].float(), dim=-1).cpu()
                assert int(inp["labels"][bi][0]) == ex["desc_id"] and int(inp["labels"][bi][1]) == ex["put_id"], (name, i)
                if full_lp_store.get(i) is None:
                    full_lp_store[i] = lp
                rec = describe_prediction(lp, ex, tokenizer, vocab_ids, None if components is None and name == "full_model" else full_lp_store[i])
                rec.update({"condition": name, "row_index": ex["row_index"], "candidate_rank": ex["candidate_rank"]})
                recs.append(rec)
                i += 1
            del logits
            torch.cuda.empty_cache()
    acc = np.mean([r["first_token_correct"] for r in recs])
    print(f"  [{name}] n={len(recs)} first-token acc={acc:.3f} desc_top2={np.mean([r['desc_in_top2'] for r in recs]):.3f} "
          f"put_top2={np.mean([r['put_in_top2'] for r in recs]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return recs


def summarize(recs):
    n = len(recs)
    def m(k): return float(np.mean([r[k] for r in recs]))
    def med(k): return float(np.median([r[k] for r in recs]))
    types = defaultdict(int)
    for r in recs:
        types[r["argmax_type"]] += 1
    s = {
        "n": n, "first_token_accuracy": m("first_token_correct"),
        "argmax_type_fraction": {k: v / n for k, v in sorted(types.items())},
        "desc_in_top2": m("desc_in_top2"), "put_in_top2": m("put_in_top2"),
        "desc_logprob_mean": m("desc_logprob"), "put_logprob_mean": m("put_logprob"),
        "desc_vocab_rank_median": med("desc_vocab_rank"), "put_vocab_rank_median": med("put_vocab_rank"),
        "desc_context_rank_mean": m("desc_context_rank"), "put_context_rank_mean": m("put_context_rank"),
        "desc_context_rank_hist": {str(k): sum(r["desc_context_rank"] == k for r in recs) / n for k in range(1, 9)},
        "put_context_rank_hist": {str(k): sum(r["put_context_rank"] == k for r in recs) / n for k in range(1, 9)},
        "desc_object_rank_median": med("desc_object_rank"), "put_object_rank_median": med("put_object_rank"),
        "desc_minus_put_logprob_mean": m("desc_minus_put_logprob"), "entropy_mean": m("entropy"),
    }
    if "kl_full_to_cond" in recs[0]:
        s.update({"kl_full_to_cond_mean": m("kl_full_to_cond"), "kl_full_to_cond_median": med("kl_full_to_cond"),
                  "desc_logprob_delta_mean": m("desc_logprob_delta"), "put_logprob_delta_mean": m("put_logprob_delta")})
    return s


# ----------------------------------------------------------------------------- main
def main():
    parser = add_args(build_parser())
    args = parser.parse_args()
    print(f"ARGS: {args}", flush=True)
    post_arg_parse_fix(args)
    fix_random_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    # 1. model + the same candidate pool the circuit scripts see (unfiltered, in loader order)
    _, dataset, model = get_model_and_dataset(args)
    tokenizer = model.tokenizer
    print(f"MODEL + DATA after {time.time() - t_start:.0f}s", flush=True)

    # 2. map loader rows back to data-file rows and to the behavioral map
    row_of_prompt = {}
    with open(args.datafile, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get("numops", 0) == 1:
                row_of_prompt[d["sentence_masked"] + " the"] = i
    bmap = {}
    with open(args.map_jsonl) as f:
        for line in f:
            d = json.loads(line)
            bmap[d["row_index"]] = d
    vocab_words = [r["object_name"] for r in csv.DictReader(open(args.object_vocabulary_file, encoding="utf-8-sig"))]
    vocab_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in vocab_words]
    assert all(len(tokenizer.encode(w, add_special_tokens=False)) == 1 for w in vocab_words)

    examples = []
    for i in range(len(dataset)):
        toks = np.asarray(dataset["base_tokens"][i]); last = int(dataset["base_last_token_indices"][i])
        prompt = tokenizer.decode(toks[: last + 1], skip_special_tokens=True)
        ridx = row_of_prompt[prompt]
        labels = [int(x) for x in dataset["labels"][i]]
        desc_w, put_w = tokenizer.decode([labels[0]]).strip(), tokenizer.decode([labels[1]]).strip()
        init = re.findall(r"[Tt]he (\w+) is in Box (\d)", prompt)
        put_m = re.search(r"Put the (\w+) into Box (\d)", prompt)
        q_m = re.search(r"Box (\d) contains the$", prompt)
        assert len(init) == 7 and put_m and q_m and put_m.group(1) == put_w and put_m.group(2) == q_m.group(1), prompt
        qbox = int(q_m.group(1))
        assert dict((int(b), o) for o, b in init)[qbox] == desc_w, prompt
        ctx_objs = [o for o, _ in init] + [put_w]
        ctx_ids = [tokenizer.encode(o, add_special_tokens=False)[0] for o in ctx_objs]
        assert ctx_ids[qbox] == labels[0] and ctx_ids[-1] == labels[1]
        desc_pos = int(np.where(toks[: last + 1] == labels[0])[0][0])
        put_pos = int(np.where(toks[: last + 1] == labels[1])[0][0])
        bm = bmap[ridx]
        examples.append({
            "loader_index": i, "row_index": ridx, "candidate_rank": bm["candidate_rank"], "prompt": prompt,
            "desc_obj": desc_w, "put_obj": put_w, "desc_id": labels[0], "put_id": labels[1],
            "query_box": qbox, "context_objects": ctx_objs, "context_ids": ctx_ids,
            "desc_distance": last - desc_pos, "put_distance": last - put_pos,
            "bmap_first_token_correct": bm["first_token_correct"], "bmap_argmax_is_desc": bm["argmax_is_desc"],
            "bmap_exact_match": bm["exact_match"], "bmap_generation": bm.get("generation"),
        })
    # candidate rank consistency: our own file-order rank must equal the behavioral map's
    own_rank = {e["row_index"]: r for r, e in enumerate(sorted(examples, key=lambda e: e["row_index"]))}
    assert all(own_rank[e["row_index"]] == e["candidate_rank"] for e in examples), "candidate ranks disagree with behavioral map"
    examples.sort(key=lambda e: e["candidate_rank"])

    fixed = [e for e in examples if e["candidate_rank"] >= args.fixed_start_rank and e["bmap_first_token_correct"]][: args.n_fixed]
    repro = [e for e in examples if e["candidate_rank"] < args.n_repro]
    assert len(fixed) == args.n_fixed, len(fixed)
    print(f"fixed set: {len(fixed)} examples, candidate ranks {fixed[0]['candidate_rank']}..{fixed[-1]['candidate_rank']}, "
          f"rows {fixed[0]['row_index']}..{fixed[-1]['row_index']}; repro set: {len(repro)} examples", flush=True)

    def make_loader(exs):
        sub = dataset.select([e["loader_index"] for e in exs])
        return DataLoader(sub, batch_size=args.batch_size, shuffle=False, collate_fn=partial(pad_batch_collate_fn, tokenizer=tokenizer))
    loaders = {"fixed": (make_loader(fixed), fixed), "repro": (make_loader(repro), repro)}

    # 3. the one mean-activation table
    mean_activations, modules = get_mean_activations(model=model, args=args, cache_dir=args.mean_activation_cache_path)
    print(f"mean activations loaded from {args.mean_activation_cache_path} ({len(modules)} modules)", flush=True)

    # 4. conditions
    circuits = {}  # name -> dict(root, heads, components, eff)
    for spec in args.circuit:
        name, root = spec.split("=", 1)
        heads = knee_heads(root)
        comps, eff = build_components(model, heads)
        ref_comps, gA, gB, gC, gD = get_circuit(model, root)  # the repo's own builder, for identity check
        assert components_equal(comps, ref_comps) and [len(gA), len(gB), len(gC), len(gD)] == [eff[g] for g in GROUPS], name
        circuits[name] = {"root": root, "heads": {g: [list(h) for h in heads[g]] for g in GROUPS},
                          "knee_sizes": {g: len(heads[g]) for g in GROUPS}, "effective_sizes": eff, "components": comps}
        print(f"circuit {name}: knee sizes {circuits[name]['knee_sizes']} effective {eff}", flush=True)

    conditions = []  # (name, components or None, ablate_non_vital_pos, meta)
    conditions.append(("full_model", None, None, {"kind": "full"}))
    conditions.append(("all_heads_skip_nonvital", all_heads_components(model), False, {"kind": "sanity", "expect": "identical to full_model"}))
    conditions.append(("all_heads_ablate_nonvital", all_heads_components(model), True, {"kind": "sanity", "expect": "cost of non-vital-position ablation alone"}))
    for name, c in circuits.items():
        conditions.append((name, c["components"], True, {"kind": "circuit", "knee_sizes": c["knee_sizes"], "effective_sizes": c["effective_sizes"]}))
    for spec in args.size_match:
        name, pair = spec.split("=", 1)
        src, ref = pair.split(":")
        heads = topk_heads(circuits[src]["root"], circuits[ref]["knee_sizes"])
        comps, eff = build_components(model, heads)
        circuits[name] = {"root": circuits[src]["root"], "heads": {g: [list(h) for h in heads[g]] for g in GROUPS},
                          "knee_sizes": {g: len(heads[g]) for g in GROUPS}, "effective_sizes": eff, "components": comps,
                          "note": f"scores of {src} cut at the knee sizes of {ref}"}
        conditions.append((name, comps, True, {"kind": "size_matched", "scores_from": src, "sizes_from": ref, "knee_sizes": circuits[name]["knee_sizes"], "effective_sizes": eff}))
        print(f"size-matched {name}: {circuits[name]['note']} -> sizes {circuits[name]['knee_sizes']} effective {eff}", flush=True)
    for spec in args.topk:
        name, rest = spec.split("=", 1)
        src, ks = rest.split(":")
        sizes = dict(zip(GROUPS, [int(k) for k in ks.split(",")]))
        heads = topk_heads(circuits[src]["root"], sizes)
        comps, eff = build_components(model, heads)
        circuits[name] = {"root": circuits[src]["root"], "heads": {g: [list(h) for h in heads[g]] for g in GROUPS},
                          "knee_sizes": {g: len(heads[g]) for g in GROUPS}, "effective_sizes": eff, "components": comps,
                          "note": f"top-k of {src} scores, requested {sizes}"}
        conditions.append((name, comps, True, {"kind": "topk", "scores_from": src, "requested_sizes": sizes, "knee_sizes": circuits[name]["knee_sizes"], "effective_sizes": eff}))
        print(f"top-k {name}: requested {sizes} -> selected {circuits[name]['knee_sizes']} effective {eff}", flush=True)
    for ref in args.keep_nonvital:
        c = circuits[ref]
        conditions.append((f"{ref}_keep_nonvital", c["components"], False, {"kind": "circuit_keep_nonvital", "circuit": ref, "knee_sizes": c["knee_sizes"], "effective_sizes": c["effective_sizes"],
                                                                        "note": "non-vital positions NOT ablated; only non-circuit heads at the 4 vital positions are mean-ablated"}))
        print(f"keep-nonvital variant of {ref}", flush=True)
    fix_random_seed(args.seed)
    for spec in args.random_like:
        ref, count = spec.split(":")
        eff = circuits[ref]["effective_sizes"]
        for k in range(int(count)):
            comps = get_random_circuit(model, eff["A"], eff["B"], eff["C"], eff["D"])
            conditions.append((f"random_like_{ref}_{k}", comps, True, {"kind": "random", "sizes_like": ref, "effective_sizes": eff}))

    print(f"\n{len(conditions)} conditions: {[c[0] for c in conditions]}", flush=True)

    # 5. run
    rows_path = Path(args.out_dir) / "per_example.jsonl"
    summary = {"args": {k: (str(v) if not isinstance(v, (int, float, bool, type(None), str, list)) else v) for k, v in vars(args).items()},
               "sets": {}, "circuits": {n: {k: v for k, v in c.items() if k != "components"} for n, c in circuits.items()},
               "conditions": [{"name": n, "ablate_non_vital_pos": a, **m} for n, _, a, m in conditions], "examples": {}}
    with open(rows_path, "w") as fout:
        for set_name, (loader, exs) in loaders.items():
            print(f"\n=== set {set_name} (n={len(exs)}) ===", flush=True)
            summary["examples"][set_name] = [{k: v for k, v in e.items() if k not in ("context_ids",)} for e in exs]
            full_store = {}
            summary["sets"][set_name] = {}
            per_cond = {}
            for name, comps, anv, meta in conditions:
                if set_name == "repro" and meta["kind"] not in ("full", "circuit"):
                    continue
                recs = run_condition(name, model, loader, exs, modules, comps, mean_activations, anv, tokenizer, vocab_ids, full_store)
                per_cond[name] = recs
                for r in recs:
                    fout.write(json.dumps({"set": set_name, **r}) + "\n")
                fout.flush()
                summary["sets"][set_name][name] = summarize(recs)
            # sanity: all_heads_skip_nonvital must reproduce full_model per example
            if "all_heads_skip_nonvital" in per_cond:
                f, s = per_cond["full_model"], per_cond["all_heads_skip_nonvital"]
                same_argmax = np.mean([a["argmax_token"] == b["argmax_token"] for a, b in zip(f, s)])
                max_dlp = max(abs(a["desc_logprob"] - b["desc_logprob"]) for a, b in zip(f, s))
                max_kl = max(b["kl_full_to_cond"] for b in s)
                summary["sets"][set_name]["sanity"] = {"argmax_agreement": float(same_argmax), "max_abs_desc_logprob_diff": float(max_dlp), "max_kl": float(max_kl)}
                print(f"SANITY {set_name}: argmax agreement {same_argmax:.3f}, max |Δ desc logprob| {max_dlp:.2e}, max KL {max_kl:.2e}", flush=True)
            # "faithfulness" as eval_circuits defines it: circuit first-token acc / full-model first-token acc
            full_acc = summary["sets"][set_name]["full_model"]["first_token_accuracy"]
            for name in per_cond:
                summary["sets"][set_name][name]["faithfulness_vs_full"] = summary["sets"][set_name][name]["first_token_accuracy"] / full_acc if full_acc else None
            json.dump(summary, open(Path(args.out_dir) / "summary.json", "w"), indent=1)

    summary["wallclock_s"] = round(time.time() - t_start)
    json.dump(summary, open(Path(args.out_dir) / "summary.json", "w"), indent=1)
    print("\n=== SUMMARY (first-token acc / faithfulness / desc top2 / put top2 / mean desc logprob / mean put logprob / KL) ===")
    for set_name in summary["sets"]:
        print(f"[{set_name}]")
        for name, s in summary["sets"][set_name].items():
            if name == "sanity":
                continue
            print(f"  {name:<34} acc={s['first_token_accuracy']:.3f} faith={s['faithfulness_vs_full']:.2f} descTop2={s['desc_in_top2']:.3f} putTop2={s['put_in_top2']:.3f} "
                  f"descLP={s['desc_logprob_mean']:.2f} putLP={s['put_logprob_mean']:.2f} KL={s.get('kl_full_to_cond_mean', 0):.3f} argmax={s['argmax_type_fraction']}")
    print(f"\nwrote {rows_path}\nwrote {Path(args.out_dir) / 'summary.json'}\ntotal {summary['wallclock_s']}s")


if __name__ == "__main__":
    main()

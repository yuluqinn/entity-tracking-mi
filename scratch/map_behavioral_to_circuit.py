"""Map behavioral (full-generation) metrics onto the examples the circuit pipeline uses.

For one single-PUT data file this script:
  1. loads the first N structural candidates with the SAME loader and SAME arguments the
     circuit scripts use (utils.load_dataloader: one PUT on the query box, two target
     objects, sorted so label 0 = DESCRIPTION object and label 1 = PUT object), with no
     success filter;
  2. re-derives, for each candidate, the first-token decision exactly as
     utils.check_prompt_success does for discovery / the mean-activation cache
     (one unpadded prompt per forward pass, argmax of the last-position logits), plus
     per-object top-2 membership as eval_circuits.eval_model_performance does;
  3. greedy-decodes a full answer from the same prompt with the settings of
     probe_experiments/inference.py (max 50 new tokens, stop at ".", newline or "Box"),
     parses object words with the llama-friendly vocabulary, and scores exact set match,
     precision and recall as that script does;
  4. writes one JSON line per candidate (keyed by its row index in the data file) and a
     summary over: all candidates, the first-token-correct subset, the unfiltered first
     100 (what eval_circuits evaluates), the first 200 first-token-correct rows (what
     run_path_patching traced), and the first 500 first-token-correct rows (mean-activation cache).

Run from the repository root (see scratch/run_map_behavioral_codellama13b.qsub).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from utils import load_dataloader  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--datafile", required=True)
    p.add_argument("--model", default="codellama/CodeLlama-13b-hf")
    p.add_argument("--num_samples", type=int, default=2000, help="structural candidates to take, in file order")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--batch_size", type=int, default=16, help="batch size for the generation pass only")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--object_vocabulary_file", default=str(REPO / "data" / "objects" / "llama_friendly_objects.csv"))
    p.add_argument("--n_eval", type=int, default=100, help="eval_circuits --num_samples (unfiltered)")
    p.add_argument("--n_discovery", type=int, default=200, help="run_path_patching --num_samples (filtered)")
    p.add_argument("--n_cache", type=int, default=500, help="mean-activation cache size (filtered)")
    p.add_argument("--seed", type=int, default=42, help="loader seed, as in the circuit scripts")
    return p.parse_args()


def load_vocab(path):
    with open(path, encoding="utf-8-sig") as f:
        return [row["object_name"] for row in csv.DictReader(f)]


def build_row_index(datafile):
    """Map the prompt string the loader produces (sentence_masked + ' the') to its row index."""
    idx = {}
    with open(datafile, encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if d.get("numops", 0) == 1:
                idx[d["sentence_masked"] + " the"] = i
    return idx


def parse_objects(text, object_set):
    return list(set(o for o in text.lower().replace(",", " ").replace(".", " ").split(" ") if o in object_set))


def summarize(records):
    n = len(records)
    if n == 0:
        return {"n": 0}
    def mean(key):
        return float(np.mean([r[key] for r in records]))
    prec_defined = [r["precision"] for r in records if r["precision"] is not None]
    return {
        "n": n,
        "full_generation_exact_match": mean("exact_match"),
        "full_generation_recall": mean("recall"),
        "full_generation_precision_over_nonempty_predictions": float(np.mean(prec_defined)) if prec_defined else None,
        "fraction_empty_predictions": 1.0 - len(prec_defined) / n,
        "first_token_argmax_accuracy": mean("first_token_correct"),
        "first_token_argmax_is_description_object": mean("argmax_is_desc"),
        "first_token_argmax_is_put_object": mean("argmax_is_put"),
        "description_object_top2_accuracy": mean("top2_has_desc"),
        "put_object_top2_accuracy": mean("top2_has_put"),
    }


def main():
    args = parse_args()
    t_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"ARGS: {vars(args)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # as in patch_utils.get_model_and_dataset while loading data

    # 1. same loader, same arguments as eval_circuits / run_path_patching, no success filter
    _, dataset = load_dataloader(
        tokenizer=tokenizer,
        datafile=args.datafile,
        num_samples=args.num_samples,
        num_boxes=7,
        ops_order=("put",),          # patch_utils converts the CLI string "put" to this tuple
        query_ops_order=("put",),
        success_filter=None,
        batch_size=1,
        return_dataset=True,
        counterfactual_format="rand_obj_rand_query_id",
        data_field="sentence",
        token_step="pred",
        prepend_space_to_answer=False,
        num_query_object=2,
        sort_query_objects=True,
        seed=args.seed,
        prompt_format=False,
        model=None,
    )
    print(f"DATALOADER CREATED ({len(dataset)=}) after {time.time() - t_start:.0f}s", flush=True)

    row_index = build_row_index(args.datafile)
    vocab = load_vocab(args.object_vocabulary_file)
    object_set = set(vocab)

    records = []
    unmatched = 0
    for i in range(len(dataset)):
        toks = np.asarray(dataset["base_tokens"][i])
        last = int(dataset["base_last_token_indices"][i])
        prompt = tokenizer.decode(toks[: last + 1], skip_special_tokens=True)
        labels = [int(x) for x in dataset["labels"][i]]
        assert len(labels) == 2, labels
        desc_word, put_word = (tokenizer.decode([t]).strip() for t in labels)
        ridx = row_index.get(prompt)
        if ridx is None:
            unmatched += 1
        records.append({
            "row_index": ridx, "prompt": prompt,
            "desc_obj": desc_word, "put_obj": put_word,
            "desc_label_id": labels[0], "put_label_id": labels[1],
        })
    print(f"prompts matched to file rows: {len(records) - unmatched}/{len(records)}", flush=True)
    assert unmatched == 0, "some loader prompts could not be matched back to data rows"
    records.sort(key=lambda r: r["row_index"])  # loader output is shuffled; restore file order
    for rank, r in enumerate(records):
        r["candidate_rank"] = rank

    # model, loaded as the circuit scripts load it (bitsandbytes 8-bit, device_map auto)
    qcfg = BitsAndBytesConfig(load_in_8bit=True) if args.load_in_8bit else None
    kwargs = {"device_map": "auto"}
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.eval()
    device = next(model.parameters()).device
    print(f"MODEL LOADED after {time.time() - t_start:.0f}s", flush=True)

    # 2. first-token pass: one unpadded prompt per forward pass, like check_prompt_success
    t0 = time.time()
    with torch.no_grad():
        for n, r in enumerate(records):
            ids = tokenizer(r["prompt"], return_tensors="pt").input_ids.to(device)
            logits = model(ids).logits[0, -1].float()
            order = torch.argsort(logits, descending=True)[:2].tolist()
            am = order[0]
            r["argmax_token_id"] = am
            r["argmax_token"] = tokenizer.decode([am])
            r["argmax_is_desc"] = int(am == r["desc_label_id"])
            r["argmax_is_put"] = int(am == r["put_label_id"])
            r["first_token_correct"] = int(am in (r["desc_label_id"], r["put_label_id"]))
            r["top2_has_desc"] = int(r["desc_label_id"] in order)
            r["top2_has_put"] = int(r["put_label_id"] in order)
            if (n + 1) % 200 == 0:
                print(f"  first-token pass {n + 1}/{len(records)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"FIRST-TOKEN PASS DONE in {time.time() - t0:.0f}s", flush=True)

    # 3. generation pass, settings of probe_experiments/inference.py
    tokenizer.padding_side = "left"
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(records), args.batch_size):
            batch = records[b: b + args.batch_size]
            enc = tokenizer([r["prompt"] for r in batch], return_tensors="pt", padding=True).to(device)
            out = model.generate(
                **enc, max_new_tokens=50, stop_strings=[".", "\n", "Box"], tokenizer=tokenizer,
                num_beams=1, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
            gens = tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            for r, g in zip(batch, gens):
                gold = [r["desc_obj"], r["put_obj"]]
                pred = parse_objects(g, object_set)
                r["generation"] = g
                r["predicted_objects"] = pred
                r["gold_objects"] = gold
                r["exact_match"] = int(set(pred) == set(gold))
                r["precision"] = float(np.mean([o in gold for o in pred])) if pred else None
                r["recall"] = float(np.mean([o in pred for o in gold]))
            if ((b // args.batch_size) + 1) % 20 == 0:
                print(f"  generation {min(b + args.batch_size, len(records))}/{len(records)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"GENERATION PASS DONE in {time.time() - t0:.0f}s", flush=True)

    # 4. subset flags that reproduce what each circuit script saw
    n_ok = 0
    for r in records:
        r["in_eval_unfiltered"] = int(r["candidate_rank"] < args.n_eval)
        if r["first_token_correct"]:
            n_ok += 1
            r["in_discovery_filtered"] = int(n_ok <= args.n_discovery)
            r["in_cache_filtered"] = int(n_ok <= args.n_cache)
        else:
            r["in_discovery_filtered"] = 0
            r["in_cache_filtered"] = 0

    base = Path(args.datafile).parent.name
    tag = f"{Path(args.model).name}{'_8bit' if args.load_in_8bit else ''}_{base}_n{args.num_samples}"
    rows_path = Path(args.out_dir) / f"map_{tag}.jsonl"
    with open(rows_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = {
        "datafile": args.datafile, "model": args.model, "load_in_8bit": args.load_in_8bit,
        "num_candidates": len(records),
        "last_file_row_scanned": records[-1]["row_index"],
        "n_first_token_correct": n_ok,
        "subsets": {
            "all_candidates_unfiltered": summarize(records),
            "first_token_correct": summarize([r for r in records if r["first_token_correct"]]),
            f"eval_first_{args.n_eval}_unfiltered": summarize([r for r in records if r["in_eval_unfiltered"]]),
            f"discovery_first_{args.n_discovery}_filtered": summarize([r for r in records if r["in_discovery_filtered"]]),
            f"cache_first_{args.n_cache}_filtered": summarize([r for r in records if r["in_cache_filtered"]]),
        },
        "rows_file": str(rows_path),
        "wallclock_s": round(time.time() - t_start),
    }
    sum_path = Path(args.out_dir) / f"summary_{tag}.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for name, s in summary["subsets"].items():
        print(f"\n[{name}] n={s['n']}")
        for k, v in s.items():
            if k != "n":
                print(f"  {k:<55} {v if v is None else round(v, 4)}")
    print(f"\nwrote {rows_path}\nwrote {sum_path}\ntotal {summary['wallclock_s']}s")


if __name__ == "__main__":
    main()

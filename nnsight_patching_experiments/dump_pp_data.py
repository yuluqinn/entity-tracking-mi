"""Dump the exact filtered dataset used by run_path_patching.py (same CLI flags) to a CSV."""
import csv, sys
sys.path.append(".."); sys.path.append(".")
from patch_utils import build_parser, get_model_and_dataset, post_arg_parse_fix
from utils import fix_random_seed, find_previous_query_box_pos

parser = build_parser(); parser.add_argument("--csv_out", required=True)
args = parser.parse_args(); post_arg_parse_fix(args); fix_random_seed(args.seed)
_, ds, model = get_model_and_dataset(args)
tok = model.tokenizer
rows = []
for i in range(len(ds)):
    ex = ds[i]
    base, src = ex["base_tokens"], ex["source_tokens"]
    last = int(ex["base_last_token_indices"])
    labels = [int(t) for t in ex["labels"]]
    prev = [int(p) for p in find_previous_query_box_pos({"base_tokens": base, "base_last_token_indices": last})]
    rows.append({
        "order": i,
        "jsonl_row_index": int(ex["dataset_indices"]) if "dataset_indices" in ex else "",
        "clean_prompt": tok.decode(base[: last + 1], skip_special_tokens=True),
        "corrupt_prompt": tok.decode(src[: last + 1], skip_special_tokens=True),
        "description_object": tok.decode([labels[0]]).strip(),
        "put_object": tok.decode([labels[-1]]).strip(),
        "label_token_ids": " ".join(map(str, labels)),
        "n_tokens": last + 1,
        "last_token_pos": last,
        "query_box_id_pos": last - 2,
        "query_box_id_token": tok.decode([int(base[last - 2])]),
        "prev_query_box_id_positions": " ".join(map(str, prev)),
    })
with open(args.csv_out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {args.csv_out} with {len(rows)} rows")

"""Ablation test: evaluate a circuit with group C, group D, or both removed (all heads at those positions mean-ablated)."""
import sys, argparse
from collections import defaultdict
from copy import deepcopy
sys.path.append(".."); sys.path.append(".")
from patch_utils import build_parser, get_model_and_dataset, post_arg_parse_fix
from utils import fix_random_seed, get_circuit, get_mean_activations, eval_circuit_performance
from eval_circuits import eval_model_performance

parser = build_parser()
parser.add_argument("--circuit_root_path", type=str, required=True)
parser.add_argument("--mean_activation_cache_path", type=str, required=True)
parser.add_argument("--n_more_heads_per_group", type=int, default=0)
args = parser.parse_args(); post_arg_parse_fix(args); fix_random_seed(args.seed)
dataloader, dataset, model = get_model_and_dataset(args)
circuit, A, B, C, D = get_circuit(model, args.circuit_root_path, n_more_heads_per_group=args.n_more_heads_per_group)
print(f"group sizes A={len(A)} B={len(B)} C={len(C)} D={len(D)}")
model_acc, m_arg, _ = eval_model_performance(model, dataloader)
import numpy as np
print(f"Model Performance {model_acc}. by label index {np.array(m_arg).mean(0)}")
mean_acts, modules = get_mean_activations(model=model, args=args, cache_dir=args.mean_activation_cache_path)
def variant(drop):
    c = deepcopy(circuit)
    if "C" in drop: c[2] = defaultdict(list)
    if "D" in drop: c[-1] = defaultdict(list)
    return c
for name, drop in [("full circuit", ""), ("no C (only A,B,D)", "C"), ("no D (only A,B,C)", "D"), ("no C, no D (only A,B)", "CD")]:
    acc, arg, _ = eval_circuit_performance(model, dataloader, modules, variant(drop), mean_acts)
    print(f"VARIANT {name}: acc {acc:.2f} faithfulness {acc/model_acc:.2f} by label index {np.round(np.array(arg).mean(0),2).tolist()}")
print("Execution finished!")

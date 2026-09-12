"""
Diagnostic: does the path-patching machinery reproduce the clean baseline when NO sender is patched?
Compares target log-prob at the last token across:
  a  : plain pass (as cache_logit_and_hidden, default use_cache)
  a2 : plain pass again (run-to-run nondeterminism)
  b  : plain pass with use_cache=False (setting used by all patched passes)
  c  : all-layer o_proj.input freeze to the cached clean values, no sender patch, use_cache=False
  d  : c + identity overwrite of q_proj/v_proj outputs at a few 'receiver' layers with values captured in the same pass
Reports mean/std of (variant - a) and of the path-patching score -(variant - a)/a per example.
"""
import sys, torch, einops, numpy as np
sys.path.append(".."); sys.path.append(".")
from patch_utils import build_parser, get_model_and_dataset, post_arg_parse_fix
from utils import fix_random_seed, force_pad
from run_path_patching import cache_logit_and_hidden, maybe_logit_soft_capping

args = build_parser().parse_args(); post_arg_parse_fix(args); fix_random_seed(args.seed)
_, dataset, model = get_model_and_dataset(args)
N_LAYERS = model.config.num_hidden_layers; N_HEADS = model.config.num_attention_heads
toks = dataset["base_tokens"]; last = list(dataset["base_last_token_indices"]); labels = list(dataset["labels"])
bs = args.batch_size; N = len(toks)
obj = slice(-1, None) if args.use_object_index is None else args.use_object_index

def target(logits, b, i):
    lp = torch.log_softmax(logits[b, last[i]].float(), dim=-1)[labels[i]]
    return lp[obj].sum().item() if isinstance(obj, slice) else lp[obj].sum().item()

clean_hs, _, clean_lp = cache_logit_and_hidden(model, bs, toks, last, labels, prefix_offset_pos=0, save_hs=True, cpu=True)
_, _, clean_lp2 = cache_logit_and_hidden(model, bs, toks, last, labels, prefix_offset_pos=0, save_hs=False)
def red(lp): return [(l[obj].sum() if isinstance(obj, slice) else l[obj].sum()).float().item() for l in lp]
res = {"a": red(clean_lp), "a2": red(clean_lp2), "b": [], "c": [], "d": []}

RECV = {5: "q_proj", 15: "v_proj", 25: "q_proj"}
with torch.no_grad():
    for b0 in range(0, N, bs):
        idx = list(range(b0, min(N, b0 + bs))); batch = force_pad(toks[idx], model.tokenizer)
        hs = torch.stack([clean_hs[i] for i in idx])  # (bs, layer, seq, nh, dh)
        # b
        with model.trace(batch, use_cache=False):
            lg = maybe_logit_soft_capping(model.lm_head.output, model).save()
        res["b"] += [target(lg, k, i) for k, i in enumerate(idx)]
        # c
        with model.trace(batch, use_cache=False):
            for l in range(N_LAYERS):
                z = model.model.layers[l].self_attn.o_proj.input
                z = einops.rearrange(z, "b s (nh dh) -> b s nh dh", nh=N_HEADS)
                z[:, 0:, ...] = hs[:, l].to(z.dtype).to(z.device)
                model.model.layers[l].self_attn.o_proj.input = einops.rearrange(z, "b s nh dh -> b s (nh dh)", nh=N_HEADS)
            lg = maybe_logit_soft_capping(model.lm_head.output, model).save()
        res["c"] += [target(lg, k, i) for k, i in enumerate(idx)]
        # d
        with model.trace(batch, use_cache=False):
            for l in range(N_LAYERS):
                if l in RECV:
                    mod = getattr(model.model.layers[l].self_attn, RECV[l])
                    zo = mod.output; mod.output = zo.clone()
                z = model.model.layers[l].self_attn.o_proj.input
                z = einops.rearrange(z, "b s (nh dh) -> b s nh dh", nh=N_HEADS)
                z[:, 0:, ...] = hs[:, l].to(z.dtype).to(z.device)
                model.model.layers[l].self_attn.o_proj.input = einops.rearrange(z, "b s nh dh -> b s (nh dh)", nh=N_HEADS)
            lg = maybe_logit_soft_capping(model.lm_head.output, model).save()
        res["d"] += [target(lg, k, i) for k, i in enumerate(idx)]

a = np.array(res["a"])
print(f"\nN={N} target logp (a) mean {a.mean():.4f}; per-example: {np.round(a, 4).tolist()}")
for k in ["a2", "b", "c", "d"]:
    v = np.array(res[k]); d = v - a; score = -(d / a)
    print(f"{k:>2}: mean diff {d.mean():+.5f}  std {d.std():.5f}  max|diff| {np.abs(d).max():.5f} | score mean {score.mean():+.5f} std {score.std():.5f}")

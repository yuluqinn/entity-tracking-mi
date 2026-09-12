from copy import deepcopy
import os

from typing import List, Tuple, Optional, Union
import argparse

import tqdm
from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
import einops
from transformers import AutoTokenizer
import nnsight
from nnsight import LanguageModel
from transformer_lens import HookedTransformer

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import plotly.io as pio
import gc

from patch_utils import build_parser, post_arg_parse_fix, maybe_patch_or_load_cache, get_model_and_dataset

pio.renderers.default = "plotly_mimetype+notebook_connected+colab+notebook"

import sys

sys.path.append("..")
from utils import get_model_and_tokenizer, load_dataloader, get_random_guess_baseline, fix_random_seed, str_to_bool, \
    find_previous_query_box_pos, is_int_with_negatives, force_pad, PROMPT_ALTFORM, compute_topk_components_knee, setup_nnsight


def plot_patching_results(
        patching_results,
        x_labels,
        plot_title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
        labels={"x": "Position", "y": "Layer", "color": "Norm. Logit Diff"},
        centered=True,
):
    # Proxy is no longer there after nnsight==0.5, not sure if this is ok
    # patching_results = util.apply(patching_results, lambda x: x.value, Proxy)
    if centered:
        fig = px.imshow(
            patching_results,
            color_continuous_midpoint=0.0,
            color_continuous_scale="RdBu",
            labels=labels,
            x=x_labels,
            title=plot_title,
        )
    else:
        fig = px.imshow(
            patching_results,
            labels=labels,
            x=x_labels,
            title=plot_title,
        )
    return fig


def visualize_top_heads_attention(
        clean_cache: torch.Tensor,
        prompt_tokens: torch.Tensor,
        last_token_index: torch.Tensor,
        output_dir: str,
        tokenizer: AutoTokenizer,
        heads: List[Tuple[int, int]],
        head_values: List[float],
        top_k: Optional[int] = None,
        rel_pos: Optional[int] = None,
        seq_pos: Optional[int] = None,
        group: str = "A",
):
    clean_cache = clean_cache.squeeze()
    # visualize attention pattern for these heads
    top_k = len(heads) if top_k is None else top_k
    total_attn_matrix = []
    for head in heads[:top_k]:
        layer_idx, head_idx = head[0], head[1]
        # seq_len X seq_len attention matrix
        attn_matrix = clean_cache[layer_idx, head_idx].cpu().numpy()
        total_attn_matrix.append(attn_matrix)

    # plot attention for generating our token of interest for all top_k heads
    seq_pos = last_token_index if seq_pos is None else seq_pos
    seq_pos = seq_pos if rel_pos is None else seq_pos + rel_pos
    all_tokens = [tokenizer.decode(t) for t in prompt_tokens]
    tokens = all_tokens[:seq_pos + 1]
    attn_matrix = np.array(total_attn_matrix)[:, seq_pos, :seq_pos + 1]
    plt.figure(figsize=(max(6.0, seq_pos / 5), max(4.0, top_k / 2)))
    yticks = [f"{heads[i]} ({head_values[i]:.2f})" for i in range(top_k)]
    sns.heatmap(attn_matrix, xticklabels=tokens, yticklabels=yticks)
    plt.title(
        f"Group {group} Top-{top_k} heads attention score (token={all_tokens[seq_pos]},idx={seq_pos}) over sequence")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/group{group}_pos{seq_pos}_heads_attn_scores.png", dpi=600)
    plt.close()


def get_attention_scores_tl(model_name: str, tokens: List[int]) -> torch.Tensor:
    # NNsight doesn't have native way of getting attention score, using transformerlens for now
    model = HookedTransformer.from_pretrained(model_name, device="cuda", center_unembed="gemma" not in model_name)
    with torch.no_grad():
        _, cache = model.run_with_cache(torch.LongTensor(tokens).to("cuda"))
    attention_scores = torch.concat([cache["attn", l] for l in range(len(model.blocks))]).cpu()
    del model
    torch.cuda.empty_cache()
    return attention_scores


def maybe_logit_soft_capping(logits: torch.Tensor, model: LanguageModel) -> torch.Tensor:
    if hasattr(model.config, "final_logit_softcapping") and model.config.final_logit_softcapping is not None:
        logits = logits / model.config.final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * model.config.final_logit_softcapping
    return logits


def get_patch_score(
    patched_logits, 
    patched_logprobs,
    clean_logits, 
    clean_logprobs,
    use_object_index: Optional[Union[List[int], slice]] = None,
    score_source: str = "logp",
    corrupted_logits: Optional[torch.Tensor] = None,
    corrupted_logprobs: Optional[torch.Tensor] = None,
) -> List[float]:
    """
    get the patching score from patched and clean logits:
    Args:
        patched_logits (List[torch.Tensor]): [batch, n_obj] logits of patched run
        patched_logprobs (List[torch.Tensor]): [batch, n_obj] logprobs of patched run
        clean_logits (List[torch.Tensor]): [batch, n_obj] logits of clean run
        clean_logprobs (List[torch.Tensor]): [batch, n_obj] probs of clean run
        use_object_index (Optional[Union[List[int], str]): index of object (or list slice object i.e. 'slice(-1)') to extract
            (when there are multiple target objects). None indicates using all objects
        score_source (str): whether to use logprob, prob or logit
        corrupted_logits (torch.Tensor): [batch, n_obj] logits of corrupted run
        corrupted_logprobs (torch.Tensor): [batch, n_obj] probs of corrupted run
    """
    # whether to use prob or logit (before softmax)
    batch_patched = patched_logprobs if score_source=="logp" else patched_logits if score_source=="logit" else [torch.exp(p) for p in patched_logprobs]
    batch_clean = clean_logprobs if score_source=="logp" else clean_logits if score_source=="logit" else [torch.exp(p) for p in clean_logprobs]
    batch_corrupted = None
    if corrupted_logprobs is not None and corrupted_logits is not None:
        batch_corrupted = corrupted_logprobs if score_source=="logp" else corrupted_logits if score_source=="logit" else [torch.exp(p) for p in corrupted_logprobs]

    batch_score = []
    for i in range(len(batch_patched)):
        # for each data instance, there could be different number of target objects
        object_index = range(len(batch_patched[i])) if use_object_index is None else use_object_index
        if isinstance(object_index, slice) or (len(object_index) > 1):
            patched_score = batch_patched[i][object_index].sum(-1)
            clean_score = batch_clean[i][object_index].sum(-1)
            if batch_corrupted is not None:
                corrupted_score = batch_corrupted[i][object_index].sum(-1)
        else:
            patched_score = batch_patched[i][object_index]
            clean_score = batch_clean[i][object_index]
            if batch_corrupted is not None:
                corrupted_score = batch_corrupted[i][object_index]
        
        if batch_corrupted is None:
            final_score = (patched_score - clean_score) / clean_score
        else:
            final_score = (patched_score - corrupted_score) / (clean_score - corrupted_score)

        if score_source=="logp": # since logp are negatives, dividing by negative value inverts the scores, we invert it here to make it comparable to probs and logits
            final_score = - final_score

        batch_score.append(final_score.detach())
    return batch_score


def get_path_patch_head_to_heads(
    receiver_heads: Optional[List[List[int]]],
    receiver_input: str,
    model: LanguageModel,
    clean_tokens: np.ndarray,
    corrupted_tokens: np.ndarray,
    label_tokens: List[List[int]],
    last_token_pos: Union[List[int], np.ndarray],
    args: argparse.Namespace,
    sender_receiver_pos: Optional[Union[Float[np.ndarray, "data_size"], Float[Tensor, "data_size n_receivers"]]] = None,
) -> List[List[str]]:

    def _as_pos_list(x):
        # returns List[int] for scalar/int, np arrays, torch tensors, lists, etc.
        if isinstance(x, (list, tuple, np.ndarray, torch.Tensor)):
            arr = np.asarray(x).reshape(-1)
            return [int(v) for v in arr.tolist()]
        return [int(x)]

    assert receiver_input in {"q", "k", "v"}
    N_LAYERS = model.config.num_hidden_layers
    N_HEADS = model.config.num_attention_heads
    N_KV_HEADS = model.config.num_key_value_heads if hasattr(model.config, "num_key_value_heads") else N_HEADS
    D_MODEL = model.config.hidden_size
    D_HEADS = int(D_MODEL / N_HEADS)
    ATTN_TO_KV_HEADS_RATIO = N_HEADS / N_KV_HEADS
    N_DATA = len(clean_tokens)
    node_name = f"{receiver_input}_proj"

    prefix_offset_pos = 0  # zero-shot

    receiver_layers = [h[0] for h in receiver_heads] if receiver_heads else [N_LAYERS]
    receiver_layers_set = set([h[0] for h in receiver_heads]) if receiver_heads else set()

    # ======== Step 1 ==========
    clean_hs, clean_logits, clean_logprobs = cache_logit_and_hidden(
        model=model, batch_size=args.batch_size,
        tokens_ids=clean_tokens, last_token_pos=last_token_pos,
        label_indices=label_tokens, save_hs=True, cpu=True, remote=args.remote,
        prefix_offset_pos=prefix_offset_pos
    )

    corrupt_hs, corrupted_logits, corrupted_logprobs = cache_logit_and_hidden(
        model=model, batch_size=args.batch_size,
        tokens_ids=corrupted_tokens, last_token_pos=last_token_pos,
        label_indices=label_tokens, save_hs=True, cpu=True, remote=args.remote,
        prefix_offset_pos=prefix_offset_pos
    )

    # ======== Step 2 ==========
    batch_size = deepcopy(args.batch_size)
    remote = deepcopy(args.remote)
    use_object_index = deepcopy(args.use_object_index)
    score_source = deepcopy(args.score_source)

    patching_results = []
    bar = tqdm.tqdm(total=max(receiver_layers) * N_HEADS)

    # Precompute receiver head indices per layer (tiny)
    receiver_head_indices_by_layer = {}
    if receiver_heads:
        for l in receiver_layers_set:
            inds = [h[1] for h in receiver_heads if h[0] == l]
            if receiver_input in {"k", "v"}:
                inds = [int(i // ATTN_TO_KV_HEADS_RATIO) for i in inds]
            receiver_head_indices_by_layer[l] = inds

    for sender_layer in range(max(receiver_layers)):
        _patching_results = []

        for sender_head in range(N_HEADS):
            patched_result_sum = torch.zeros(1)

            for batch_i in range(0, N_DATA, batch_size):
                batch_indices = range(batch_i, min(N_DATA, batch_i + batch_size))
                batch_indices_list = list(batch_indices)
                bs = len(batch_indices_list)

                batch_clean_tokens = force_pad(clean_tokens[batch_indices_list], model.tokenizer)

                # print(f"batch_indices_list = {batch_indices_list}")
                # print(f"sender_receiver_pos = {sender_receiver_pos}")
                # print()

                # # positions (prefix_offset_pos is 0 here, but keep pattern)
                sender_pos_orig = [sender_receiver_pos[b] for b in batch_indices_list]
                sender_pos_suf = [p - prefix_offset_pos for p in sender_pos_orig]
                sender_pos_suf = torch.tensor(sender_pos_suf)

                def _as_1d_np(x):
                    return np.asarray(x, dtype=np.int64).reshape(-1)

                pos_lists = [_as_1d_np(sender_receiver_pos[b]) for b in batch_indices_list]
                R = len(pos_lists[0])
                assert all(len(p) == R for p in pos_lists), "sender_receiver_pos has inconsistent lengths"

                pos_orig_cpu = torch.from_numpy(np.stack(pos_lists, axis=0)).long()  # (bs, R) CPU
                pos_suf_cpu  = (pos_orig_cpu - int(prefix_offset_pos)).long()        # (bs, R) CPU

                assert torch.allclose(pos_suf_cpu.reshape(-1), sender_pos_suf.reshape(-1))

                clean_hs_batch = torch.stack([clean_hs[b] for b in batch_indices]) # (bs, layer, seq_trim, nh, dh)
                corrupt_patch = torch.stack([
                    corrupt_hs[b][sender_layer, :, sender_head, :]
                    for b in batch_indices
                ]) # (bs, seq_trim, dh)

                torch.cuda.empty_cache()


                # ======== Step 2a: run once, cache receiver activations ========
                patched_hs_saved = None
                with torch.no_grad():
                    with model.trace(batch_clean_tokens, remote=remote, use_cache=False) as tracer:
                        patched_hs_layers = []

                        for l in range(N_LAYERS):
                            # IMPORTANT: request receiver output FIRST to avoid OutOfOrder when receiver is q/k/v
                            z_patched = getattr(model.model.layers[l].self_attn, node_name).output
                            z_patched = einops.rearrange(
                                z_patched,
                                "b s (nh dh) -> b s nh dh",
                                nh=(N_KV_HEADS if receiver_input in {"k", "v"} else N_HEADS),
                            )
                            patched_hs_layers.append(z_patched.detach())

                            # now patch/freeze o_proj.input for layers before last receiver layer
                            if l < max(receiver_layers):
                                z_orig = model.model.layers[l].self_attn.o_proj.input  # [b, s, d_model]
                                z_orig = einops.rearrange(z_orig, "b s (nh dh) -> b s nh dh", nh=N_HEADS)
                                dest_dtype = z_orig.dtype
                                dest_device = z_orig.device

                                # freeze to clean cache
                                z_orig[:, prefix_offset_pos:, ...] = clean_hs_batch[:, l].to(dest_dtype).to(dest_device)

                                # sender patch
                                if l == sender_layer:
                                    # b_idx = torch.arange(bs, device=dest_device)
                                    # pos_orig_old = torch.tensor(sender_pos_orig, dtype=torch.long, device=dest_device)
                                    # pos_suf_old = torch.tensor(sender_pos_suf, dtype=torch.long, device=dest_device)

                                    # src = corrupt_patch.to(dest_dtype).to(dest_device)
                                    # z_orig[b_idx, pos_orig, sender_head, :] = src[b_idx, pos_suf, :]
                                    pos_orig = pos_orig_cpu.to(dest_device)  # (bs, R)
                                    pos_suf  = pos_suf_cpu.to(dest_device)   # (bs, R)

                                    b_idx = torch.arange(bs, device=dest_device).view(bs, 1).expand(bs, R)  # (bs, R)

                                    src = corrupt_patch.to(dest_dtype).to(dest_device)  # (bs, seq_trim, dh)

                                    # print(f"pos_orig_old={pos_orig_old}")
                                    # print(f"pos_orig={pos_orig}")
                                    # print("\n")
                                    # print(f"pos_suf_old={pos_suf_old}")
                                    # print(f"pos_suf={pos_suf}")

                                    # assert torch.allclose(pos_orig.reshape(-1), pos_orig_old.reshape(-1))
                                    # assert torch.allclose(pos_suf.reshape(-1), pos_suf_old.reshape(-1))

                                    # PER-EXAMPLE (old semantics), now vectorized over R
                                    z_orig[b_idx, pos_orig, sender_head, :] = src[b_idx, pos_suf, :]

                                z_orig = einops.rearrange(z_orig, "b s nh dh -> b s (nh dh)", nh=N_HEADS)
                                # (fix: write back to layer l, not sender_layer)
                                model.model.layers[l].self_attn.o_proj.input = z_orig

                        # save once (layer, bs, seq, nh, dh)
                        patched_hs_saved = torch.stack(patched_hs_layers).save()

                patched_hs = patched_hs_saved.value if hasattr(patched_hs_saved, "value") else patched_hs_saved

                # ======== Step 3 ==========
                with torch.no_grad():
                    with model.trace(batch_clean_tokens, remote=remote, use_cache=False) as tracer:
                        for l in range(N_LAYERS):
                            if receiver_heads and (l in receiver_layers_set):
                                z_orig = getattr(model.model.layers[l].self_attn, node_name).output
                                z_orig = einops.rearrange(
                                    z_orig,
                                    "b s (nh dh) -> b s nh dh",
                                    nh=(N_KV_HEADS if receiver_input in {"k", "v"} else N_HEADS),
                                )
                                dest_dtype = z_orig.dtype
                                dest_device = z_orig.device

                                # b_idx = torch.arange(bs, device=dest_device)
                                # pos_orig = torch.tensor(sender_pos_orig, dtype=torch.long, device=dest_device)

                                # for receiver_head_index in receiver_head_indices_by_layer[l]:
                                #     src = patched_hs[l]  # (bs, seq, nh, dh) on CPU
                                #     # gather per-example positions, then move to device/dtype
                                #     src_sel = src[b_idx.cpu(), pos_orig.cpu(), receiver_head_index, :].to(dest_dtype).to(dest_device)
                                #     z_orig[b_idx, pos_orig, receiver_head_index, :] = src_sel

                                pos_orig_dev = pos_orig_cpu.to(dest_device)  # (bs, R)
                                pos_orig_idx = pos_orig_cpu                  # (bs, R) CPU for indexing patched_hs (CPU)

                                b_idx_dev = torch.arange(bs, device=dest_device).view(bs, 1).expand(bs, R)  # (bs, R)
                                b_idx_cpu = torch.arange(bs).view(bs, 1).expand(bs, R)                      # (bs, R)

                                for receiver_head_index in receiver_head_indices_by_layer[l]:
                                    rhs = patched_hs[l][b_idx_cpu, pos_orig_idx, receiver_head_index, :].to(dest_dtype).to(dest_device)  # (bs, R, dh)
                                    z_orig[b_idx_dev, pos_orig_dev, receiver_head_index, :] = rhs

                                #     if R == 1:
                                #         pos1d = pos_orig_cpu.squeeze(1)
                                #         b1d = torch.arange(bs)
                                #         rhs_1d = patched_hs[l][b1d, pos1d, receiver_head_index, :]          # (bs, dh)
                                #         rhs_2d = patched_hs[l][b_idx_cpu, pos_orig_idx, receiver_head_index, :].squeeze(1)  # (bs, dh)

                                #         print(f"rhs_1d={rhs_1d}")
                                #         print(f"rhs_2d={rhs_2d}")
                                #         assert torch.equal(rhs_1d, rhs_2d)

                                # print("All the tests passed...")
                                # exit()

                                z_orig = einops.rearrange(
                                    z_orig,
                                    "b s nh dh -> b s (nh dh)",
                                    nh=(N_KV_HEADS if receiver_input in {"k", "v"} else N_HEADS),
                                )
                                getattr(model.model.layers[l].self_attn, node_name).output = z_orig

                        patched_logits = model.lm_head.output
                        patched_logits = maybe_logit_soft_capping(patched_logits, model)
                        patched_logprobs = torch.log_softmax(patched_logits, dim=-1)

                        patched_logits_batch = [
                            patched_logits[bi, last_token_pos[batch_indices_list[bi]], label_tokens[batch_indices_list[bi]]]
                            for bi in range(bs)
                        ]
                        patched_logprobs_batch = [
                            patched_logprobs[bi, last_token_pos[batch_indices_list[bi]], label_tokens[batch_indices_list[bi]]]
                            for bi in range(bs)
                        ]

                        dev_patched = patched_logits.device
                        baseline_logits_batch = [clean_logits[i].to(dev_patched) for i in batch_indices_list]
                        baseline_logprobs_batch = [clean_logprobs[i].to(dev_patched) for i in batch_indices_list]

                        batch_patched_result = get_patch_score(
                            patched_logits_batch,
                            patched_logprobs_batch,
                            baseline_logits_batch,
                            baseline_logprobs_batch,
                            use_object_index,
                            score_source
                        )

                        patched_result_sum = patched_result_sum.to(dev_patched)
                        for bi in range(bs):
                            patched_result_sum = (patched_result_sum + batch_patched_result[bi]).save()

            patch_result_avg = patched_result_sum / N_DATA
            _patching_results.append(patch_result_avg.detach().cpu().item())
            bar.update(1)

        patching_results.append(_patching_results)

    for i in range(min(len(clean_tokens), 2)):
        print(f"example {i}")
        print(f"Clean logit: {clean_logits[i].tolist()}, clean logprob: {clean_logprobs[i].tolist()}, clean prob: {torch.exp(clean_logprobs[i]).tolist()}")
        print(f"Corrupted logit: {corrupted_logits[i].tolist()}, corrupted logprob: {corrupted_logprobs[i].tolist()}, corrupted prob: {torch.exp(corrupted_logprobs[i]).tolist()}")

    return patching_results

def path_patching_heads_to_final_residual_stream(
    model: LanguageModel,
    clean_tokens: np.ndarray,
    corrupted_tokens: np.ndarray,
    label_tokens: List[List[int]],
    last_token_pos: List[int],
    args: argparse.Namespace,
    sender_receiver_pos: Optional[Union[Float[Tensor, "data_size"], Float[Tensor, "data_size n_receivers"]]] = None,
) -> List[List[float]]:
    # default to receive at the last token position
    sender_receiver_pos = last_token_pos if sender_receiver_pos is None else sender_receiver_pos

    N_LAYERS = model.config.num_hidden_layers
    N_HEADS = model.config.num_attention_heads
    D_MODEL = model.config.hidden_size
    D_HEADS = int(D_MODEL / N_HEADS)
    N_DATA = len(clean_tokens)
    N_BATCHES = len(range(0, N_DATA, args.batch_size))

    # Computing the prefix offset to not save the 2-shot prompt
    #TODO: make this as a conditional given the args
    #prefix_offset_pos = len(model.tokenizer.encode(PROMPT_ALTFORM.strip()))
    prefix_offset_pos = 0 # For zero shot, no offset

    # ======== Step 1 ==========
    clean_hs, clean_logits, clean_logprobs = cache_logit_and_hidden(
        model=model,
        batch_size=args.batch_size,
        tokens_ids=clean_tokens,
        last_token_pos=last_token_pos,
        label_indices=label_tokens,
        save_hs=True,
        cpu=False,
        remote=args.remote,
        prefix_offset_pos=prefix_offset_pos,
    )

    corrupt_hs, corrupted_logits, corrupted_logprobs = cache_logit_and_hidden(
        model=model,
        batch_size=args.batch_size,
        tokens_ids=corrupted_tokens,
        last_token_pos=last_token_pos,
        label_indices=label_tokens,
        save_hs=True,
        cpu=False,
        remote=args.remote,
        prefix_offset_pos=prefix_offset_pos,
    )

    # ======== Step 2 ==========
    patching_results = []

    batch_size = deepcopy(args.batch_size)
    remote = deepcopy(args.remote)
    use_object_index = deepcopy(args.use_object_index)
    score_source = deepcopy(args.score_source)

    bar = tqdm.tqdm(total=N_LAYERS * N_HEADS)

    for sender_layer in range(N_LAYERS):
        _patching_results = []

        for sender_head in range(N_HEADS):
            patched_result_sum = torch.zeros(1)

            for batch_i in range(0, N_DATA, batch_size):
                batch_indices = range(batch_i, min(N_DATA, batch_i + batch_size))
                batch_indices_list = list(batch_indices)
                batch_clean_tokens = force_pad(clean_tokens[batch_indices], model.tokenizer)
                bs = len(batch_clean_tokens)  # FIX: true batch size (last batch may be smaller)
                b_idx = torch.arange(bs)

                clean_hs_batch = torch.stack([clean_hs[b] for b in batch_indices])
                corrupt_patch = torch.stack([
                    corrupt_hs[b][sender_layer, :, sender_head, :]
                    for b in batch_indices
                ]).to("cuda")

                torch.cuda.empty_cache()
                with torch.no_grad():
                    # Disable cache to avoid remote HF forward kwargs issues
                    with model.trace(batch_clean_tokens, remote=remote, use_cache=False) as tracer:
                        for l in range(N_LAYERS):
                            z_orig = model.model.layers[l].self_attn.o_proj.input  # [b, s, d_model]
                            z_orig = einops.rearrange(
                                z_orig, "b s (nh dh) -> b s nh dh", nh=N_HEADS
                            )
                            dest_dtype = z_orig.dtype

                            z_orig[:, prefix_offset_pos:, ...] = clean_hs_batch[:, l].to(dest_dtype)

                            if l == sender_layer:
                                sender_receiver_pos_batch = [sender_receiver_pos[b] for b in batch_indices]

                                sender_receiver_pos_batch = [int(x) - prefix_offset_pos for x in sender_receiver_pos_batch]

                                # corrupt_patch: [bs, seq_trim, d_head]
                                z_orig[b_idx, sender_receiver_pos_batch, sender_head, :] = \
                                    corrupt_patch[b_idx, sender_receiver_pos_batch, :].to(dest_dtype)

                            z_orig = einops.rearrange(
                                z_orig, "b s nh dh -> b s (nh dh)", nh=N_HEADS
                            )
                            model.model.layers[l].self_attn.o_proj.input = z_orig

                        patched_logits = model.lm_head.output
                        patched_logits = maybe_logit_soft_capping(patched_logits, model)
                        patched_logprobs = torch.log_softmax(patched_logits, dim=-1)

                        patched_logits_batch = [
                            patched_logits[bi, last_token_pos[batch_indices_list[bi]], label_tokens[batch_indices_list[bi]]]
                            for bi in range(bs)
                        ]
                        patched_logprobs_batch = [
                            patched_logprobs[bi, last_token_pos[batch_indices_list[bi]], label_tokens[batch_indices_list[bi]]]
                            for bi in range(bs)
                        ]

                        dev_patched = patched_logits.device
                        baseline_logits_batch = [clean_logits[i].to(dev_patched) for i in batch_indices_list]
                        baseline_logprobs_batch = [clean_logprobs[i].to(dev_patched) for i in batch_indices_list]

                        batch_patched_result = get_patch_score(
                            patched_logits_batch,
                            patched_logprobs_batch,
                            baseline_logits_batch,
                            baseline_logprobs_batch,
                            use_object_index,
                            score_source,
                        )

                        patched_result_sum = patched_result_sum.to(dev_patched)

                        for bi in range(bs):
                            patched_result_sum = (patched_result_sum + batch_patched_result[bi]).save()

            patch_result_avg = patched_result_sum / N_DATA
            _patching_results.append(patch_result_avg.detach().cpu().item())
            bar.update(1)

        patching_results.append(_patching_results)

    for i in range(min(len(clean_tokens), 2)):
        print(f"example {i}")
        print(
            f"Clean logit: {clean_logits[i].tolist()}, clean logprob: {clean_logprobs[i].tolist()}, clean prob: {torch.exp(clean_logprobs[i]).tolist()}"
        )
        print(
            f"Corrupted logit: {corrupted_logits[i].tolist()}, corrupted logprob: {corrupted_logprobs[i].tolist()}, corrupted prob: {torch.exp(corrupted_logprobs[i]).tolist()}"
        )

    return patching_results

def cache_logit_and_hidden(
    model: LanguageModel,
    batch_size: int,
    tokens_ids: np.array,
    last_token_pos: List[int],
    label_indices: List[np.array],
    prefix_offset_pos: int,
    save_hs=False,
    cpu=False,
    reshape=True,
    module="o_proj",
    remote=False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Cache the logits and hidden states of the model
    Args:
        model (LanguageModel): Language model
        batch_size (int): Batch size
        tokens_ids (np.array): input Token ids to cache the logit from
        last_token_pos (List[int]): last token position, usually where we are in the logits at 
        label_indices (List[np.array]): indices of the labels for each data instance (could be multiple labels)
        save_hs (bool): whether to save hidden states
        cpu (bool): cache hidden states from cuda to cpu to save memory or not
        reshape (bool): whether to reshape hidden states (so head is a separate dimension)
        module (str): which module to save hidden states for
        remote (bool): whether to run model in NDIF remote mode

    returns
        hidden state in shape of (n_data, Torch[layer, seq, head, d_head])
    """
    N_LAYERS = model.config.num_hidden_layers
    N_HEADS = model.config.num_attention_heads

    clean_hs_all = []
    logits_all = []
    logprobs_all = []

    with torch.no_grad():
        for batch_i in tqdm.tqdm(range(0, len(tokens_ids), batch_size), f"caching with {save_hs=}"):
            batch_indices = range(batch_i, min(len(tokens_ids), batch_i + batch_size))
            tokens_batch = force_pad(tokens_ids[batch_indices], model.tokenizer)

            clean_hs_saved = None

            # use_cache=False to match the patched passes exactly: with default use_cache the HF attention path
            # differs numerically (up to ~0.15 nats/example in fp16), which otherwise leaks into every patch score
            with model.trace(tokens_batch, remote=remote, use_cache=False) as tracer:
                if save_hs:
                    clean_hs = []
                    for sender_layer in range(N_LAYERS):
                        if module == "o_proj":
                            z = model.model.layers[sender_layer].self_attn.o_proj.input
                        elif module == "resid":
                            z = model.model.layers[sender_layer].output[0]
                        else:
                            raise NotImplementedError(f"module {module} not implemented")

                        if reshape and module == "o_proj":
                            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh', nh=N_HEADS)
                            clean_hs.append(z_reshaped.cpu().detach())
                        else:
                            clean_hs.append(z)

                # Get logits from the lm_head.
                logits = model.lm_head.output
                logits = maybe_logit_soft_capping(logits, model)
                logits_saved = logits.detach().save()                
                
                if save_hs:
                    clean_hs_saved = (
                        torch.stack(clean_hs)
                        .transpose(0, 1)[:, :, prefix_offset_pos:, ...]
                        .save()
                    )

            # Saving logits
            for batch_i, data_i in enumerate(batch_indices):
                logit = logits_saved[batch_i, last_token_pos[data_i], label_indices[data_i]]
                logprob = torch.log_softmax(
                    logits_saved[batch_i, last_token_pos[data_i]], dim=-1
                )[label_indices[data_i]]
                logits_all.append(logit)
                logprobs_all.append(logprob)

            # NEW: append after exiting trace
            if save_hs:
                clean_hs = clean_hs_saved.value if hasattr(clean_hs_saved, "value") else clean_hs_saved
                for b in range(len(batch_indices)):
                    clean_hs_all.append(clean_hs[b])

            # Cleaning up memory
            del tokens_batch
            if save_hs:
                del clean_hs
                del clean_hs_saved
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if save_hs and cpu:
        clean_hs_all = [h.cpu() for h in clean_hs_all]

    return clean_hs_all, logits_all, logprobs_all


def path_patch_abcd_circuit(args):
    """
    Path patch group ABCD same way as Nikhil's paper 2024 (finetuning ...)
    """
    if args.remote:
        setup_nnsight()

    _, dataset, model = get_model_and_dataset(args)
    clean_prompt = model.tokenizer.decode(dataset['base_tokens'][0], skip_special_tokens=True)
    corrupted_prompt = model.tokenizer.decode(dataset['source_tokens'][0], skip_special_tokens=True)

    correct_index = dataset['labels'][0]
    final_token_position = dataset["base_last_token_indices"][0]
    print(f"clean_prompt: \n{clean_prompt}")
    print(f"corrupted_prompt: \n{corrupted_prompt}")
    print(f"correct_index for label {model.tokenizer.decode(correct_index)} = {correct_index}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/n{len(dataset)}", exist_ok=True)

    # cache attention matrices across (using transformer lens)
    # if not args.remote:
    #     attn_scores = get_attention_scores_tl(args.model, dataset['base_tokens'][0])
    # else:
    # Setting this to always none because codellama is not directly available on TL
    # TODO: make it conditional again
    attn_scores = None

    # patch for group A heads
    print("Path patching for group A ...")
    patching_results = maybe_patch_or_load_cache(
        f"{args.output_dir}/n{len(dataset)}/pp_groupA.npy",
        path_patching_heads_to_final_residual_stream,
        model=model,
        clean_tokens=dataset['base_tokens'],
        corrupted_tokens=dataset['source_tokens'],
        label_tokens=list(dataset['labels']),
        last_token_pos=list(dataset["base_last_token_indices"]),
        args=args
    )
    head_labels = list(range(len(patching_results[0])))
    fig = plot_patching_results(patching_results, head_labels,
                                    f"Path Patching {args.model} GroupA {args.ops_order=}, {args.query_ops_order=}",
                                    labels={"x": "Head", "y": "Layer","color": "(patch - clean) / clean"})
    plotly.offline.plot(fig, filename=f"{args.output_dir}/n{len(dataset)}/pp_groupA.html", auto_open=False)
    # group_a_heads, head_values = compute_topk_components(
    #     patching_scores=torch.tensor(patching_results), k=args.n_groupA, largest=False, return_values=True, top_p=args.top_p
    # )
    group_a_heads, head_values = compute_topk_components_knee(torch.tensor(patching_results), 
                                    largest=False, return_values=True, 
                                    knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"}, 
                                    return_knee=False)
    print("\ngroup_a_heads")
    print(group_a_heads)
    print()
    # group_a_heads, head_values = ([[22, 5], [18, 6], [22, 4], [20, 3], [19, 1], [23, 6], [16, 4]], [-0.3669336438179016, -0.20949073135852814, -0.2065345048904419, -0.1578013151884079, -0.14925500750541687, -0.10442657023668289, -0.08579779416322708])
    if attn_scores is not None:
        visualize_top_heads_attention(attn_scores, dataset['base_tokens'][0], dataset['base_last_token_indices'][0],
                                      output_dir=f"{args.output_dir}/n{len(dataset)}", tokenizer=model.tokenizer, heads=group_a_heads,
                                      head_values=head_values)

    # group B
    print("Path patching for group B ...")
    patching_results = maybe_patch_or_load_cache(
        f"{args.output_dir}/n{len(dataset)}/pp_groupB.npy",
        get_path_patch_head_to_heads,
        receiver_heads=group_a_heads, receiver_input="q", model=model, clean_tokens=dataset['base_tokens'],
        corrupted_tokens=dataset['source_tokens'],
        label_tokens=list(dataset['labels']),
        last_token_pos=list(dataset["base_last_token_indices"]),
        sender_receiver_pos=list(dataset["base_last_token_indices"]),
        args=args
    )
    head_labels = list(range(len(patching_results[0])))
    fig = plot_patching_results(patching_results, head_labels,
                                f"Path Patching model={args.model} groupB ops_order={args.ops_order} query_ops_order={args.query_ops_order}\n",
                                labels={"x": "Head", "y": "Layer", "color": "(patch - clean) / clean" if args.score_source=="prob" else "(l_patch - l_org) / l_org"})
    plotly.offline.plot(fig, filename=f"{args.output_dir}/n{len(dataset)}/pp_groupB.html", auto_open=False)
    # group_b_heads, head_values = compute_topk_components(
    #     patching_scores=torch.tensor(patching_results), k=args.n_groupB, largest=False, return_values=True, top_p=args.top_p
    # )
    group_b_heads, head_values = compute_topk_components_knee(torch.tensor(patching_results), 
                                    largest=False, return_values=True, 
                                    knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"}, 
                                    return_knee=False)
    print("\ngroup_b_heads")
    print(group_b_heads)
    print()
    # group_b_heads, head_values = ([[15, 7], [16, 4], [17, 7], [14, 4], [13, 6], [17, 3], [12, 3], [11, 5], [16, 6], [16, 7]], [-0.5634344816207886, -0.2842274606227875, -0.2712571918964386, -0.20199373364448547, -0.1997925490140915, -0.18619365990161896, -0.10157862305641174, -0.10109157860279083, -0.10030236095190048, -0.0919051319360733])
    if attn_scores is not None:
        visualize_top_heads_attention(attn_scores, dataset['base_tokens'][0], dataset['base_last_token_indices'][0],
                                      output_dir=f"{args.output_dir}/n{len(dataset)}", tokenizer=model.tokenizer, heads=group_b_heads,
                                      head_values=head_values, group="B")

    # patch for group C heads
    print("Path patching for group C ...")
    patching_results = maybe_patch_or_load_cache(
        f"{args.output_dir}/n{len(dataset)}/pp_groupC.npy",
        get_path_patch_head_to_heads,
        receiver_heads=group_b_heads, receiver_input="v", model=model,
        clean_tokens=dataset['base_tokens'],
        corrupted_tokens=dataset['source_tokens'],
        label_tokens=list(dataset['labels']),
        last_token_pos=np.array(dataset["base_last_token_indices"]),
        sender_receiver_pos=np.array(dataset["base_last_token_indices"])-2,
        args=args
    )
    head_labels = list(range(len(patching_results[0])))
    fig = plot_patching_results(patching_results, head_labels,
                                f"Path Patching {args.model} groupC, {args.ops_order=}, {args.query_ops_order=}",
                                labels={"x": "Head", "y": "Layer", "color": "(patch - clean) / clean"})
    plotly.offline.plot(fig, filename=f"{args.output_dir}/n{len(dataset)}/pp_groupC.html", auto_open=False)
    # group_c_heads, head_values = compute_topk_components(
    #     patching_scores=torch.tensor(patching_results), k=args.n_groupC, largest=False, return_values=True, top_p=args.top_p
    # )
    group_c_heads, head_values = compute_topk_components_knee(torch.tensor(patching_results), 
                                    largest=False, return_values=True, 
                                    knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"}, 
                                    return_knee=False)
    print("\ngroup_c_heads")
    print(group_c_heads)
    print()
    # group_c_heads, head_values = ([[12, 3], [8, 1], [14, 4], [10, 5], [12, 0], [6, 2]], [-0.49024349451065063, -0.444313108921051, -0.24101078510284424, -0.13956013321876526, -0.10947693884372711, -0.0635497123003006])
    if attn_scores is not None:
        visualize_top_heads_attention(attn_scores, dataset['base_tokens'][0], dataset['base_last_token_indices'][0],
                                      output_dir=f"{args.output_dir}/n{len(dataset)}", tokenizer=model.tokenizer, heads=group_c_heads,
                                      head_values=head_values, group="C", seq_pos=final_token_position-2)

    # patch for group D heads
    print("Path patching for group D ...")
    prev_query_box_id_pos_list = [find_previous_query_box_pos(dataset[i]) for i in range(len(dataset))]
    patching_results = maybe_patch_or_load_cache(
        f"{args.output_dir}/n{len(dataset)}/pp_groupD.npy",
        get_path_patch_head_to_heads,
        receiver_heads=group_c_heads, receiver_input="v", model=model,
        clean_tokens=dataset['base_tokens'],
        corrupted_tokens=dataset['source_tokens'],
        label_tokens=list(dataset['labels']),
        last_token_pos=list(dataset["base_last_token_indices"]),
        sender_receiver_pos=prev_query_box_id_pos_list,
        args=args
    )
    head_labels = list(range(len(patching_results[0])))
    fig = plot_patching_results(patching_results, head_labels,
                                f"Path Patching {args.model} groupD, {args.ops_order=}, {args.query_ops_order=}",
                                labels={"x": "Head", "y": "Layer", "color": "(patch - clean) / clean"})
    plotly.offline.plot(fig, filename=f"{args.output_dir}/n{len(dataset)}/pp_groupD.html", auto_open=False)
    # group_d_heads, head_values = compute_topk_components(
    #     patching_scores=torch.tensor(patching_results), k=args.n_groupD, largest=False, return_values=True, top_p=args.top_p
    # )  # right now k=5,
    group_d_heads, head_values = compute_topk_components_knee(torch.tensor(patching_results), 
                                    largest=False, return_values=True, 
                                    knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"}, 
                                    return_knee=False)
    print("\ngroup_d_heads")
    print(group_d_heads)
    print()
    # group_d_heads, head_values = ([[11, 5], [11, 2], [9, 3], [7, 1], [8, 0]], [-0.08275768905878067, -0.025827284902334213, -0.02450619451701641, -0.022663826122879982, -0.021986696869134903])
    if attn_scores is not None:
        for prev_query_box_id_pos in prev_query_box_id_pos_list[0]:
            visualize_top_heads_attention(attn_scores, dataset['base_tokens'][0], dataset['base_last_token_indices'][0],
                                          output_dir=f"{args.output_dir}/n{len(dataset)}", tokenizer=model.tokenizer, heads=group_d_heads,
                                          head_values=head_values, group="D", seq_pos=prev_query_box_id_pos)



if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print(f"ARGS: {args}")
    post_arg_parse_fix(args)
    fix_random_seed(args.seed)
    path_patch_abcd_circuit(args)


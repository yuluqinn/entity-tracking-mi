import os
# set up logging
import logging
from functools import partial
import pdb
import h5py  # was `from wandb.old.summary import h5py`; wandb>=0.22 removed wandb.old
# ignore warnings
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)
import time
import argparse
from typing import List

import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import nnsight
from nnsight import LanguageModel

from src.dataset import ProbeDataLoader, LMDataloader, GPTDataloaderForInference, ObjectLocationProbeDataLoader, BinaryProbeDataLoader, SpanProbeDataLoader, PhraseProbeDataLoader, IncrementalLocalStateProbeDataLoader, GPTDataloaderForIncrementalLocalState, MentionedProbeDataLoader
from src.model import T5ForProbing, GPTForProbing, LlamaForProbing
from src.probe_trainer import Trainer, TrainerConfig, Mention_Trainer
from src.probe_model import BatteryProbeClassification, ObjectLocationProbeClassification, BatteryProbeClassificationTwoLayer
from src.probing_utils import get_token_pos_given_span_types, get_object_mapping, get_quantization_config


import sys
sys.path.append(".")
from utils import fix_random_seed, free_gpu_cache, get_basis_directions, pad_batch_collate_fn, setup_nnsight



_MAX_SOURCE_TEXT_LENGTH = {
    "t5": 512,
    "gpt": 512,
    "llama": 2048,
    "Llama-3.1-8B": 4096,
    "Llama-3.1-70B": 2048,
    "Llama-3.1-405B": 2048,
    "CodeLlama-13b-hf": 4096,
    "gemma-2-2b": 8192,
    "Qwen3-14B": 5120,
    "Qwen3-VL-8B-Instruct": 4096,
    "InternVL3_5-8B": 4096,
    "Molmo2-8B": 4096,
    "llava-onevision-qwen2-7b-ov-hf": 4096
}

_MAX_TARGET_TEXT_LENGTH = 100


_INPUT_DIMENSIONS = {
    "t5": 768,
    "gpt": 1600,
    "llama": 5120,  # codellama13b
    "Llama-3.1-8B": 4096,
    "Llama-3.1-70B": 8192,
    "Llama-3.1-405B": 16384,
    "CodeLlama-13b-hf": 5120,
    "gemma-2-2b": 2304,
    "Qwen3-14B": 5120,
    "Qwen3-VL-8B-Instruct": 4096,  # text tower hidden size
    "InternVL3_5-8B": 4096,        # qwen3 text tower, 36 layers
    "Molmo2-8B": 4096,             # molmo2_text tower, 36 layers
    "llava-onevision-qwen2-7b-ov-hf": 3584  # qwen2 text tower, 28 layers
}

# shared VLM support table (trc / loader / unwrap flags) — see src/probing_utils.py
from src.probing_utils import VLM_REGISTRY as _VLM_REGISTRY

# make deterministic
torch.manual_seed(0)

LARGE_FILE_LIMIT = 10000  # if more than this number of activations to cache, use h5


def load_act_containers_from_box_model_repo(args):
    act_container_train, act_container_test = [], []    
    files = os.listdir(args.model_representation_path)
    use_torch = any(f.endswith(".pt") for f in files)
    if use_torch:
        train_rep_path = os.path.join(args.model_representation_path, "representations_train.pt")
        test_rep_path = os.path.join(args.model_representation_path, "representations_test.pt")
    else:
        train_rep_path = os.path.join(args.model_representation_path, "representations_train.p")
        test_rep_path = os.path.join(args.model_representation_path, "representations_test.p")
    # pdb.set_trace(header="debuging load act")
    with open(train_rep_path, "rb") as rep_f:
        if use_torch:
            act_all_container_train = torch.load(rep_f)
        else:
            
            act_all_container_train = pickle.load(rep_f)
    # print(f"shape of act_all_container_train: {len(act_all_container_train)}, each of shape {len(act_all_container_train[0])}")
    
    
    if type(act_all_container_train) is list:
        # When using codellama
        for act in act_all_container_train:
            # cast: caches saved from bf16 models must match the fp32 probe
            act_container_train.append(act[args.layer - 1].to(torch.float32))
            # print(f"each act_container_train shape: {act[args.layer - 1].shape}")
        act_all_container_train.clear()
    elif type(act_all_container_train) is torch.Tensor:
        # Just used for the old activation files transferred from nnsight local-dev branch
        # For the remote-executed ones, they should be lists of tensors already
        act_container_train = act_all_container_train.permute(1,0,2)[args.layer - 1].to(torch.float32)
        del act_all_container_train
    
    

    with open(test_rep_path, "rb") as rep_f:
        if use_torch:
            act_all_container_test = torch.load(rep_f)
        else:
            act_all_container_test = pickle.load(rep_f)

    # print(f"shape of act_all_container_test: {len(act_all_container_test)}, each of shape {len(act_all_container_test[0])}")
    if type(act_all_container_test) is list:
        # When using codellama
        for act in act_all_container_test:
            act_container_test.append(act[args.layer - 1].to(torch.float32))
        act_all_container_test.clear()
    elif type(act_all_container_test) is torch.Tensor:
        act_container_test = act_all_container_test.permute(1,0,2)[args.layer - 1].to(torch.float32)
        del act_all_container_test
    return act_container_train, act_container_test

def read_caching_history(ckpt_folder_path: str, load_act: bool):
    # read existing caching history
    num_examples_cached = 0
    num_batch_cached = 0
    act_all_container = []
    if os.path.exists(ckpt_folder_path):
        for folder_name in tqdm(os.listdir(ckpt_folder_path), desc="reading caching history"):
            if folder_name.startswith("batch_idx_"):
                # parse batch idx and batch size
                parts = folder_name.split("_")
                batch_idx = int(parts[2])
                batch_size = int(parts[5])
                num_examples_cached += batch_size
                num_batch_cached += 1
                if load_act:
                    # load cached activations
                    batch_ckpt_folder_path = os.path.join(ckpt_folder_path, folder_name)
                    with open(os.path.join(batch_ckpt_folder_path, "act_all_container_elems.p"), "rb") as f:
                        act_all_container_elems = pickle.load(f)
                    for elem in act_all_container_elems:
                        act_all_container.append(elem)
                        
    return num_examples_cached, num_batch_cached, act_all_container

def write_caching_history(ckpt_folder_path: str, batch_idx, batch_size: int, act_all_container_elems:list):
    folder_name = f"batch_idx_{batch_idx}_batch_size_{batch_size}"
    batch_ckpt_folder_path = os.path.join(ckpt_folder_path, folder_name)
    if not os.path.exists(batch_ckpt_folder_path):
        os.makedirs(batch_ckpt_folder_path, exist_ok=True)
    with open(os.path.join(batch_ckpt_folder_path, "info.txt"), "w") as f:
        f.write(f"Batch Index: {batch_idx}\nBatch Size: {batch_size}\n")
    # pdb.set_trace(header="saving ndif remote activations")
    # print estimated size of act_all_container_elems to be saved
    
    with open(os.path.join(batch_ckpt_folder_path, "act_all_container_elems.p"), "wb") as f:
        pickle.dump(act_all_container_elems, f)
    
def load_ndif_cached_activations(ckpt_folder_path: str, layer_idx: int):
    # pdb.set_trace(header="loading ndif cached activations")
    act_container = []
    if os.path.exists(ckpt_folder_path):
        num_batches = len(os.listdir(ckpt_folder_path))
        # make sure loading in correct order
        for batch_idx in tqdm(range(num_batches), desc="loading ndif cached activations"):
            folder_name = f"batch_idx_{batch_idx}_batch_size_"
            # find the folder that starts with this name
            matched_folder = None
            for fn in os.listdir(ckpt_folder_path):
                if fn.startswith(folder_name):
                    matched_folder = fn
                    break
            if matched_folder is None:
                raise ValueError(f"Could not find folder for batch idx {batch_idx} in {ckpt_folder_path}")
            batch_ckpt_folder_path = os.path.join(ckpt_folder_path, matched_folder)
            with open(os.path.join(batch_ckpt_folder_path, "act_all_container_elems.p"), "rb") as f:
                act_all_container_elems = pickle.load(f)
            for elem in act_all_container_elems:
                if len(elem) == 0:
                    act_container.append([])
                else:
                    
                    act_container.append(elem[layer_idx - 1].to(torch.float32))  # layer_idx is 1-indexed
                        
    return act_container

def get_activations_from_data(act_all_container, act_container, args, end_idx, inference_device, dataloader, model, tokenizer, object_list, split: str):
        # save batched results to a separate file just in case it get stuck
    if not os.path.exists(args.model_representation_path):
        os.makedirs(args.model_representation_path, exist_ok=True)
    # pdb.set_trace(header="saving rep")
    ckpt_folder_path = os.path.join(args.model_representation_path, f'ckpt_{split}')
    if not os.path.exists(ckpt_folder_path):
        os.makedirs(ckpt_folder_path, exist_ok=True)
    num_examples_cached, num_batch_cached, _ = read_caching_history(ckpt_folder_path, load_act=False)

    pooling_boundaries = None
    if getattr(args, "pooling", "last") == "mean_context":
        # mean over context tokens only: boundary = first token of the query phrase " Box k contains"
        assert args.caching_batch_size == 1, "mean_context pooling requires caching_batch_size=1 (left-padding would poison the mean)"
        ds = dataloader.dataset
        base_ds = ds.dataset if isinstance(ds, torch.utils.data.Subset) else ds
        texts = list(base_ds.prefix_text)
        if isinstance(ds, torch.utils.data.Subset):
            texts = [texts[j] for j in ds.indices]
        # context tokenization is a prefix of the full tokenization (verified: sentence
        # boundary "." + " Box" tokenize independently for the supported tokenizers)
        pooling_boundaries = [len(tokenizer(t[:t.rfind(" Box ")])["input_ids"]) for t in texts]

    for batch_idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="caching activations"):
        # if bnatch_idx < num_batch_cached or this batch already cached, skip
        # assert num_examples_cached % args.caching_batch_size == 0, "cached examples not aligned with caching batch size"
        # pdb.set_trace(header="checking caching progress")
        is_incremental_state_data = False
        if batch_idx < num_batch_cached:
            print(f"Skipping already cached batch {batch_idx} ...")
            continue
            
        # if num_examples_met <= num_examples_cached:
        #     print(f"Skipping already cached batch {batch_idx} ...")
        #     continue

        
        ids = data['prefix_ids'].to(inference_device, dtype=torch.long)
        mask = data['prefix_attn_masks'].to(inference_device, dtype=torch.long)
        # pdb.set_trace(header="checking dataloader output")
        box_id_positions = data['box_id_positions_flattened'] if 'box_id_positions_flattened' in data else None # used to identify whether it's incremental state probe or not. In this case we're not caching a single token position. 
        is_incremental_state_data = box_id_positions is not None
        
        
        
        if end_idx is not None:
        
            ids = ids[0][:end_idx].unsqueeze(0)
            mask = mask[0][:end_idx].unsqueeze(0)
        # pdb.set_trace(header="after slicing ids and mask")
        if "-" in args.condition_on:
            batch_token_pos = [get_token_pos_given_span_types(input_ids, tokenizer, args.condition_on, object_list) for input_ids in ids]
            
        elif "period_comma" in args.condition_on:
            batch_token_pos = [get_token_pos_given_span_types(input_ids, tokenizer, args.condition_on, object_list) for input_ids in ids]
            if "period_comma_prior" in args.condition_on:
                batch_token_pos = [[pos-1 for pos in token_pos] for token_pos in batch_token_pos]
        elif "number_all" in args.condition_on or "object_all" in args.condition_on:
            batch_token_pos = [get_token_pos_given_span_types(input_ids[:-4], tokenizer, args.condition_on, object_list) for input_ids in ids]
        else:
            batch_token_pos = [-1] *len(ids)
            
        if is_incremental_state_data:
            batch_token_pos = [box_id_positions]
            batch_token_pos = [[int(btp) for btp in _] for _ in batch_token_pos]
        if args.save_model_representation:
            # pdb.set_trace(header="entering ndif remote")
            if any([isinstance(model, c) for c in [LlamaForProbing, GPTForProbing, T5ForProbing]]):
                # pdb.set_trace(header="entering llama/gpt/t5")
                act = model(input_ids=ids, attention_mask=mask, return_all_layers=True) # list(tensor(batch_size, padded_max_len_seq, hidden_dim) * num_layers)
                # pdb.set_trace(header="got llama/gpt/t5 activations")
            elif args.ndif_remote or is_incremental_state_data:
                # TODO: get model activation for this example
                # raise NotImplementedError
                num_sample = ids.shape[0]
                num_layers = len(model.model.layers) + 1 # plus embedding layer
                hidden_size = model.model.config.hidden_size
                seq_len = ids.shape[1]
                layer_idx = args.layer
                condition_on = args.condition_on
                # pdb.set_trace(header="entering ndif remote")
                with model.trace({
                    "input_ids": ids,
                    "attention_mask": mask
                }, remote = (args.ndif_remote or is_incremental_state_data)) as tr:
                    stacked_hs = torch.zeros((num_layers, num_sample, seq_len, hidden_size))# hardcode [num_layers + 1, batch_size, hidden_size]
                    stacked_hs[0] = model.model.layers[0].input.detach().cpu()  # input of the first layer, B, SeqL, D
                    act_all_container_elems = [].save()
                    act_container_elems = [].save()
                    for idx, layer in enumerate(model.model.layers):
                        stacked_hs[idx + 1] = layer.output.detach().cpu()
                        
                    # format as act 
                    # pdb.set_trace(header="check output")
                    act = [stacked_hs[i] for i in range(num_layers)]# should be good, need to check shape with other models thought
                    for i in range(len(ids)):
                        # might be a little bit inefficient but whatever...
                        # shape of a: should be num+samples, seq_len, hidden_dim
                            # pdb.set_trace(header="checking batch size > 1 caching ndif remote")
                        act_all_container_elem = [a[i, batch_token_pos[i], :].clone().detach().cpu().unsqueeze(0) for a in act]
                        act_container_elem = act[layer_idx - 1][i, batch_token_pos[i], :].clone().detach().cpu().unsqueeze(0)
                        act_all_container_elems.append(act_all_container_elem)
                        act_container_elems.append(act_container_elem)
                        if "-" in condition_on or "_" in condition_on:
                            # add 6 empty caches so the total number of datapoint matches with original data without subseting
                            for _ in range(6):
                                act_all_container_elems.append([])
                                act_container_elems.append([])
                # save to checkpoint folder
                # pdb.set_trace(header="saving ndif remote caching history")
                write_caching_history(ckpt_folder_path, batch_idx, len(ids), act_all_container_elems) # for inc, its [[1, n_tks, hdm] * n_layers]
                        

            else: # hf models
                act = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states

            for i in range(len(ids)):

                if not args.ndif_remote:
                    if pooling_boundaries is not None:
                        b = pooling_boundaries[batch_idx * args.caching_batch_size + i]
                        start = 1 if (tokenizer.bos_token_id is not None and ids[i][0].item() == tokenizer.bos_token_id) else 0
                        assert b > start and b < ids.shape[1], f"bad pooling boundary {b} for seq len {ids.shape[1]}"
                        act_all_container.append([a[i, start:b, :].to(torch.float32).mean(dim=0).detach().cpu().unsqueeze(0) for a in act])
                        act_container.append(act[args.layer - 1][i, start:b, :].to(torch.float32).mean(dim=0).detach().cpu().unsqueeze(0))
                    else:
                        act_all_container.append([a[i, batch_token_pos[i], :].detach().cpu().unsqueeze(0) for a in act])
                        act_container.append(act[args.layer - 1][i, batch_token_pos[i], :].detach().cpu().unsqueeze(0))
                else:
                    act_all_container.append(act_all_container_elems[i])
                    act_container.append(act_container_elems[i])
                # estimate act_all_container size
                if "-" in args.condition_on or "_" in args.condition_on:
                # add 6 empty caches so the total number of datapoint matches with original data without subseting
                    for _ in range(6):
                        act_all_container.append([])
                        act_container.append([])
        else:
            # last hidden state
            # NDIF not implemented
            if [isinstance(model, c) for c in [LlamaForProbing, GPTForProbing, T5ForProbing]]:
                act = model(input_ids=ids, attention_mask=mask, return_all_layers=True)
            else: # hf models
                act = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states
            act_container.append(act[args.layer-1][0, batch_token_pos, :].detach().cpu())

    


def save_representation(rep: List, split:str, args):
    # For gpt2, 21 examples, rep is a list of length 21, item of index // 7 is a list of shape 49(num layers), each item is a tensor of shape 1, num_saved_pos, 1600
    if not os.path.exists(args.model_representation_path):
        os.makedirs(args.model_representation_path, exist_ok=True)
    # pdb.set_trace(header="saving rep")
    subset_str = '_subset' if (split == 'train' and args.dataset_subset) or (split == 'test' and (args.dataset_subset or args.dataset_subset_test_only)) else ''
    if len(rep) > LARGE_FILE_LIMIT:
        rep_path = os.path.join(args.model_representation_path, f"representations_{split}{subset_str}")
        # having rank-specific path (as opposed to only have rank-0 save) works, but not sure if rank-0 save only works
        # rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        # rep_path_rank = os.path.join(rep_path, f"rank{rank}")
        # os.makedirs(rep_path_rank, exist_ok=True)
        os.makedirs(rep_path, exist_ok=True)

        num_layers = len(rep[0])
        # open close 1 file at a time (less efficient than all files open, but that cause large file save to fail in 
        # in scc with torch run distributed process
        # create groups once (optional) by touching files first
        for i in range(num_layers):
            p = os.path.join(rep_path, f"representations_l{i + 1}.h5")
            if not os.path.exists(p):
                with h5py.File(p, "w") as f:
                    f.create_group("activations")

        # then for each activation, open-append-close per write
        for act_i, act in tqdm(enumerate(rep), total=len(rep), desc=f"Saving layer-wise activations"):
            if len(act) == 0:
                continue
            for layer_i in range(num_layers):
                p = os.path.join(rep_path, f"representations_l{layer_i + 1}.h5")
                with h5py.File(p, "a") as f:  # 'a' == read/write, create if missing
                    grp = f["activations"]
                    grp.create_dataset(f"activations_{act_i}", data=act[layer_i][0].numpy(), compression="gzip")

        # OLD, save all layers in 1 file
        # rep_path = os.path.join(args.model_representation_path, f"representations_{split}.h5")
        # with h5py.File(rep_path, "w") as f:
        #     group = f.create_group('activations')
        #     for i, act in tqdm(enumerate(rep), total=len(rep)):
        #         if len(act) > 0:  # every 7 exmaples is non empty
        #             group.create_dataset(f"activations_{i}", data=torch.cat(act).numpy())
    else:
        rep_path = os.path.join(args.model_representation_path, f"representations_{split}{subset_str}.p")
        with open(rep_path, "wb") as rep_f:
            pickle.dump(rep, rep_f)
            rep.clear()


def fsdp_model(args):
    if "llama" in args.model_path.lower():
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    else:
        raise NotImplementedError("FSDP layering not implemented for non-llama model yet")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load model on CPU first (important!)
    config = AutoConfig.from_pretrained(args.model_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config) #,device_map=None)

    for name, p in model.named_parameters(recurse=True):
        if p.device.type != "meta":
            setattr(model, name, p.to("meta"))

    # diagnostic — run immediately AFTER init_empty_weights() and model creation
    bad = []
    for name, p in model.named_parameters(recurse=True):
        if p.device.type != "meta":
            bad.append((name, p.device, p.size()))
    for name, b in model.named_buffers(recurse=True):
        if b.device.type != "meta":
            bad.append(("buffer:" + name, b.device, b.size()))

    if bad:
        print("Found non-meta tensors BEFORE to_empty():")
        for n, dev, sz in bad[:20]:
            print(n, dev, sz)
        raise SystemError("Some parameters/buffers were materialized before to_empty() — fix the creation pipeline.")
    else:
        print("All params & buffers are meta. Safe to call to_empty().")
    # pdb.set_trace()

    # model.to_empty(device=device)

    # Create auto-wrap policy
    auto_wrap_policy = partial(transformer_auto_wrap_policy,transformer_layer_cls={LlamaDecoderLayer})

    # Wrap in FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False),
        use_orig_params=True,
        device_id=device,
        param_init_fn=lambda module: None,  # <---- prevents reset_parameters()
    )

    model = load_checkpoint_and_dispatch(
        model,
        args.model_path,
        device_map={"": f"cuda:{local_rank}"},
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.bfloat16,
        offload_state_dict=True,
    )

    return model


def main():

    parser = argparse.ArgumentParser(description='Train classification network')
    parser.add_argument("--model_type",
                        required=True,
                        choices=_INPUT_DIMENSIONS.keys(),
                        help=f"{_INPUT_DIMENSIONS.keys()} supported.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model with 8-bit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model with 4-bit")
    parser.add_argument("--dataset_path",
                        required=True,
                        type=str)
    parser.add_argument("--dataset_subset",
                        dest='dataset_subset',
                        action='store_true'
                        )
    parser.add_argument("--dataset_subset_test_only",
                        dest='dataset_subset_test_only',
                        action='store_true'
                        )
    parser.add_argument('--dataset_loader_num_workers',
                        default=4,
                        type=int)
    parser.add_argument("--model_path",
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('--checkpoint_root',
                        default="./probe_checkpoints", type=str)
    parser.add_argument(
            "--object_vocabulary_file",
            type=str,
            default="data/objects/objects_with_bnc_frequency.csv",
            help='Path to a .csv file with a string field "object_names".')
    parser.add_argument('--layer',
                        required=True,
                        default=-1,
                        type=int)
    parser.add_argument('--epo',
                        default=16,
                        type=int)
    parser.add_argument('--condition_on', 
                        choices=["box", "period", "the", "number", "contains",
                                 "number-object-remove", "number-object-exist",  # span probs
                                 "number-remove", "object-remove",  # span probs (using only one of the tokens)
                                 "period_comma_local", "period_comma_cumulative", # period or comma, only 1 token, ternary probe
                                 "period_comma_prior_local", "period_comma_prior_cumulative", # 1 token before period or comma (box id), only 1 token , ternary probe (should use number_all)
                                 "number_all_local", "number_all_cumulative", # ternary probe, condition on all box ids (1 at a time)
                                 "object_all_local", "object_all_cumulative", # ternary probe, condition on all objects (1 at a time)
                                 ],
                        type=str,
                        dest='condition_on',
                        default='number')
    parser.add_argument('--box_label_base',
                        type=int,
                        default=0,
                        help="lowest box number in the dataset (0 = original boxes 0-6; 1 = vlm-data boxes 1-7). "
                             "Used by the global-probe labels and the state-matrix parser")
    parser.add_argument('--cache_prefix_field',
                        type=str,
                        default=None,
                        help="dataset column to use as the model input at caching time (e.g. a "
                             "chat-template-wrapped prompt); label parsing still reads prefix/sentence. "
                             "Default: use the prefix column (original behavior)")
    parser.add_argument('--pooling',
                        type=str,
                        choices=['last', 'mean_context'],
                        default='last',
                        help="representation pooling at caching time: 'last' = the conditioned token "
                             "(original behavior); 'mean_context' = mean over context tokens (description "
                             "+ operations, through the final period before the query phrase), BOS excluded")
    parser.add_argument('--incremental_local_state',
                        dest='incremental_local_state',
                        action='store_true',
                        help="A flag to indicate incremental local state probes.")
    parser.add_argument('--mention',
                        dest='mention',
                        action='store_true',
                        help="A flag to indicate mentioned object probes.")
    parser.add_argument('--max_train_data',
                        type=int,
                        default=None)
    parser.add_argument('--max_test_data',
                        type=int,
                        default=None)
    parser.add_argument('--num_prior_state',
                        default=-1,
                        type=int)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3)
    parser.add_argument("--overwrite_cache",
                        dest='overwrite_cache',
                        action='store_true')

    # span probe args (when condition_on=="number-object")
    parser.add_argument("--expand_query_box",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="whether to cache query box id token (in addition to other latest tokens)")
    parser.add_argument("--balance_label_sampling",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="whether to balance label distribution by downsampling no-remove cases")
    parser.add_argument('--same_phrase_only',
                        choices=['train', 'test', 'both', 'neither'],
                        default='neither',
                        type=str,
                        )
    
    parser.add_argument('--from_box_model',
                        dest='from_box_model', 
                        action='store_true',
                        help="whether to load activations from the box model repo instead of running the model forward here to get activations"
                        )

    # Non-linear probe
    parser.add_argument('--mid_dim',
                        default=128,
                        type=int)
    parser.add_argument('--twolayer',
                        dest='twolayer', 
                        action='store_true')
    parser.add_argument('--object_location',
                        dest='object_location', 
                        action='store_true')    
    parser.add_argument('--binary_probe',
                        dest='binary_probe', 
                        action='store_true')
    parser.add_argument('--random',
                        dest='random', 
                        action='store_true')
    parser.add_argument('--eval_only',
                        dest='eval_only', 
                        action='store_true')
    parser.add_argument('--exclude_empty',
                        dest='exclude_empty', 
                        action='store_true')
    parser.add_argument('--condition_on_obj',
                        default=0,
                        type=int)
    parser.add_argument('--debug_train',
                        action='store_true')

    parser.add_argument('--model_representation_path',
                        default=None,
                        type=str)
    parser.add_argument('--standardize_inputs',
                        action='store_true',
                        help='z-score each activation dimension with the TRAIN split mean/std before the probe '
                             '(applied to train and test; default off = released recipe, raw activations)')

    parser.add_argument('--save_model_representation',
                        dest="save_model_representation",
                        action="store_true")
    
    parser.add_argument('--load_model_representation',
                        dest="load_model_representation",
                        action="store_true")

    parser.add_argument('--include_prompt',
                        type=str,
                        default=False,
                        choices=[False, "PROMPT", "PROMPT_ALTFORM", "PROMPT_ALLBOX_ALTFORM", "INSTRUCTION", ]
                        )
    
    parser.add_argument(
        "--chat", action="store_true", help="format prompt into chat templates"
    )
    
    # distributed inference ( for caching embedding) 
    parser.add_argument('--distributed',
                        dest="distributed",
                        action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int)
    parser.add_argument('--fsdp',
                        dest="fsdp", # TODO attempts to use fsdp to load llama 70b into 4GPU runs, still failing
                        action="store_true")
    parser.add_argument('--ndif_remote',
                        dest="ndif_remote",
                        action="store_true",
                        help="whether to use NDIF remote option to save model hiddens"
                        )
    parser.add_argument('--from_torch', help="whether the cached activations to load are in pt", action="store_true")
    
    parser.add_argument('--caching_batch_size',
                        default=1,
                        type=int)

    args, _ = parser.parse_known_args()

    if (args.condition_on not in ["number", "contains"]) and args.model_type == 't5':
        raise ValueError("--condition_on must be set to 'number' or 'contains' when training a probe on T5.")
    # if args.eval_only:
    #     # TODO(Sebastian): debug eval_only
    #     raise ValueError("--eval_only is buggy, do not use for now")

    if args.exclude_empty and not args.binary_probe:
        raise ValueError("--exclude_empty only works with --binary_probe")

    if args.exclude_empty and args.condition_on not in ["contains", "the"]:
        raise ValueError("--exclude_empty can only be used with --condition_on 'contains' or 'the'")
    
    # if args.condition_on in ["contains", "the"] and not args.exclude_empty:
    #     raise ValueError("--condition_on 'contains' or 'the' can only be used with --exclude_empty")

    if args.save_model_representation and args.model_representation_path is None:
        raise ValueError("--save_model_representation requires --model_representation_path to be set")
    
    if args.load_model_representation and args.model_representation_path is None:
        raise ValueError("--load_model_representation requires --model_representation_path to be set")
    
    if args.load_model_representation and args.save_model_representation:
        raise ValueError("--load_model_representation and --save_model_representation cannot be used together")

    if args.max_train_data is not None:
        assert args.max_train_data % 7 == 0, "number of data points must be divisible by 7"

    if args.max_test_data is not None:
        assert args.max_test_data % 7 == 0, "number of data points must be divisible by 7"

    if args.dataset_subset or args.dataset_subset_test_only:
        # pdb.set_trace(header="checking dataset subset options")
        assert os.path.exists(os.path.join(args.dataset_path,f'test-subsample-states-{"t5" if args.model_type == "t5" else "gpt"}.jsonl'))
        if "movecontent" in args.dataset_path.lower() or "move_content" in args.dataset_path.lower():
            assert os.path.exists(os.path.join(args.dataset_path, f'train-subsample-states-mask.p'))

    if args.fsdp:
        assert args.distributed, "--fsdp requires --distributed"

    if args.ndif_remote:
        setup_nnsight()

    folder_name = f"probing/state"
    if args.include_prompt:
        folder_name = folder_name + f"_fs{args.include_prompt}"
    if args.chat:
        folder_name = folder_name + f"_chat"
    if args.twolayer:
        folder_name = folder_name + f"_tl{args.mid_dim}"  # tl for probes without batchnorm
    if args.random:
        folder_name = folder_name + "_random"
    if args.object_location:
        folder_name = folder_name + "_object_location"
    if args.binary_probe:
        folder_name = folder_name + "_binary"
    if args.exclude_empty:
        folder_name = folder_name + "_exclude_empty"
    if args.condition_on_obj > 0:
        folder_name = folder_name + f"_condition_on_obj_{args.condition_on_obj}"
    if args.num_prior_state != -1:
        folder_name = folder_name + f"_prior_state_{args.num_prior_state}"
    if "-" in args.condition_on:
        folder_name = folder_name + f"_span_{args.condition_on}"
    if "_" in args.condition_on:
        folder_name = folder_name + f"_{args.condition_on}"
    if args.incremental_local_state:
        folder_name = folder_name + f"_incremental_local_state"
    if args.mention:
        folder_name = folder_name + f"_mention"
    if args.pooling != "last":
        folder_name = folder_name + f"_{args.pooling}"

    training_file = os.path.join(args.checkpoint_root, folder_name, f"layer{args.layer}_token1", "tensorboard.txt")
    if not args.overwrite_cache and os.path.exists(training_file) and len(open(training_file).readlines())>0:
        print(f"Found trained probe, skipping: {training_file}")
        exit(0)

    print(f"Running experiment for {folder_name}")
    print("[Data]: Reading data...\n")

    device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
    print(f'Training Device: {device}')


    # Load data
    data_type = "t5" if args.model_type == "t5" else "gpt"
    dataset_path_train = os.path.join(args.dataset_path, f'train{"-subsample-states" if args.dataset_subset else ""}-{data_type}.jsonl')
    print("Train dataset:", dataset_path_train)
    dataset_path_test = os.path.join(args.dataset_path, f'test{"-subsample-states" if (args.dataset_subset or args.dataset_subset_test_only) else ""}-{data_type}.jsonl')

    train_df = pd.read_json(dataset_path_train, orient='records', lines=True)
    test_df = pd.read_json(dataset_path_test, orient='records', lines=True)

    if args.eval_only:
        train_df = train_df.head(0)

    if args.model_type == "t5":
        train_df = train_df[["sentence_masked", "masked_content"]]
        test_df = test_df[["sentence_masked", "masked_content"]]

    # Load object names
    print(args.object_vocabulary_file)
    object_map, object_list = get_object_mapping(args.object_vocabulary_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=_VLM_REGISTRY.get(args.model_type, {}).get("trc", False))

    act_container_train = []
    act_all_container_train = []
    act_container_test = []
    act_all_container_test = []
    act_all_h5_train_dir = None
    act_all_h5_test_dir = None
    # TODO: need to add logic for loading mini-batches from ndif ckpts.
    if args.load_model_representation:

        # load pre-computed representations
        train_rep_path = os.path.join(args.model_representation_path, f"representations_train{'_subset' if args.dataset_subset and not args.from_box_model else ''}.p")
        test_rep_path = os.path.join(args.model_representation_path, f"representations_test{'_subset' if ((args.dataset_subset or args.dataset_subset_test_only) and not args.from_box_model) else ''}.p")

        act_all_h5_train_dir = train_rep_path.replace(".p", "")
        act_all_h5_test_dir = test_rep_path.replace(".p", "")
        if args.from_torch:
            train_rep_path = train_rep_path.replace(".p", ".pt")
            test_rep_path = test_rep_path.replace(".p", ".pt")
        
        
        if os.path.isdir(act_all_h5_train_dir): # and os.path.exists(exploded_train_path):
            print("h5 cache for train representation found, skipping loading ...")
        else:
            act_all_h5_train_dir = None
            if not args.ndif_remote and not args.from_box_model:
                print(f"loading cached train representations from {train_rep_path} ...")
                with open(train_rep_path, "rb") as rep_f:
                    act_all_container_train = pickle.load(rep_f)

                for act in act_all_container_train:
                    if len(act) == 0 and ("-" in args.condition_on or "_" in args.condition_on):
                        act_container_train.append([])
                    else:
                        # cast: caches saved from bf16 models must match the fp32 probe
                        act_container_train.append(act[args.layer - 1].to(torch.float32))
                        
            elif args.from_box_model: # load from box model repo, just copying the code from that repo here since it's a bit different from what's been doing here and I don't want to handle the conflicts rn.
                act_container_train, act_container_test = load_act_containers_from_box_model_repo(args)
                
                
            else:
                train_ckpt_folder_path = os.path.join(args.model_representation_path, f'ckpt_train')
                print(f"loading cached train representations from ndif ckpts in {train_ckpt_folder_path} ...")
                act_container_train = load_ndif_cached_activations(train_ckpt_folder_path, args.layer)
                

            act_all_container_train.clear()
            
        if os.path.exists(act_all_h5_test_dir): # and os.path.exists(exploded_test_path):
            print("h5 cache for test representation found, skipping loading ...")
        else:
            act_all_h5_test_dir = None
            if not args.ndif_remote and not args.from_box_model:
                print(f"loading cached test representations from {test_rep_path} ...")
                with open(test_rep_path, "rb") as rep_f:
                    act_all_container_test = pickle.load(rep_f)

                for act in act_all_container_test:
                    if len(act) == 0 and ("-" in args.condition_on or "_" in args.condition_on):
                        act_container_test.append([])
                    else:
                        act_container_test.append(act[args.layer - 1].to(torch.float32))
            elif args.from_box_model: # load from box model repo, just copying the code from that repo here since it's a bit different from what's been doing here and I don't want to handle the conflicts rn.
                act_container_train, act_container_test = load_act_containers_from_box_model_repo(args)
            else:
                test_ckpt_folder_path = os.path.join(args.model_representation_path, f'ckpt_test')
                print(f"loading cached test representations from ndif ckpts in {test_ckpt_folder_path} ...")
                act_container_test = load_ndif_cached_activations(test_ckpt_folder_path, args.layer)    
                    

            act_all_container_test.clear()

    else:
        # set up distributed inference
        if args.distributed:
            rank = int(os.environ.get("RANK",'0'))
            print(f"{rank=}")
            inference_device = torch.device(f"cuda:{rank}")
            if not args.fsdp:
                torch.cuda.set_device(inference_device)  # https://github.com/pytorch/pytorch/issues/146767
            torch.distributed.init_process_group("nccl", device_id=inference_device)
            tp_plan = "auto"
        else:
            inference_device = device
            tp_plan = None
        model_kwargs = {"tp_plan": tp_plan}

        # Load model to compute representations
        if args.distributed:
            if args.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif args.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["quantization_config"] = get_quantization_config(args)
        if model_kwargs.get("quantization_config") is None and transformers.utils.is_torch_bf16_gpu_available():
            model_kwargs["torch_dtype"] = torch.bfloat16

        if args.model_type == "t5":
            model = T5ForProbing.from_pretrained(args.model_path)
        elif args.model_type == "gpt":
            model = GPTForProbing.from_pretrained(args.model_path)
        elif "llama" in args.model_type.lower() and not args.fsdp and not args.ndif_remote:
            model = LlamaForProbing.from_pretrained(args.model_path, **model_kwargs) #,device_map="auto",
        elif args.model_type in _VLM_REGISTRY and not args.fsdp and not args.ndif_remote:
            # VLMs probed text-only: hidden states come from the language tower through
            # the generic output_hidden_states branch below
            spec = _VLM_REGISTRY[args.model_type]
            from transformers import AutoModel, AutoModelForImageTextToText
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    args.model_path, trust_remote_code=spec["trc"], **model_kwargs)
            except (ValueError, KeyError):
                # custom-code repos (e.g. InternVLChatModel) may only map AutoModel
                model = AutoModel.from_pretrained(
                    args.model_path, trust_remote_code=spec["trc"], **model_kwargs)
            if spec["unwrap"]:
                # top-level forward needs pixel_values; the language tower alone is
                # exactly what text-only probing requires
                model = model.language_model
        elif args.fsdp:
            model = fsdp_model(args)
        elif args.ndif_remote:
            # TODO here, initialize ndif remote model
            model = LanguageModel(args.model_path, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs) #,device_map="auto",

        if model_kwargs.get("quantization_config") is None and not args.distributed and not args.ndif_remote:
            model = model.to(device)

        # Set probe layer (1-indexed)
        model.probe_layer = args.layer
        inference_device = model.device if not args.ndif_remote else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        # pdb.set_trace(header="111")
        # initialze LM test dataset
        if args.model_type == "t5":
            train_dataset = LMDataloader(train_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type], _MAX_TARGET_TEXT_LENGTH, "sentence_masked", "masked_content", include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
            test_dataset = LMDataloader(test_df, tokenizer, _MAX_SOURCE_TEXT_LENGTH[args.model_type], _MAX_TARGET_TEXT_LENGTH, "sentence_masked", "masked_content", include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj)
        else: #if args.model_type in ["gpt", "llama"]:
            if args.incremental_local_state:
                train_dataset = GPTDataloaderForIncrementalLocalState(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
                test_dataset = GPTDataloaderForIncrementalLocalState(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
            else:
                train_dataset = GPTDataloaderForInference(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, args=args)
                test_dataset = GPTDataloaderForInference(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, args=args)
        collate_fn = None
        if args.caching_batch_size > 1:
            # get custom collate fn for batching
            collate_fn = train_dataset.get_collate_fn() if hasattr(train_dataset, "get_collate_fn") else None

        # truncate dataset if needed
        if args.max_train_data is not None:
            train_dataset = torch.utils.data.Subset(train_dataset, range(args.max_train_data))
        if args.max_test_data is not None:
            test_dataset = torch.utils.data.Subset(test_dataset, range(args.max_test_data))

        if ("-" in args.condition_on or "_" in args.condition_on) and args.save_model_representation:
            # for span probe, we really just need the representation for each unique data point (disregarding query box)
            # so on caching runs, inference every 7 datapoints to get the representation to save some space
            train_dataset = torch.utils.data.Subset(train_dataset, range(0, len(train_dataset), 7))
            test_dataset = torch.utils.data.Subset(test_dataset, range(0, len(test_dataset), 7))
        # pdb.set_trace(header="checking dataloader before caching")
        loader_train = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=args.caching_batch_size, num_workers=1, collate_fn=collate_fn)
        loader_test = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=args.caching_batch_size, num_workers=1, collate_fn=collate_fn)

        # compute hidden representations  (deleted T5 ones)
        end_idx = None
        # I've checked that " n" between 0 and 7 are all single tokens.
        # But this is probably a patchy solution if box num >= 8
        if args.condition_on == "box":
            end_idx = -3  # originally -1
        # "Box" is one token, "Box 3" is 3 tokens
        elif args.condition_on == "period":
            end_idx = -4  # originally -2, our data ends with "contains", so one extra word, also 'Box 3' is 3 tokens (space is one)

        get_activations_from_data(act_all_container_train, act_container_train, args, end_idx, inference_device, loader_train, model, tokenizer, object_list, split="train")
        if args.save_model_representation:  # save them as we go, cheaper in memory
            if (not args.distributed or torch.distributed.get_rank() == 0) and not args.ndif_remote:
                # TODO NDIF representations is too large to save here, need to figure out how to do that later. so far just disable this branch to start caching test set
                save_representation(act_all_container_train, split="train", args=args)

        get_activations_from_data(act_all_container_test, act_container_test, args, end_idx, inference_device, loader_test, model, tokenizer, object_list, split="test")
        if args.save_model_representation:
            if (not args.distributed or torch.distributed.get_rank() == 0) and not args.ndif_remote:
                save_representation(act_all_container_test, split="test", args=args)

        if args.distributed:
            torch.distributed.destroy_process_group()

        print("caching done! existing program for now. re-run with load_model_representation! ")
        exit()



    probe_class = 8 if not args.binary_probe else 2
    input_dim = _INPUT_DIMENSIONS[args.model_type]
    if args.standardize_inputs:
        # per-dimension z-scoring with TRAIN statistics only; test rows use the same mu/sd (no leakage)
        assert all(torch.is_tensor(a) for a in act_container_train), "standardize_inputs expects cached per-row activation tensors"
        _stack = torch.stack([a.reshape(-1) for a in act_container_train])
        _mu, _sd = _stack.mean(0, keepdim=True), _stack.std(0, keepdim=True) + 1e-6
        act_container_train = [((a.reshape(1, -1) - _mu) / _sd).reshape(a.shape) for a in act_container_train]
        act_container_test = [((a.reshape(1, -1) - _mu) / _sd).reshape(a.shape) for a in act_container_test]
        print(f"standardized inputs with train stats: {len(act_container_train)} train / {len(act_container_test)} test rows, dim {_stack.shape[1]}")
        del _stack
    # pdb.set_trace(header="before initializing probing datasets")
    # if moveContent split, need to load the whole dataset, then apply subsample mask after computing prior states
    train_subset_mask, test_subset_mask = None, None
    if args.num_prior_state != -1 and ("movecontent" in dataset_path_train.lower() or "move_content" in dataset_path_train.lower()):
        dataset_path_train = dataset_path_train.replace("-subsample-states", "")
        dataset_path_test = dataset_path_test.replace("-subsample-states", "")
        with open(os.path.join(args.dataset_path, f'train-subsample-states-mask.p'), "rb") as rep_f:
            train_subset_mask = pickle.load(rep_f)
        with open(os.path.join(args.dataset_path, f'test-subsample-states-mask.p'), "rb") as rep_f:
            test_subset_mask = pickle.load(rep_f)
            
    # pdb.set_trace(header="before initializing probing datasets")
    if args.object_location:
        probing_dataset_train = ObjectLocationProbeDataLoader(act_container_train, dataset_path_train, max_data=args.max_train_data)
        probing_dataset_test = ObjectLocationProbeDataLoader(act_container_test, dataset_path_test, max_data=args.max_test_data)
    elif args.incremental_local_state:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=_VLM_REGISTRY.get(args.model_type, {}).get("trc", False))
        train_dataset = GPTDataloaderForIncrementalLocalState(train_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
        test_dataset = GPTDataloaderForIncrementalLocalState(test_df, tokenizer, max_length=_MAX_SOURCE_TEXT_LENGTH[args.model_type], include_empty=not args.exclude_empty, condition_on=args.condition_on, min_prev_objects=args.condition_on_obj, include_prompt=args.include_prompt, object_map=object_map)
        probing_dataset_train = IncrementalLocalStateProbeDataLoader(act_container_train, train_dataset)
        probing_dataset_test = IncrementalLocalStateProbeDataLoader(act_container_test, test_dataset)
    elif args.binary_probe:
        if "-" in args.condition_on:  # span probe
            probing_dataset_train = SpanProbeDataLoader(act_container_train, dataset_path_train, object_map,include_empty=not args.exclude_empty,min_prev_objects=args.condition_on_obj,max_data=args.max_train_data, tokenizer=tokenizer,expand_query_box=args.expand_query_box,balance_label_sampling=args.balance_label_sampling, span_probe_type=args.condition_on, args=args, split="train", same_phrase_only=args.same_phrase_only in ["train", "both"])
            probing_dataset_test = SpanProbeDataLoader(act_container_test, dataset_path_test, object_map,include_empty=not args.exclude_empty,min_prev_objects=args.condition_on_obj,max_data=args.max_test_data, tokenizer=tokenizer,expand_query_box=args.expand_query_box,balance_label_sampling=args.balance_label_sampling, span_probe_type=args.condition_on, args=args, split="test", same_phrase_only=args.same_phrase_only in ["test", "both"])
        elif args.mention:
            probing_dataset_train = MentionedProbeDataLoader(act_container_train, dataset_path_train, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj, box_label_base=args.box_label_base)
            probing_dataset_test = MentionedProbeDataLoader(act_container_test, dataset_path_test, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj, box_label_base=args.box_label_base)
        else:
            probing_dataset_train = BinaryProbeDataLoader(act_container_train, dataset_path_train, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj, max_data=args.max_train_data, local_operation_order=args.num_prior_state, subset_mask=train_subset_mask)
            probing_dataset_test = BinaryProbeDataLoader(act_container_test, dataset_path_test, object_map, include_empty=not args.exclude_empty, min_prev_objects=args.condition_on_obj, max_data=args.max_test_data, local_operation_order=args.num_prior_state, subset_mask=test_subset_mask)
    elif "_" in args.condition_on:  # phrase probes, takes 1 hidden state to predict 3 classes
        probing_dataset_train = PhraseProbeDataLoader(act_container_train, dataset_path_train, object_map,include_empty=not args.exclude_empty, max_data=args.max_train_data, tokenizer=tokenizer, args=args, split="train", activation_h5_path=act_all_h5_train_dir)
        probing_dataset_test = PhraseProbeDataLoader(act_container_test, dataset_path_test, object_map,include_empty=not args.exclude_empty, max_data=args.max_test_data, tokenizer=tokenizer, args=args, split="test", activation_h5_path=act_all_h5_test_dir)
    else:
        probing_dataset_train = ProbeDataLoader(act_container_train, dataset_path_train, object_map, max_data=args.max_train_data, box_label_base=args.box_label_base)
        probing_dataset_test = ProbeDataLoader(act_container_test, dataset_path_test, object_map, max_data=args.max_test_data, box_label_base=args.box_label_base)

    train_dataset, test_dataset = probing_dataset_train, probing_dataset_test
    
    if args.object_location:
        if args.twolayer:
            raise ValueError("Parameter --twolayer is not supported when using the object location probe.")
        probe = ObjectLocationProbeClassification(device,
            input_dim=input_dim,
            probe_class=probe_class,
            ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32))
    elif "-" in args.condition_on:
        if args.twolayer:
            probe = BatteryProbeClassificationTwoLayer(device,
                input_dim=input_dim*2 if args.condition_on.startswith("number-object") else input_dim,
                probe_class=probe_class,
                num_task=1,
                mid_dim=args.mid_dim,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                # dtype=probing_dataset_train.activations[0].dtype
                )
        else:
            probe = BatteryProbeClassification(device,
                input_dim=input_dim*2 if args.condition_on.startswith("number-object") else input_dim,
                probe_class=probe_class,
                num_task=1,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                # dtype=probing_dataset_train.activations[0].dtype
                )
    elif "_" in args.condition_on:
        if args.twolayer:
            probe = BatteryProbeClassificationTwoLayer(device,
                input_dim=input_dim,
                probe_class=3,  # ternary probe of [exist, non-exist, removed]
                num_task=7 * 100,  # not 8 because non-exist covers the case
                mid_dim=args.mid_dim,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )
        else:
            probe = BatteryProbeClassification(device,
                input_dim=input_dim,
                probe_class=3,   # ternary probe of [exist, non-exist, removed]
                num_task=7 * 100,  # not 8 because non-exist covers the case
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )
    else:
        if args.twolayer:
            probe = BatteryProbeClassificationTwoLayer(device,
                input_dim=input_dim,
                probe_class=probe_class,
                num_task=len(object_map),  # one task per vocabulary object (=100 for the original vocab)
                mid_dim=args.mid_dim,
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )
        else:
            probe = BatteryProbeClassification(device,
                input_dim=input_dim,
                probe_class=probe_class,
                num_task=len(object_map),  # one task per vocabulary object (=100 for the original vocab)
                ce_weights=probing_dataset_train.get_weights().to(device, dtype=torch.float32),
                )
            
    max_epochs = args.epo
    t_start = time.strftime("_%Y%m%d_%H%M%S")
    tconf = TrainerConfig(
        max_epochs=max_epochs, batch_size=1024, learning_rate=args.lr,#1e-3,
        betas=(.9, .999), 
        lr_decay=True, warmup_tokens=len(train_dataset)*5, 
        final_tokens=len(train_dataset)*max_epochs,
        num_workers=args.dataset_loader_num_workers, # originally 4, if loading h5 files during training, need more worker because IO is bottleneck
        weight_decay=0.,
        ckpt_path=os.path.join(args.checkpoint_root, folder_name, f"layer{args.layer}_token1"),
        debug_train=args.debug_train,
    )
    os.makedirs(tconf.ckpt_path, exist_ok=True)
    trainer = Trainer(probe, train_dataset, test_dataset, tconf) if not args.mention else Mention_Trainer(probe, train_dataset, test_dataset, tconf)
    if not args.eval_only:
        try:
            import wandb
            try:
                logged_in = bool(wandb.api.api_key)  # wandb<=0.21
            except Exception:
                logged_in = bool(os.environ.get("WANDB_API_KEY")) or os.path.exists(os.path.expanduser("~/.netrc"))
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "entity-tracking-probing"),
                name=f"{args.model_type}_{folder_name}_layer{args.layer}",
                mode=None if logged_in else "offline",  # offline when not logged in; sync later with `wandb sync`
                config={
                    "model_type": args.model_type,
                    "model_path": args.model_path,
                    "layer": args.layer,
                    "condition_on": args.condition_on,
                    "binary_probe": args.binary_probe,
                    "exclude_empty": getattr(args, "exclude_empty", None),
                    "standardize_inputs": args.standardize_inputs,
                    "max_epochs": tconf.max_epochs,
                    "batch_size": tconf.batch_size,
                    "learning_rate": args.lr,
                    "weight_decay": tconf.weight_decay,
                    "lr_decay": tconf.lr_decay,
                    "input_dim": input_dim,
                    "probe_class": probe_class,
                    "train_size": len(train_dataset),
                    "test_size": len(test_dataset),
                },
            )
        except Exception as e:
            print(f"wandb logging disabled: {e}")
        predictions_matrix = trainer.train(prt=True).astype(int)
        trainer.save_traces()
        trainer.save_checkpoint()
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
    else:
        trainer.load_checkpoint()
        predictions_matrix = trainer.predict(prt=True).astype(int)
    
    predictions_file = os.path.join(tconf.ckpt_path, "predictions.txt")

    if "-" in args.condition_on:
        predictions_df = pd.DataFrame({
            "input": test_dataset.analysis_strings,
            "prediction": predictions_matrix.squeeze().tolist(),
            "label": [i.item() for i in test_dataset.examples],
            "same_phrase": [i.item() for i in test_dataset.mentioned_objects],
        })
        predictions_df.to_json(predictions_file.replace(".txt", ".jsonl"),lines=True, orient="records")
    elif "_" in args.condition_on:
        header = " ".join(object_list) * 7
        # np.savetxt(predictions_file, predictions_matrix, delimiter=" ", fmt='%i', header=header, comments="")
        np.save(predictions_file.replace(".txt", ".npy"), predictions_matrix)
        test_input_path = os.path.join(tconf.ckpt_path, "test_inputs.txt")
        if not os.path.isfile(test_input_path):
            with open(test_input_path, "w") as f:
                f.writelines("\n".join(test_dataset.analysis_strings))
    else:
        header = " ".join(object_list)
        np.savetxt(predictions_file, predictions_matrix, delimiter=" ", fmt='%i', header=header, comments="")

    # save plot
    fig_file = os.path.join(tconf.ckpt_path, "predictions.png")
    trainer.flush_plot(fig_file)



if __name__ == "__main__":
    main()
    


import functools
import os
import gc
import json
import pdb
import random
import math
import sys
import pickle
from typing import Tuple, Optional, List, Dict, Any, Union, Iterable, Set, Literal
from pathlib import Path
import regex as re
import argparse
import operator
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import nnsight
from nnsight import LanguageModel, CONFIG
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from einops import rearrange, einsum
from datasets import Dataset
from torch.utils.data import DataLoader

from tqdm import tqdm
from jaxtyping import Float
from torch import Tensor
import plotly
import plotly.express as px
import matplotlib.pyplot as plt

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformer_lens import utils, HookedTransformer

from kneed import KneeLocator

# from dataset import INSTRUCTION, PROMPT, PROMPT_ALTFORM, PROMPT_ALLBOX_ALTFORM
# from utils import format_sentence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
NUM_BOXES=7
OG_OBJ_TO_NEW_OBJ = {
    "beer": "wine",
    "tissue": "pill",
    "cash": "coin",
    "bowl": "jar",
    "shoe": "magnet",
    "dish": "fork",
    "bone": "horn",
    "cheese": "stock",
    "tape": "photo",
    "knife": "pin",
    "jacket": "vest",
    "cake": "soup",
    "bottle": "container",
    "cream": "sugar",
    "cigarette": "pen",
    "shirt": "frame"
}

NON_OBJ_WORDS=[
    "put", "remove", "move",
    "contains", "the",
    # "container",
    "are", "from", "into", "and",  "is", "in", "box", "to",
    ",", ".",
    *[str(i) for i in range(10)]
]

MODEL_TO_SHORT={
    "codellama/CodeLlama-13b-hf": "codellama-13b",
    "google/gemma-2-2b": "gemma-2-2b",
    "meta-llama/Llama-2-13b-hf": "llama-2-13b",
    "meta-llama/Llama-3.2-3b": "llama-3.2-3b",
}

## TODO for now copied from entity-tracking-probing.src.dataset, but need to merge it once we refactor the repos
PROMPT = """Given the description after "Description:", write a true statement about a box and its contents according to the description after "Statement:".

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.
Statement: Box 3 contains the paper and the string.

Description: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map from Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """
PROMPT_ALTFORM = """Given the description after "Description:", write a true statement about a box and its contents according to the description after "Statement:". If a box is empty, write "Box X contains nothing".

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6.
Statement: Box 3 contains the paper and the string.

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map in Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 2 contains the bag and the machine and the map.

Description: """

PROMPT_ALLBOX_ALTFORM = """Given the description after "Description:", write a true statement about all boxes and their contents according to the description after "Statement:". If a box is empty, write "Box X contains nothing".

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6.
Statement: Box 0 contains the car, Box 1 contains the cross, Box 2 contains the bag and the machine, Box 3 contains the paper and the string, Box 4 contains the bill, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle and the map.

Description: The car is in Box 0, the cross is in Box 1, the bag and the machine are in Box 2, the paper and the string are in Box 3, the bill is in Box 4, the apple and the cash and the glass are in Box 5, the bottle and the map are in Box 6. Remove the car from Box 0. Remove the paper and the string from Box 3. Put the plane into Box 0. Move the map in Box 6 to Box 2. Remove the bill from Box 4. Put the coat into Box 3.
Statement: Box 0 contains the plane, Box 1 contains the cross, Box 2 contains the bag and the machine and the map, Box 3 contains the coat, Box 4 contains nothing, Box 5 contains the apple and the cash and the glass, Box 6 contains the bottle.

Description: """


INSTRUCTION = """Given the description after "Description:", write a true statement about a box and its contents according to the description after "Statement:". If a box is empty, write "Box X contains nothing".

Description: """

def free_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()

def _nn_apply(fn, *args, **kwargs):
    """nnsight<=0.5 exposed nnsight.apply(fn, ...) to call a python function inside a trace; nnsight>=0.6 executes
    the trace body on real tensors, so the function can be called directly."""
    if hasattr(nnsight, "apply"):
        return nnsight.apply(fn, *args, **kwargs)
    return fn(*args, **kwargs)


def setup_nnsight():
    """
    Setup script for nnsight
    """
    assert "NDIF_APIKEY" in os.environ, "pass NDIF_APIKEY environment variable!"
    CONFIG.API.APIKEY = os.environ['NDIF_APIKEY']
    assert "HF_TOKEN" in os.environ, "pass HF_TOKEN environment variable!"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)


def pad_batch_collate_fn(batch, tokenizer):
    """
    Args:
        batch (List[Dict[str, Any]]):
        tokenizer (Tokenizer):

    Returns:
        Dict[str, np.ndarray]: key-value pairs of data by field
    """
    new_batch = {}
    for k in batch[0].keys():
        if k in ["base_tokens", "source_tokens", "input_ids"]:
            batch_vals = force_pad(np.array([b[k] for b in batch], dtype=object), tokenizer)
            new_batch[k] = batch_vals

        elif k in ['base_last_token_indices', 'source_last_token_indices', "last_token_indices", "dataset_indices"]:
            new_batch[k] = np.array([b[k] for b in batch])

        elif k in ['source_label_types','source_labels', 'labels', "phrase_spans", "query_operation_phrase_spans", "query_description_phrase_spans"]:
            new_batch[k] = np.array([b[k] for b in batch], dtype=object)

    return new_batch

def load_dataloader(
    tokenizer: LlamaTokenizer,
    datafile: str,
    num_samples: int,
    num_boxes: int,
    ops_order: Optional[Tuple[str]]=None,
    query_ops_order: Optional[Tuple[str]]=None,
    min_numops: Optional[int] = None,
    min_query_numops: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    batch_size: int=1,
    return_dataset : bool=False,
    max_initial_objects_per_box: Optional[str] = None,
    counterfactual_format: str="rand_obj_rand_query_id",
    data_field: str="sentence",
    token_step: str="pred",
    prepend_space_to_answer: bool=False,
    model: Optional[Union[HookedTransformer,LanguageModel]]=None,
    success_filter: Optional[bool]=None,
    operations_on_same_obj: Optional[bool]=None,
    copy_filter: Optional[bool]=None,
    put_globally_removed_filter: Optional[bool]=None,
    num_query_object: Optional[int]=None,
    sort_query_objects: Optional[bool]=False,
    seed: int=42,
    object_data_file: str=Path(__file__).parents[1].resolve()/"data"/"objects"/"llama_friendly_objects.csv",
    prompt_format: Union[bool,str]=False,
    remote: bool=False,
):
    """
    Loads the data (original and counterfactual) from the datafile and creates a dataloader.

    Args:
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        num_boxes: number of boxes in the datafile.
        ops_order: sequence of operations
        query_ops_order: sequence of operations applied to query box
        min_numops: minimum number of operations
        min_query_numops: minimum number of query box operations
        max_seq_len: maximum number of tokens per example
        batch_size: batch size to use for the dataloader.
        return_dataset: whether to return dataset object in addition to dataloader
        max_initial_objects_per_box: maximum number of objects in initial states
        counterfactual_format: one of {rand_obj, rand_query_id, rand_box_id, rand_obj_rand_query_id, rand_obj_rand_box_id}
        data_field: which field in the data to sample from. Default is "sentence". If not sentence, the datapoint is wrapped with prompt.
        token_step: which token to do MI on. Default is "pred" (options exp_{x} where x-th reasoning step prediction).
        prepend_space_to_answer: whether to prepend space to the answer. llama1-7b doesn't need it but llama3.2 and gemma7b does.'
        model: model to use for filtering successful prompts only. Default is None.
        success_filter: whether to filter only successful prompts. Default is None.
        operations_on_same_obj: whether to only consider operations done on Same Object. Default is None.
        copy_filter: whether to filter out examples that can be solved with a simple copy mechanism (where the previous mention's first item is the same as label item). False is to remove those degenerate examples. None is no filter. True is to keep only those degenerate examples.
        put_globally_removed_filter: whether to filter out examples where query obj was previously removed from a non-query box
        num_query_object: how many objects in the query box. Default is None.
        sort_query_objects: whether to sort query objects by the order of appearance in the prompt.
        seed: random seed for sampling.
        object_data_file: path to object data file.
        prompt_format: whether to use prompt and if so what prompt format
        remote: whether to use remote NDIF machine.

    Returns:
        dataloader (and dataset object if specified)
    """
    fix_random_seed(seed)

    raw_data = load_pp_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        num_boxes=num_boxes,
        data_file=datafile,
        object_data_file=object_data_file,
        ops_order=ops_order,
        query_ops_order=query_ops_order,
        min_numops=min_numops,
        min_query_numops=min_query_numops,
        max_initial_objects_per_box=max_initial_objects_per_box,
        max_seq_len=max_seq_len,
        counterfactual_format=counterfactual_format,
        data_field=data_field,
        token_step=token_step,
        prepend_space_to_answer=prepend_space_to_answer,
        model=model,
        success_filter=success_filter,
        operations_on_same_obj=operations_on_same_obj,
        copy_filter=copy_filter,
        num_query_object=num_query_object,
        sort_query_objects=sort_query_objects,
        put_globally_removed_filter=put_globally_removed_filter,
        prompt_format=prompt_format,
        remote=remote,
    )

    dataset = Dataset.from_dict(raw_data).with_format("numpy")
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=partial(pad_batch_collate_fn, tokenizer=tokenizer))
    if return_dataset:
        return dataloader, dataset
    else:
        return dataloader


def operation_applies_to_query_box(sentence:str, op_idx: int) -> Tuple[bool, int]:
    """
    Check if operation is applied to the final box we query, also
    returns the rank of the query box (i.e. for Move operation, rank-0 would mean
    the object is removed from query box, rank-1 would mean object is added to the
    query box)

    Args:
        sentence (str): the sentence of entity tracking states + movements
        op_idx (int): location of the movement phrase (white space word index)

    Returns:
        whether operation applies to the query box, and the rank of the box in the movement phrase
    """
    words = sentence.split()
    query_box = sentence[sentence.rfind("Box"):].split()[1]
    affected_boxes = []
    i = op_idx
    while ("." not in words[i] or words[i] == ".") and i < len(words):
        if words[i] == "Box":
            affected_boxes.append(words[i+1].replace(".", ""))
        i += 1
    if query_box in affected_boxes:
        return True, affected_boxes.index(query_box)
    else:
        return False, -1


def get_ops_order(sentence: str, relevant: bool=False) -> Tuple[str]:
    ops = []
    for i, w in enumerate(sentence.split()):
        # need to think about whether the operation is applied to the box
        if w in {"Move", "Remove", "Put"}:
            movement_applies_to_query, movement_type = operation_applies_to_query_box(sentence, i)
            if (relevant and movement_applies_to_query) or not relevant:
                if w == "Move":
                    # move_0 means the object is removed from the box
                    # move_1 means the object is added to the box
                    ops.append(f"{w.lower()}_{movement_type}")
                else:
                    ops.append(w.lower())

    return tuple(ops)


def nothing_to_no_items(data: Dict[str, Any]) -> Dict[str, Any]:
    for field in ["sentence", "sentence_masked", "masked_content"]:
        data[field] = data[field].replace("nothing", "no items")
    return data


def add_the_to_end_of_prompt(data: Dict[str, Any]) -> Dict[str, Any]:
    data["sentence_masked"] += " the"
    data["masked_content"] = data["masked_content"].replace("the ", "", 1)
    return data


def permute_lists(*lists):
    assert np.var([len(l) for l in lists]) == 0
    order = np.random.permutation(range(len(lists[0])))
    new_list = []
    for i in range(len(lists)):
        new_list.append([lists[i][j] for j in order])
    return new_list


def replace_string_with_obj_map(s: str) -> str:
    if s in OG_OBJ_TO_NEW_OBJ:
        return OG_OBJ_TO_NEW_OBJ[s]

    for old, new in OG_OBJ_TO_NEW_OBJ.items():
        s = s.replace(old, new)
    return s


def replace_data_objects_with_map(data: Dict) -> Dict:
    new_data = {}
    for k, v in data.items():
        if isinstance(v, str):
            new_data[k] = replace_string_with_obj_map(v)
        elif isinstance(v, list):
            if len(v) == 0:
                new_data[k] = []
            else:
                assert isinstance(v[0], str)
                new_data[k] = [replace_string_with_obj_map(s) for s in v]
        elif isinstance(v, dict):
            new_data[k] = replace_data_objects_with_map(v)
    return new_data


def operations_match(desired_ops: Tuple[str], actual_ops: Tuple[str]) -> bool:
    """
    To check if specified operation match with actual operations
    mainly to deal with move. If one specifies "move", it should match with
    either "move_0" (equivalent to "remove") or "move_1" (equivalent to "put").
    But if user specifies "move_0", then it should not match with "move_1".
    """

    if desired_ops == actual_ops:
        return True

    if "move" not in desired_ops:
        return False

    if len(desired_ops) != len(actual_ops):
        return False

    for desired_op, actual_op in zip(desired_ops, actual_ops):
        if (desired_op == "move" and actual_op in {"move_0", "move_1"}) or desired_op==actual_op:
            continue
        else:
            return False

    return True


def get_full_word_after_index(full_str:str, start:int) -> str:
    end = start
    while full_str[end] not in [" ", "\n", ".", ","]:
        if end == len(full_str)-1:
            return full_str[start:]
        end += 1
    return full_str[start:end]

PROMPT_SUCCESS_CACHE={}
def check_prompt_success(
    model: Union[HookedTransformer, LanguageModel, AutoModelForCausalLM],
    prompt: str,
    label_tokens: torch.Tensor,
    tokenizer: AutoTokenizer,
    remote: bool=False,
    cache_path: Optional[str]=None,
) -> bool:
    global PROMPT_SUCCESS_CACHE
    # caching prompt success with remote execution because these onse are really slow
    # so saving some time
    if remote and cache_path is None:
        cache_path = Path(__file__).resolve().parent / "cache" / f"{model.name_or_path.split('/')[-1]}.pkl"
        os.makedirs(cache_path.parent, exist_ok=True)

    if remote and cache_path is not None:
        if not cache_path.is_file():
            PROMPT_SUCCESS_CACHE={}
        else:
            if not PROMPT_SUCCESS_CACHE:
                with open(cache_path, "rb") as f:
                    PROMPT_SUCCESS_CACHE = pickle.load(f)
            if prompt in PROMPT_SUCCESS_CACHE:
                return PROMPT_SUCCESS_CACHE[prompt]
            
    # pdb.set_trace(header="checking prompt success")
    if isinstance(model, HookedTransformer):
        tokens = model.to_tokens(prompt)
        logits = model(tokens)
    elif isinstance(model, LanguageModel):
        with model.trace(prompt, remote=remote):
            logits = model.lm_head.output.save()
    else:  # huggingface model
        tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        logits = model(tokens)["logits"]
    argmax_token = logits[0,-1].argmax(dim=-1)
    prompt_success = argmax_token.item() in label_tokens
    if prompt_success == False:
        # print prompt and response
        decoded_response = tokenizer.decode(argmax_token)
        print(f"Prompt failed:\n{prompt}\nModel response: {decoded_response}\nExpected one of: {[tokenizer.decode(t.item()) for t in label_tokens]}")

    if remote and cache_path is not None:
        PROMPT_SUCCESS_CACHE[prompt] = prompt_success
        with open(cache_path, "wb") as f:
            pickle.dump(PROMPT_SUCCESS_CACHE, f)

    return prompt_success

def remove_substrings(text, substrings):
    for sub in substrings:
        if sub.isnumeric() or sub in [",", ".", ":", "[", "]"]:
            text = text.replace(sub, "")
        else:
            text = re.sub(r'(^|\s+)'+sub+r'($|\s+)', ' ', text)
    return text.replace("  ", " ")


def if_operations_on_same_obj(sentence_masked: str, operations_on_same_obj: bool) -> bool:
    """ Check if operations are all applying to the same object """
    objs = remove_substrings(sentence_masked[sentence_masked.find(".")+1:].lower(), NON_OBJ_WORDS).split()
    if operations_on_same_obj:
        return len(set(objs)) == 1
    else:
        return len(set(objs)) == len(objs)


def is_put_globally_removed(data: Dict, filter_flag)-> bool:
    label_objs = data["masked_content"].split(" and the ")
    query_id = data["sentence"].split()[-4]
    op_phrases = data["sentence"].strip(".").split(".")[:-1]
    is_put_globally_removed = False
    for obj in label_objs:
        for op_phrase in op_phrases:
            if ("Remove" in op_phrase) and (f" {obj} " in op_phrase) and (f"Box {query_id}" not in op_phrase):
                is_put_globally_removed = True
                break
            # elif ("Move" in op_phrase) and (obj in op_phrase) and (f"Box {query_id}" not in op_phrase):
            #     is_put_globally_removed = True
            #     break
    return is_put_globally_removed == filter_flag


def compute_phrase_spans(sentence:str, tokenizer: AutoTokenizer) -> List[Tuple[int,int]]:
    """
    Given sentence and tokenizer, compute the spans (start, end) of token indices
    for each phrase. A phrase could be an initial description ("box A contains
    Apple") or an operation phrase ("Remove Apple from box A").

    Args:
        sentence (str): Sentence to compute spans for
        tokenizer (AutoTokenizer): Tokenizer used during experiment.

    Returns:
        List[Tuple[int,int]]: List of tuples containing start and end indices of each phrase
    """

    sentence_tokens = tokenizer.encode(sentence)
    spans = []
    # initial description phrases
    init_desc = sentence.split(". ")[0] + "."
    init_desc_tokens = tokenizer.encode(init_desc)
    # codellama for example use different token for ',' and 'key,'.
    period_token = tokenizer.encode("key.", add_special_tokens=False)[-1]
    comma_token = tokenizer.encode("key,", add_special_tokens=False)[-1]
    assert comma_token in init_desc_tokens, "something funky with tokenization"

    start = 0
    for i, token in enumerate(init_desc_tokens):
        if token == comma_token:
            end = i
            spans.append([start, end])
            start = i + 1
    spans.append([start, len(init_desc_tokens) - 1])

    # operation phrase spans
    op_phrases = ". ".join(sentence.strip(".").split(". ")[1:-1])
    op_phrases_tokens = tokenizer.encode(op_phrases)
    start = len(init_desc_tokens)
    for i, token in enumerate(op_phrases_tokens):
        if token == period_token:
            end = len(init_desc_tokens) + i - 1
            spans.append([start, end])
            start = len(init_desc_tokens) + i
    spans.append([start, len(init_desc_tokens) + len(op_phrases_tokens) - 1])

    # query phrase span
    spans.append([len(init_desc_tokens) + len(op_phrases_tokens), len(sentence_tokens)-1])
    return spans


def compute_phrase_span_tokens(sentence: str, phrase_spans: List[Tuple[int, int]], tokenizer: AutoTokenizer) -> List[str]:
    """
    Given sentence, phrase span, return list of phrase tokens (good for visualization labels)

    Args:
        sentence (str): sentence to compute phrase spans names
        phrase_spans (List[Tuple[int, int]]): list of (start, end) tuples of phrase spans to name for

    Returns:
        List[str]: list of phrase names
    """
    tokens = []
    s = tokenizer.encode(sentence)
    for i in range(NUM_BOXES):
        tokens.append(f"DESC_{i}: {tokenizer.decode(s[phrase_spans[i][0]:phrase_spans[i][1]+1], skip_special_tokens=True)}")
    for i in range(sentence.strip(".").count(". ")-1):
        tokens.append(f"OP_{i}: {tokenizer.decode(s[phrase_spans[i+NUM_BOXES][0]:phrase_spans[i+NUM_BOXES][1]+1], skip_special_tokens=True)}")
    tokens.append(f"QUERY: {tokenizer.decode(s[phrase_spans[-1][0]:phrase_spans[-1][1]+1], skip_special_tokens=True)}")
    return tokens


def compute_query_phrase_spans(sentence:str, phrase_spans: List[Tuple[int, int]]) -> Tuple[Tuple[int, int],List[Tuple[int, int]]]:
    """
    Given a sentence and phrase_spans, return a subset of the spans that corresponds to phrases that mentions query box

    Args:
        sentence (str): sentence to compute spans for
        phrase_spans (List[Tuple[int, int]]): list of (start, end) token indices for phrase relevant to query box

    Returns:
        List of (start, end) token indices for phrases that mention query box
    """
    op_phrases = sentence.strip(".").split(". ")[1:-1]
    query_box = sentence[sentence.rfind("Box") + 4]
    query_operation_spans = []
    # get operation phrase spans
    for i, op_phrase in enumerate(op_phrases):
        if f"Box {query_box}" in op_phrase:
            query_phrase_idx = NUM_BOXES + i
            query_operation_spans.append(phrase_spans[query_phrase_idx])

    # get initial description phrase spans
    query_description_span = phrase_spans[int(query_box)]
    return query_description_span, query_operation_spans


def get_phrase_around_idx(s: str, idx: int) -> str:
    delimiters = ["[", "]", "\n", ".", ","]
    start = max([s.rfind(delim,0, idx)+1 for delim in delimiters])
    end = min([s.find(delim, max(idx, start))+1 for delim in delimiters if s.find(delim, max(idx, start))!=-1])
    return s[start:end]

def is_copyable(prompt: str, label_items_only:str, data: Dict[str, Any], copy_filter: bool) -> bool:
    """
    trace back to the phrase that does not contain any operations (either original world state description, or previous CoT explanation)
    and check whether that phrase's first object is the same as the first object in the label to be predicted
    if it is, then a model can simply solve it with a copy mechanism.
    if copy_filter=False, we remove the easy copy-able data
    """
    prompt = prompt[:prompt.rfind(".")+1]
    query_box = data["sentence"][data["sentence"].rfind("contains") - 3:data["sentence"].rfind("contains")].strip()
    correct_phrase_found = False
    while not correct_phrase_found:
        prev_mention_idx = prompt.rfind(f"Box {query_box}")
        prev_mention_phrase = get_phrase_around_idx(prompt, prev_mention_idx)
        prompt = prompt[:prev_mention_idx]
        correct_phrase_found = not any([op in prev_mention_phrase for op in ["Move", "Put", "Remove"]])
    prev_mention_objs = remove_substrings(prev_mention_phrase.lower(), NON_OBJ_WORDS).split()
    copyable = label_items_only.split()[0] == prev_mention_objs[0]
    return copyable == copy_filter


def sample_box_data(
    tokenizer,
    num_samples,
    data_file,
    object_data_file: str="",
    desired_ops_order: Optional[Tuple[str]]=None,
    desired_query_ops_order: Optional[Tuple[str]]=None,
    min_numops: Optional[int]=None,
    min_query_numops: Optional[int]=None,
    max_initial_objects_per_box: Optional[int]=None,
    max_seq_len: Optional[int]=None,
    counterfactual_format: str="rand_obj_rand_query_id",
    data_field: str="sentence",
    token_step: str="pred",
    prepend_space_to_answer: bool=False,
    model: Optional[Union[HookedTransformer,LanguageModel]]=None,
    success_filter: Optional[bool]=None,
    operations_on_same_obj: Optional[bool]=None,
    copy_filter: Optional[bool]=None,
    put_globally_removed_filter: Optional[bool]=None,
    num_query_object: Optional[int]=None,
    sort_query_objects: Optional[bool]=False,
    prompt_format: Union[str, bool]=False,
    remote: bool=False
):
    """
    Sample data from the box data file

    Args:
        tokenizer: Tokenizer to be used
        num_samples: Number of samples to be generated
        data_file: Path to the box data file
        object_data_file: Path to the object data file
        desired_ops_order: Sequence of operations we want data sampled to have
        desired_query_ops_order: Sequence of operation appliedto the query box
        min_numops: minimum number of operations
        min_query_numops: minimum number of query box operations
        max_initial_objects_per_box: maximum objects per box for initial states
        max_seq_len: maximum number of tokens
        counterfactual_format: one of {rand_obj, rand_query_id, rand_box_id, rand_obj_rand_query_id, rand_obj_rand_box_id}
        data_field: which field in the data to sample from. Default is "sentence". If not sentence, the datapoint is wrapped with prompt.
        token_step: which step to do MI on. {pred, exp_<x>}
        prepend_space_to_answer: whether to prepend space to the answer. llama1-7b doesn't need it but llama3.2 and gemma7b does.
        model: model to be used for filtering success examples.
        success_filter: whether to only load success examples.
        operations_on_same_obj: whether to only apply operations on same object.
        copy_filter: whether to filter out examples that can be solved with a simple copy mechanism (where the previous mention's first item is the same as label item). False is to remove those degenerate examples. None is no filter. True is to keep only those degenerate examples.
        put_globally_removed_filter: whether to filter in/out examples that added an object previously removed from non-query box.
        num_query_object: number of query objects to sample from.
        sort_query_objects: whether to sort query objects by appearance order.
        prompt_format: whether to add prompt structure, default is just completion
        remote: whether to use NDIF remote execution
    """

    with open(data_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # read in list of valid objects
    if os.path.exists(object_data_file):
        all_objects = pd.read_csv(object_data_file).object_name.unique()
    else:
        all_objects = []
    token_prefix = " " if prepend_space_to_answer else ""

    assert num_samples <= len(data)
    prompts, labels, output_data, dataset_indices = [], [], [], []
    ctf_field = f"{'' if data_field == 'sentence' else data_field + '_'}counterfactual_{counterfactual_format}"
    bar = tqdm(total=num_samples)
    for i in range(len(data)):
        bar.set_description(f"Processing {i}/{len(data)}:")
        # extra processing to replace "contains nothing" -> "contains no items"
        data[i] = nothing_to_no_items(data[i])

        # add "the" to end of prompt sentence
        data[i] = add_the_to_end_of_prompt(data[i])
        # issue here is "no items" means we can't prompt model with "box n contains the"
        if "no items" in data[i]["masked_content"]:
            continue

        if any(p is not None for p in [desired_ops_order, desired_query_ops_order, min_numops, min_query_numops]):
            if "ops_order" not in data[i]:
                data[i]["ops_order"] = get_ops_order(data[i]["sentence"], relevant=False)
                data[i]["query_ops_order"] = get_ops_order(data[i]["sentence"], relevant=True)

        if desired_ops_order is not None:
            if not operations_match(desired_ops_order, data[i]["ops_order"]):
                continue

        if desired_query_ops_order is not None:
            # if we only care about only applied order matching, then this means 
            # noised input may have way more operations (but applied to other boxes)
            # unless we ennforce both ops_order and applied_ops_order to be the same
            if not operations_match(desired_query_ops_order, data[i]["query_ops_order"]):
                continue

        if min_numops is not None and len(data[i]["ops_order"]) < min_numops:
            continue

        if min_query_numops is not None and len(data[i]["query_ops_order"]) < min_query_numops:
            continue

        if operations_on_same_obj is not None:
            # if we want all operations to be done on the same object (i.e. for move_0=remove,put data)
            if not if_operations_on_same_obj(data[i]["sentence_masked"], operations_on_same_obj):
                continue

        # if number of objects in initial states are too much, remove
        if max_initial_objects_per_box is not None:
            pdb.set_trace()
            if any(len(objs) > max_initial_objects_per_box for objs in data[i]["initial_state"].values()):
                continue

        # if a previously removed object (from other boxes) is added to query box filter
        if put_globally_removed_filter is not None:
            if not is_put_globally_removed(data[i], put_globally_removed_filter):
                continue

        # if any objects in used objects contains more than 1 sub-word, then remove (keeping sequence same length)
        if "used_objects" in data[i] and any(len(tokenizer.encode(obj, add_special_tokens=False))>1 for obj in data[i]["used_objects"]):
            continue

        # if token_step != "pred", then we have to modify data to trace circuit for the right step
        if token_step != "pred":
            assert data_field != "sentence", "analyzing exp steps require using prompts not original sentences"
            step_idx = int(token_step.split("_")[-1])
            step_annotation = data[i]["annotations"]["explanation"][data_field][step_idx]
            full_prompt = data[i][data_field]
            label_items_only = " ".join([get_full_word_after_index(full_prompt, obj_idx) for obj_idx in step_annotation["objects"]])
            label_items_only = 'nothing' if len(label_items_only)==0 else label_items_only
            prompt = full_prompt[:full_prompt.find(label_items_only.split()[0], step_annotation["box"])].strip()
            # also have to cut the counterfactual sentences at the right place, here since we designed
            # counterfactuals to have same number of tokens as original sentence, it is easy
            prompt_token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            data[i][ctf_field] = tokenizer.decode(tokenizer.encode(data[i][ctf_field], add_special_tokens=False)[:prompt_token_len])
        else:
            # if there are multiple objects, only keep the actual object words, because at patching time, we sum up
            # the label token's probs if there are multiple tokens
            label_text = data[i]["masked_content"].replace(".", "").strip()
            label_items_only = label_text.replace("and ", "").replace("the ", "").replace("  ", " ")
            if data_field == "sentence":
                prompt = data[i]["sentence_masked"]
            else:
                prompt = data[i][data_field].removesuffix(".").removesuffix(label_text).strip()

        # remove data where model can perform the task with just copying
        if copy_filter is not None:
            if not is_copyable(prompt, label_items_only, data[i], copy_filter):
                continue

        # takes only the first token (as we just need to know the model is on the right track)
        label_tokens = torch.tensor([tokenizer.encode(token_prefix + obj, add_special_tokens=False)[0] for obj in label_items_only.split(" ")])

        # optionally remove long sequences
        if max_seq_len is not None and len(tokenizer.encode(prompt)) > max_seq_len:
            continue

        if num_query_object is not None and len(label_tokens) != num_query_object:
            continue

        if sort_query_objects:
            # sort query object based on the order in which they appear in prompt
            label_tokens = sorted(label_tokens, key=lambda x: torch.nonzero(tokenizer.encode(prompt, return_tensors="pt").squeeze()==x).min(), reverse=False)

        # Optionally filter for successful examples only
        if success_filter is not None:
            if model is None:
                raise Exception("model is None, but success_filter is True")
            try:
                if prompt_format:
                    formated_prompt = format_sentence(prompt, prompt_format=prompt_format, prompt_prefix=globals().get(prompt_format), tokenizer=tokenizer)
                else:
                    formated_prompt = prompt
                prompt_success = check_prompt_success(model, formated_prompt, label_tokens, tokenizer, remote=remote)
            except Exception as e:
                print(f"Exception while checking prompt success, likely OOM, skipping: {e}")
                pdb.set_trace()
                continue
            if prompt_success != success_filter:
                continue


        labels.append(label_tokens)
        prompts.append(prompt)
        output_data.append(data[i])
        dataset_indices.append(i)

        # at last compute phrase spans, phrase span tokens, and query phrase spans
        data[i]["phrase_spans"] = compute_phrase_spans(data[i]["sentence"], tokenizer)
        data[i]["phrase_span_tokens"] = compute_phrase_span_tokens(data[i]["sentence"], data[i]["phrase_spans"], tokenizer)
        data[i]["query_description_phrase_spans"], data[i]["query_operation_phrase_spans"] = (
            compute_query_phrase_spans(data[i]["sentence"], data[i]["phrase_spans"]))

        bar.update(1)
        if len(prompts) >= num_samples:
            break

    if len(labels) < num_samples:
        raise Exception(f"Not enough samples ({len(labels)}/{num_samples}) with operation sequence {desired_ops_order}")

    # assert np.var([len(p.split()) for p in prompts]) == 0, "number of white space tokens differ across prompts!"
    input_tokens = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_indices = input_tokens["attention_mask"].sum(dim=1) - 1
    output_ids = labels  # torch.tensor(labels), could be different length due to multi-objects
    # input_ids = input_tokens["input_ids"]
    input_ids = list(input_tokens["input_ids"])

    # tokenize counterfactuals here so padding is the same as input
    if any(s in ctf_field for s in ["dcm", "cma", "add_single_op"]):
        # Generate these counterfactuals on the spot
        for i, prompt in enumerate(prompts):
            ctf_output = {}
            if "dcm_remove" in ctf_field:
                dcm_type = ctf_field[ctf_field.find("dcm_"):]
                dcm_args = dcm_type.split("_")
                # break_up_description = dcm_args[-1] if dcm_args[-1] in ["no", "break"] else "no"
                pos_options = ["all", "box", "obj", "period", "phrase"]
                patch_pos = dcm_args[-1] if dcm_args[-1] in pos_options else dcm_args[-2] if dcm_args[-2] in pos_options else "both"
                if "alt" in dcm_type:
                    ctf_output = generate_counterfactual_dcm_remove_second_oid_alt(prompt, tokenizer=tokenizer, pos=patch_pos)
                elif "across" in dcm_type:
                    ctf_output = generate_counterfactual_dcm_remove_across(prompt, tokenizer=tokenizer)
                else:
                    ctf_output = generate_counterfactual_dcm_remove_second_oid(prompt, tokenizer=tokenizer, pos=patch_pos)
            elif "dcm" in ctf_field:
                ctf_output = generate_counterfactual_dcm_obj_or_pos(prompt, all_objects=all_objects, hypothesis=ctf_field[ctf_field.find("dcm_") + 4:])
                output_data[i]["ctf_label_types"] = ctf_output["label_types"]
            elif "add_single_op" in ctf_field:
                arg_strs = ctf_field[ctf_field.find("single_op_") + 10:].split("_")
                op, operat_box_type = arg_strs[0], arg_strs[1]
                label_type = arg_strs[2] if len(arg_strs)>=3 else "op"
                swap_order = arg_strs[3] if len(arg_strs)>=4 else False
                ctf_output = generate_counterfactual_add_single_op(prompt, all_objects, op=op, is_query=operat_box_type,label=label_type, swap_description_order=swap_order)
            elif "cma_remove" in ctf_field:
                cma_type = ctf_field[ctf_field.find("cma_"):]
                ctf_output = globals()[f"generate_counterfactual_{cma_type}"](prompt, all_objects=all_objects)
            elif "cma" in ctf_field:
                cma_type = ctf_field[ctf_field.find("cma_"):]
                ctf_output = globals()[f"generate_counterfactual_{cma_type}"](prompt)
            else:
                raise NotImplementedError(f"Unknown ctf field {ctf_field}")

            output_data[i][ctf_field] = ctf_output["new_prompt"]
            if ctf_output.get("labels") is not None:
                ctf_label_tokens = [tokenizer.encode(f"{token_prefix}{l}", add_special_tokens=False)[0] for l in ctf_output["labels"]]
                output_data[i]["ctf_output_ids"] = ctf_label_tokens

            if ctf_output.get("changed_input_prompt") is not None:
                # new_input_ids = torch.Tensor(tokenizer.encode(ctf_output["changed_input_prompt"]))
                # input_ids[i][:len(new_input_ids)] = new_input_ids
                input_ids[i] = torch.tensor(tokenizer.encode(ctf_output["changed_input_prompt"]))#.to(torch.int16)

            if ctf_output.get("changed_input_labels") is not None: # assumes a list of strings
                output_ids[i] = torch.tensor([tokenizer.encode(token_prefix + obj, add_special_tokens=False)[0] for obj in ctf_output["changed_input_labels"]])

            if ctf_output.get("patch_locations") is not None:
                output_data[i]["patch_locations"] = ctf_output["patch_locations"]

    cf_sentences = [output_data[i][ctf_field] for i in range(len(prompts))]
    if token_step == "pred":  # TODO this might be done the same way as not pred case where we just truncate ctf at the same token len as sentence
        cf_sentences = [s[:s.rfind("contains the") + 12].strip() for s in cf_sentences]
    cf_tokens = tokenizer(cf_sentences, padding=True, return_tensors="pt")
    for i in range(len(output_data)):
        output_data[i]["ctf_input_ids"] = cf_tokens["input_ids"][i]
        output_data[i]["ctf_attention_mask"] = cf_tokens["attention_mask"].sum(dim=-1)[i] - 1

    input_ids, last_token_indices, output_ids, output_data = permute_lists(input_ids, last_token_indices, output_ids, output_data)
    return input_ids, last_token_indices, output_ids, output_data, dataset_indices


def load_pp_data(
    tokenizer:AutoTokenizer,
    num_samples:int,
    num_boxes:int,
    ops_order: Optional[Tuple[str]]=None,
    query_ops_order: Optional[Tuple[str]]=None,
    min_numops: Optional[int] = None,
    min_query_numops: Optional[int] = None,
    data_file: str="",
    object_data_file: str="",
    max_initial_objects_per_box: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    counterfactual_format: str = "rand_obj_rand_query_id",
    data_field: str="sentence",
    token_step: str="pred",
    prepend_space_to_answer: bool=False,
    model: Optional[Union[HookedTransformer,LanguageModel]]=None,
    success_filter: Optional[bool]=None,
    operations_on_same_obj: Optional[bool]=None,
    copy_filter: Optional[bool]=None,
    put_globally_removed_filter: Optional[bool]=None,
    num_query_object: Optional[int]=None,
    sort_query_objects: Optional[bool]=False,
    prompt_format: Union[bool, str]=False,
    remote: bool=False,
)  -> Dict[str, torch.Tensor]:
    """
    Load data for path patching task consisting of original and counterfactual
    examples (random label and random object).

    args:
        ops_order(Optional[Tuple[str]]): operation order for data filter
        query_ops_order(Optional[Tuple[str]]): operation order applied to query box for data filter
        min_numops(Optional[int]): minimum number of operations
        min_query_numops(Optional[int]): minimum number of query operations
        max_initial_objects_per_box (int): filter out data where there are too many initial objects.
            good for keeping sequence same length
        max_seq_len (Optional[int]): maximum number of tokens
        counterfactual_format (str): one of {rand_obj, rand_query_id, rand_box_id, rand_obj_rand_query_id, rand_obj_rand_box_id}
        data_field (str): which field in the data to sample from. Default is "sentence". If not sentence, the datapoint is wrapped with prompt.
        token_step (str): which step to do MI on. {pred, exp_<x>}
        prepend_space_to_answer (bool): whether to prepend space to the answer. llama1-7b doesn't need it but llama3.2 and gemma7b does.'
        model (Optional[HookedTransformer]): model to be used for filtering success examples.
        success_filter (Optional[bool]): whether to only load success/unsuccessful examples.
        operations_on_same_obj (Optional[bool]): whether to only use objects in same object.
        copy_filter (Optional[bool]): whether to filter out examples that can be solved with a simple copy mechanism (where the previous mention's first item is the same as label item). False is to remove those degenerate examples. None is no filter. True is to keep only those degenerate examples.
        put_globally_removed_filter (Optional[bool]): whether to filter in/out examples where query obj was previously removed from non-query box
        num_query_object (Optional[int]): number of query objects (in label) to filter for.
        sort_query_objects (Optional[bool]): whether to sort query objects by appearance order.
        prompt_format (Union[bool, str]): whether to use prompt and if so what prompt format. Defaults to False.
        remote (bool): whether to use NDIF remote sever
    """
    input_ids,last_token_indices,output_ids,output_data,dataset_indices = sample_box_data(
        tokenizer=tokenizer,
        num_samples=num_samples,
        data_file=data_file,
        object_data_file=object_data_file,
        desired_ops_order=ops_order,
        desired_query_ops_order=query_ops_order,
        min_numops=min_numops,
        min_query_numops=min_query_numops,
        max_initial_objects_per_box=max_initial_objects_per_box,
        max_seq_len=max_seq_len,
        counterfactual_format=counterfactual_format,
        data_field=data_field,
        token_step=token_step,
        prepend_space_to_answer=prepend_space_to_answer,
        model=model,
        success_filter=success_filter,
        operations_on_same_obj=operations_on_same_obj,
        copy_filter=copy_filter,
        put_globally_removed_filter=put_globally_removed_filter,
        num_query_object=num_query_object,
        sort_query_objects=sort_query_objects,
        prompt_format=prompt_format,  # inputted only when we need to verify prompt success
        remote=remote,
    )

    all_base_input_ids = []         # Clean inputs
    all_base_input_last_pos = []    # Clean last token indices
    all_source_input_ids = []       # Corrupt inputs
    all_source_input_last_pos = []  # Corrupt last token indices
    all_base_output_ids = []        # Correct answer token
    all_source_output_ids = []      # Corrupt answer token
    all_source_label_types = []     # Corrupt answer token type (which operation is the object from)
    all_phrase_spans = []           # Spans of all phrases
    all_query_operation_phrase_spans = []       # Span of description phrase related to query box (box a contains apple)
    all_query_description_phrase_spans = []     # Span of operation phrases related to query box (put apple into box a)
    all_patch_locations = []        # Tuples of source target patch locations
    all_dataset_indices = []        # Index of the datapoints in the original dataset file

    for i in range(0, num_samples):
        # randomly pick a initial state, and query box
        all_base_input_ids += [input_ids[i]]
        all_base_input_last_pos += [last_token_indices[i]]
        all_base_output_ids += [output_ids[i]]
        all_dataset_indices += [dataset_indices[i]]

        ctf_field = f"{'' if data_field=='sentence' else data_field+'_'}counterfactual_{counterfactual_format}"
        if ctf_field not in output_data[i]:  # old behavior
            # randomly sample from another source that has the same order of ops
            # just to guarantee it is not from the same group of query with same box states
            # same box state meaning not any boxes contain the same objects
            random_source_index = random.randint(0, len(input_ids)-1)
            while any(output_data[i]["initial_state"][k] == output_data[random_source_index]["initial_state"][k] for k in output_data[i]["initial_state"].keys()):
                random_source_index = random.randint(0, len(input_ids)-1)
            source_example = input_ids[random_source_index].clone()

            # Change the query box label with a random number
            random_box_label = str(random.randint(0, 9))
            random_box_label_token = tokenizer(
                f" {random_box_label}", return_tensors="pt", add_special_tokens=False
            ).input_ids[0, -1] # TODO: verify if by not adding special tokens, the label will be the first index
            source_example[-3] = random_box_label_token

            all_source_input_ids += [source_example]
            all_source_input_last_pos += [last_token_indices[random_source_index]]
        else:
            # new version of box data generation include counterfactual generation
            all_source_input_ids += [output_data[i]["ctf_input_ids"]]
            all_source_input_last_pos += output_data[i]["ctf_attention_mask"].unsqueeze(-1)
            all_source_output_ids += [output_data[i].get("ctf_output_ids")]
            all_source_label_types += [output_data[i].get("ctf_label_types")]
            all_patch_locations += [output_data[i].get("patch_locations")]

        # add query phrase span
        all_phrase_spans += [output_data[i]["phrase_spans"]]
        all_query_operation_phrase_spans += [output_data[i]["query_operation_phrase_spans"]]
        all_query_description_phrase_spans += [output_data[i]["query_description_phrase_spans"]]

        # add/format prompt
        if prompt_format:
            # change input ids

            all_base_input_ids[-1] = tokenizer.encode(format_sentence(all_base_input_ids[-1].tolist(), prompt_format=prompt_format, prompt_prefix=globals().get(prompt_format), tokenizer=tokenizer))
            all_base_input_last_pos[-1] = len(all_base_input_ids[-1])-1
            all_source_input_ids[-1] = tokenizer.encode(format_sentence(all_source_input_ids[-1].tolist(), prompt_format=prompt_format, prompt_prefix=globals().get(prompt_format), tokenizer=tokenizer))
            all_source_input_last_pos[-1] = len(all_source_input_ids[-1])-1
            # phrase span/ patch locations (just add # tokens for prefix)
            prefix_offset = len(tokenizer.encode(globals().get(prompt_format).strip()))  # strip because we don't want trailing space
            all_phrase_spans[-1] = [[s[0]+prefix_offset, s[1]+prefix_offset] for s in all_phrase_spans[-1]]
            all_query_operation_phrase_spans[-1] = [[s[0]+prefix_offset, s[1]+prefix_offset] for s in all_query_operation_phrase_spans[-1]]
            all_query_description_phrase_spans[-1] = [all_query_description_phrase_spans[-1][0]+prefix_offset, all_query_description_phrase_spans[-1][1]+prefix_offset]
            if all_patch_locations[-1] is not None:
                pdb.set_trace()
                prefix_context_offset = all_base_input_last_pos[-1] - 5
                all_patch_locations[-1] = [loc + prefix_context_offset if all_base_input_last_pos[-1]-loc < 5 else loc + prefix_offset for loc in all_patch_locations[-1]]
                # TODO if patch location in query phrase, need to offset by even more
                assert NotImplementedError("patch location not none, need to be adjusted with few-shot prompts")
    return {
        "base_tokens": all_base_input_ids,
        "base_last_token_indices": all_base_input_last_pos,
        "source_tokens": all_source_input_ids,
        "source_last_token_indices": all_source_input_last_pos,
        "labels": all_base_output_ids,
        "source_labels": all_source_output_ids,
        "source_label_types": all_source_label_types,
        "phrase_spans": all_phrase_spans,
        "query_operation_phrase_spans": all_query_operation_phrase_spans,
        "query_description_phrase_spans": all_query_description_phrase_spans,
        "patch_locations": all_patch_locations,
        "dataset_indices": all_dataset_indices,
    }


# Helper function (plot)
def imshow(tensor, renderer=None, save_path=None, **kwargs):
    update_layout_set = {
        "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
        "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
        "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
    }

    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)
    if save_path is not None:
        plotly.offline.plot(fig, filename=save_path)
    else:
        fig.show(renderer=renderer)


def probability_correct_box(
    logits: Float[Tensor, "batch seq d_vocab"],
    last_token_name: str,
    dataset: dict,
) -> Float[Tensor, "batch"]:

    # Probability of the correct token predicted by the original run
    p = logits[range(logits.size(0)), dataset[last_token_name], :]
    with torch.no_grad():
        p = torch.softmax(p, axis=1)

    if len(dataset["labels"].shape) > 1:  # (i.e. query box contains multiple objs, shape would be [batch, #objs])
        p = p[range(logits.size(0)), dataset["labels"]].sum(1)
    else:
        p = p[range(logits.size(0)), dataset["labels"]]
    return p
        

def _entity_tracking_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    p_org: Float[Tensor, "batch"],
    dataset: dict,
) -> float:
    p_patch = probability_correct_box(logits, "source_last_token_indices", dataset)

    return ((p_patch - p_org) / p_org).mean().item()


def force_pad(tokens: np.array, tokenizer) -> torch.Tensor:
    """
    Takes in a np.Column object (list of potentially different len arrays), return a padded batch indices
    """
    prompts = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="right")["input_ids"]
    return input_ids


def get_multi_object_phrase_spans(input_ids: torch.Tensor, tokenizer: AutoTokenizer) -> List[List[int]]:
    """
    given a batch of input ids, return list of object phrase spans. 
    i.e. "the cake and the book are in ..." will be averaged to "the [phrase_span] are in ..."
    Args:
        input_ids (torch.tensor): dim [bs, seq, ...]
        tokenizer (AutoTokenizer):
    """
    and_token_id = tokenizer.encode(" and")[-1]
    batch_spans = []
    for i in range(len(input_ids)):
        and_pos_list = torch.nonzero(input_ids[i] == and_token_id).squeeze(-1)
        # for every datapoint we calculate span serially
        spans = []
        curr_span = None
        for and_pos in and_pos_list:
            # Gemma-specific, but pretty sure it generalizes to other tokenizers
            if curr_span is None:
                curr_span = [and_pos-1, and_pos+2]
            else:
                # if current token is part of previous span, add to it
                if and_pos-2 < curr_span[1]:
                    curr_span[1] = and_pos+2
                else:  # otherwise we append previous span and start a new one
                    spans.append(curr_span)
                    curr_span = [and_pos-1, and_pos+2]
        if curr_span is not None:
            spans.append(curr_span)

        batch_spans.append(spans)
    return batch_spans


def normalize_object_phrase(batch_t: torch.Tensor, batch_input_ids: torch.Tensor, tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    If there are multiple objects in a phrase, aggregate them into single token
    Args:
        batch_t: input tensor of shape [bs, seq, d_model]
        input_ids: token id of batched input sequence [bs, seq]
        tokenizer: AutoTokenizers
    """
    assert batch_t.dim() == 3 and batch_input_ids.dim() == 2, f"input needs to be batched, {batch_t.dim()=}, {batch_input_ids.dim=}"

    batch_spans = get_multi_object_phrase_spans(batch_input_ids, tokenizer)
    new_batch_t = []
    
    for spans, t, input_ids in zip(batch_spans, batch_t, batch_input_ids):
        pad_indices = torch.nonzero(input_ids == tokenizer.pad_token_id).squeeze()

        if len(spans) == 0:
            if len(pad_indices) > 0:
                new_batch_t.append(t[:pad_indices.min()])
            else:
                new_batch_t.append(t)
            continue

        t_new = t[:spans[0][0]]
        for i, span in enumerate(spans):
            # average the tokens within the span
            t_new = torch.concat([t_new, t[span[0]:span[1]+1].mean(0).unsqueeze(0)], dim=0)

            # first next span start, or padding token or end of sequence
            next_idx = spans[i+1][0]-1 if i+1 < len(spans) else pad_indices.min()-1 if len(pad_indices)>0 else len(t)

            # concatenate until the next span
            t_new = torch.concat([t_new, t[span[1]+1:next_idx+1]], dim=0)
        new_batch_t.append(t_new)

    new_batch_t = torch.stack(new_batch_t)
    return new_batch_t


def get_root_exp_dir(out_path: str) -> str:
    out_path_list = out_path.split("/")
    out_path_root = "/".join(out_path_list[:out_path_list.index("outputs")+1])
    return out_path_root


def get_basis_directions(
    model: LanguageModel,
    args: argparse.Namespace,
    position: Literal["last_token"]="last_token",
    if_normalize_object_phrase: bool = True,
    cache_dir: Optional[str] = None,
    remote: bool = False,
) -> Tuple[torch.Tensor,List[str]]:
    """
    Computes the SVD of residual streams at different positions.

    Args:
        model: model under investigation.
        args: argparse namespace.
        position: token position we want to compute directions for.
        if_normalize_object_phrase: whether to normalize the object phrase (concat multi-objects into one token).
        cache_dir: directory to cache the results.
        remote: whether to use NDIF remote server or not.

    Returns:
        directions (Tensor): [n_layer, n_basis, n_dim] per-layer basis
        modules (
    """
    position_to_norm_indices = {"last_token": -1}
    print(f"Computing basis directions ...")
    modules = [get_module(model, layer, module_type="resid") for layer in range(model.config.num_hidden_layers)]

    if cache_dir is not None:
        cache_path = f"{cache_dir}/svd_{position}.pt"
        if os.path.exists(cache_path):
            print(f"Loading cached basis directions from: {cache_path}")
            directions = torch.load(open(cache_path, "rb"))
            print(f"Loaded {len(directions)} directions for {position}, shape: {directions.shape}")
            return directions, modules

    ablation_dataloader = load_ablation_data(model=model, tokenizer=model.tokenizer,args=args,)
    raw_activations = defaultdict(list)
    with torch.no_grad():
        for _, inp in enumerate(tqdm(ablation_dataloader)):
            inp["input_ids"] = inp["input_ids"].to(model.device)
            with model.trace(inp["input_ids"], remote=remote) as tracer:
                for layer_idx, layer in enumerate(modules):
                    getter = operator.attrgetter(layer)
                    # rs = getter(model).output[0].detach().save()  # nnsight <= 0.5 behavior, where output used to be a tuple, deprecated after
                    rs = getter(model).output.detach()   # nnsight >= 0.5 behavior, output is just (batch, seq, h_dim)
                    if if_normalize_object_phrase:
                        rs = _nn_apply(normalize_object_phrase, rs, inp["input_ids"], model.tokenizer)
                    rs = rs[:, position_to_norm_indices[position]].cpu().numpy().save()  # [bs, model_dim]
                    raw_activations[layer_idx].append(rs)

            torch.cuda.empty_cache()
    directions = []
    for layer_idx in range(len(modules)):
        activations = np.vstack(raw_activations[layer_idx])
        U, s, Vt = np.linalg.svd(activations.astype(np.float32))
        directions.append(torch.Tensor(Vt))

    directions = torch.stack(directions)

    if cache_dir is not None:
        print(f"Caching {len(directions)} directions for {position}, shape: {directions.shape}")
        torch.save(directions, cache_path)

    return directions, modules



def get_mean_activations(
    model: LanguageModel,
    args: argparse.Namespace,
    if_normalize_object_phrase: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor],List[str]]:
    """
    Computes the mean activations of every attention head at all positions.

    Args:
        model: model under investigation.
        args: argparse namespace.
        if_normalize_object_phrase: whether to normalize the object phrase (concat multi-objects into one token).
    """

    print(f"Computing mean activations ...")
    modules = [get_module(model, layer) for layer in range(model.config.num_hidden_layers)]

    if cache_dir is not None:
        cache_path = f"{cache_dir}/mean_activations.pkl"
        if os.path.exists(cache_path):
            print(f"Loading cached mean activations from: {cache_path}")
            mean_activations = pickle.load(open(cache_path, "rb"))
            return mean_activations, modules

    ablation_dataloader = load_ablation_data(model=model, tokenizer=model.tokenizer,args=args,)
    mean_activations = {}
    with torch.no_grad():
        for _, inp in enumerate(tqdm(ablation_dataloader)):
            inp["input_ids"] = inp["input_ids"].to(model.device)

            with model.trace(inp["input_ids"]) as tracer:
                for layer in modules:
                    getter = operator.attrgetter(layer)
                    o_proj_input = getter(model).input.detach()
                    if if_normalize_object_phrase:
                        o_proj_input = _nn_apply(normalize_object_phrase, o_proj_input, inp["input_ids"], model.tokenizer)
                    if layer in mean_activations:
                        mean_activations[layer] = (mean_activations[layer] + o_proj_input.sum(0)).save()
                    else:
                        mean_activations[layer] = o_proj_input.sum(0).save()
            torch.cuda.empty_cache()

        for layer in modules:
            mean_activations[layer] /= len(ablation_dataloader.dataset)

    if cache_dir is not None:
        print(f"Saving mean activations to: {cache_path}")
        pickle.dump(mean_activations, open(cache_path, "wb"))

    return mean_activations, modules


def load_ablation_data(
    model: LanguageModel,
    tokenizer: LlamaTokenizer,
    args: argparse.Namespace,
):
    """
    Loads the dataset for ablation.

    Args:
        model: model under investigation.
        tokenizer: tokenizer to use.
        datafile: path to the datafile.
        num_samples: number of samples to use from the datafile.
        batch_size: batch size to use for the dataloader.
        num_boxes: number of boxes in the datafile.
    """
    
    raw_data = load_pp_data(
        tokenizer=tokenizer,
        num_samples=int(3500/7), #3500/7, our loader already take into account of unique box states
        num_boxes=7,
        data_file=args.datafile,
        ops_order=args.ops_order,
        query_ops_order=args.query_ops_order,
        max_initial_objects_per_box=args.max_initial_objects_per_box,
        counterfactual_format="rand_obj_rand_query_id", #args.counterfactual_format,
        data_field=args.data_field,
        prepend_space_to_answer=True if any([t in args.model for t in ["gemma"]]) else False,
        model=model,
        success_filter=True, #args.success_filter,  # should be True?
        operations_on_same_obj=args.operations_on_same_obj,
        copy_filter=args.copy_filter,
    )

    ablate_dataset = Dataset.from_dict(
        {
            "input_ids": raw_data["source_tokens"],
            "last_token_indices": raw_data["source_last_token_indices"],
        }
    ).with_format("torch")

    ablate_dataloader = torch.utils.data.DataLoader(
        ablate_dataset, batch_size=args.batch_size, collate_fn=partial(pad_batch_collate_fn, tokenizer=tokenizer)
    )
    return ablate_dataloader


def get_circuit_old(
    model: LanguageModel,
    circuit_root_path: str,
    n_value_fetcher: int,
    n_pos_trans: int,
    n_pos_detect: int,
    n_struct_read: int,
    largest: bool=False
) -> Tuple[Dict, List, List, List, List]:
    """
    Computes the circuit components.

    Args:
        model: model under investigation.
        circuit_root_path: path to the circuit components.
        n_value_fetcher: number of value fetcher heads.
        n_pos_trans: number of position transformer heads.
        n_pos_detect: number of position detector heads.
        n_struct_read: number of structure reader heads.
        top_p: top cumulative probability threshold to select number of heads in each group. Use this or specify number
            of heads per group manually.
        largest: whether to get the heads with largest or the smallest values (from patching).
    """

    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)

    value_fetcher_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupA.npy")), k=n_value_fetcher, largest=largest)
    pos_transmitter_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupB.npy")), k=n_pos_trans, largest=largest)
    pos_detector_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupC.npy")), k=n_pos_detect, largest=largest)
    struct_reader_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupD.npy")), k=n_struct_read, largest=largest)

    print(f"Value fetcher heads: {value_fetcher_heads}")
    print(f"Position transmitter heads: {pos_transmitter_heads}")
    print(f"Position detector heads: {pos_detector_heads}")
    print(f"Structure reader heads: {struct_reader_heads}")

    intersection = []
    for head in value_fetcher_heads:
        if head in pos_transmitter_heads:
            intersection.append(head)

    for head in intersection:
        value_fetcher_heads.remove(head)

    for layer_idx, head in value_fetcher_heads:
        layer = get_module(model, layer_idx)
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_transmitter_heads:
        layer = get_module(model, layer_idx)
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_detector_heads:
        layer = get_module(model, layer_idx)
        circuit_components[2][layer].append(head)

    for layer_idx, head in struct_reader_heads:
        layer = get_module(model, layer_idx)
        circuit_components[-1][layer].append(head)
 
    return (
        circuit_components,
        value_fetcher_heads,
        pos_transmitter_heads,
        pos_detector_heads,
        struct_reader_heads,
    )

def get_circuit(
    model: LanguageModel,
    circuit_root_path: str,
    largest: bool=False,
    n_more_heads_per_group: int=0
) -> Tuple[Dict, List, List, List, List]:
    """
    Computes the circuit components.

    Args:
        model: model under investigation.
        circuit_root_path: path to the circuit components.
        largest: whether to get the heads with largest or the smallest values (from patching).
        n_more_heads_per_group: number of additional heads per group to return. Use this for evaluation.
    """

    circuit_components = {}
    circuit_components[0] = defaultdict(list)
    circuit_components[2] = defaultdict(list)
    circuit_components[-1] = defaultdict(list)

    # value_fetcher_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupA.npy")), k=n_value_fetcher, largest=largest, top_p=top_p)
    # pos_transmitter_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupB.npy")), k=n_pos_trans, largest=largest, top_p=top_p)
    # pos_detector_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupC.npy")), k=n_pos_detect, largest=largest, top_p=top_p)
    # struct_reader_heads = compute_topk_components(torch.tensor(np.load(f"{circuit_root_path}/pp_groupD.npy")), k=n_struct_read, largest=largest, top_p=top_p)

    value_fetcher_heads = compute_topk_components_knee(
        torch.tensor(np.load(f"{circuit_root_path}/pp_groupA.npy")),
        largest=largest,
        knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"},
        increase_cutoff_n=n_more_heads_per_group
    )
        
    pos_transmitter_heads = compute_topk_components_knee(
        torch.tensor(np.load(f"{circuit_root_path}/pp_groupB.npy")), 
        largest=largest, 
        knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"},
        increase_cutoff_n=n_more_heads_per_group
    )
    pos_detector_heads = compute_topk_components_knee(
        torch.tensor(np.load(f"{circuit_root_path}/pp_groupC.npy")), 
        largest=largest, 
        knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"},
        increase_cutoff_n=n_more_heads_per_group
    )
    struct_reader_heads = compute_topk_components_knee(
        torch.tensor(np.load(f"{circuit_root_path}/pp_groupD.npy")), 
        largest=largest, 
        knee_kwargs={"S": 1.0, "curve": "convex", "direction": "decreasing"},
        increase_cutoff_n=n_more_heads_per_group
    )


    print(f"Value fetcher heads: {value_fetcher_heads}")
    print(f"Position transmitter heads: {pos_transmitter_heads}")
    print(f"Position detector heads: {pos_detector_heads}")
    print(f"Structure reader heads: {struct_reader_heads}")

    intersection = []
    for head in value_fetcher_heads:
        if head in pos_transmitter_heads:
            intersection.append(head)

    for head in intersection:
        value_fetcher_heads.remove(head)

    for layer_idx, head in value_fetcher_heads:
        layer = get_module(model, layer_idx)
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_transmitter_heads:
        layer = get_module(model, layer_idx)
        circuit_components[0][layer].append(head)

    for layer_idx, head in pos_detector_heads:
        layer = get_module(model, layer_idx)
        circuit_components[2][layer].append(head)

    for layer_idx, head in struct_reader_heads:
        layer = get_module(model, layer_idx)
        circuit_components[-1][layer].append(head)
 
    return (
        circuit_components,
        value_fetcher_heads,
        pos_transmitter_heads,
        pos_detector_heads,
        struct_reader_heads,
    )


def get_random_circuit(
    model: LanguageModel,
    n_value_fetcher: int,
    n_pos_transmitter: int,
    n_pos_detector: int,
    n_struct_reader: int,
):
    """
    Computes a random circuit with same #heads in each group as in the circuit.

    Args:
        model: model under investigation.
        circuit: dictionary of an existing circuit
    """

    random_circuit = {}
    random_circuit[0] = defaultdict(list)
    random_circuit[2] = defaultdict(list)
    random_circuit[-1] = defaultdict(list)

    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers

    heads_at_last_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_value_fetcher + n_pos_transmitter
    )
    heads_at_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_pos_detector
    )
    heads_at_prev_query_box_pos = np.random.choice(
        list(range(num_heads * num_layers)), n_struct_reader
    )

    heads_at_last_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_last_pos
    ]
    heads_at_query_box_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_query_box_pos
    ]
    heads_at_prev_query_box_pos = [
        [head // num_layers, head % num_heads] for head in heads_at_prev_query_box_pos
    ]

    for layer_idx, head in heads_at_last_pos:
        layer = get_module(model, layer_idx)
        random_circuit[0][layer].append(head)

    for layer_idx, head in heads_at_query_box_pos:
        layer = get_module(model, layer_idx)
        random_circuit[2][layer].append(head)

    for layer_idx, head in heads_at_prev_query_box_pos:
        layer = get_module(model, layer_idx)
        random_circuit[-1][layer].append(head)

    return random_circuit


def eval_circuit_performance(
    model: LanguageModel,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    circuit_components: dict,
    mean_activations: dict,
    ablate_non_vital_pos: bool = True,
):
    """
    Evaluates the performance of the model/circuit.

    Args:
        model: model under investigation.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        circuit_components: circuit components.
        mean_activations: mean activations of the model.
    """
    argmax_correct_any, total_count = 0, 0
    argmax_correct_full, topk_correct_full = [], []

    with torch.no_grad():
        for _, inp in enumerate(tqdm(dataloader)):
            
            inp["input_ids"] = inp["base_tokens"].to(model.device)
            object_spans_batch = get_multi_object_phrase_spans(inp["input_ids"], model.tokenizer)
            
            with model.trace(inp["input_ids"]) as tracer:
                for layer in modules:
                    getter = operator.attrgetter(layer)
                    original_input = getter(model).input.clone()
                    ablated_input = _nn_apply(
                        mean_ablate,
                        original_input,
                        layer,
                        model,
                        circuit_components,
                        mean_activations,
                        inp["input_ids"],
                        inp["base_last_token_indices"],
                        ablate_non_vital_pos,
                        object_spans_batch
                    )
                    rsetattr(model, f"{layer}.input", ablated_input)

                logits = model.lm_head.output.save()

            # moving this outside of tracer context because += operation does not work inside
            for bi in range(len(inp["labels"])):
                labels = inp["labels"][bi]  # multiple target objects
                topk_pred = torch.argsort(logits[bi][inp["base_last_token_indices"][bi]], descending=True)[:len(labels)].cpu().numpy()
                if (topk_pred[0] == labels).sum() > 0:
                    argmax_correct_any += 1

                argmax_correct_full_batch = []
                topk_correct_full_batch = []
                for k, label in enumerate(labels):
                    argmax_correct_full_batch.append(1 if topk_pred[0] == label > 0 else 0)
                    topk_correct_full_batch.append(1 if (topk_pred == label).sum() > 0 else 0)

                total_count += 1
                argmax_correct_full.append(argmax_correct_full_batch)
                topk_correct_full.append(topk_correct_full_batch)

            del logits
            torch.cuda.empty_cache()

    current_acc = round(argmax_correct_any / total_count, 2)
    return current_acc, argmax_correct_full, topk_correct_full


def compute_normalized_token_index_map(
    input_ids: torch.Tensor,
    object_spans:List[Tuple[int, int]]
) -> Dict[int, int]:
    m = {}

    # already normalized, no positional changes
    if len(object_spans) == 0:
        return {i:i for i in range(len(input_ids))}

    i = 0
    norm_i = 0
    for obj_span in object_spans:
        # for every token before the next span, advance both pointers
        while i < obj_span[0]:
            m[i] = norm_i
            i += 1
            norm_i += 1
        # for every token in the span, advance only the source index
        for i in range(obj_span[0], obj_span[1]):
            m[i] = norm_i
            i += 1

    # then from the last span till last token, advance both spans
    while i < len(input_ids):
        m[i] = norm_i
        i += 1
        norm_i += 1

    return m


def mean_ablate(
    inputs=None,
    layer: str=None,
    model: LanguageModel = None,
    circuit_components: dict[int, Dict[str, List[int]]] = None,
    mean_activations: dict[str, torch.Tensor] = None,
    input_tokens: torch.tensor = None,
    last_pos_batch: np.array = None,
    ablate_non_vital_pos: bool = None,
    object_spans_batch: List[List[Tuple[int,int]]] = None,
):
    """
    Ablates the model components that are not present in `circuit_components`
    by substituting their output with their corresponding mean activations.

    Args:
        inputs: inputs to the layer.
        layer: layer to patch.
        model: model to patch.
        circuit_components: circuit components.
        mean_activations: mean activations of the model.
        input_tokens: input tokens.
    """

    if isinstance(inputs, tuple):
        inputs = inputs[0]

    inputs = rearrange(inputs,"batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",n_heads=model.config.num_attention_heads)
    mean_act = rearrange(mean_activations[layer],"seq_len (n_heads d_head) -> seq_len n_heads d_head",n_heads=model.config.num_attention_heads)

    for bi in range(inputs.size(0)):
        prev_query_box_pos = find_previous_query_box_pos({"base_tokens":input_tokens[bi], "base_last_token_indices":last_pos_batch[bi]})
        norm_pos_map = compute_normalized_token_index_map(input_tokens[bi], object_spans_batch[bi])
        last_pos = last_pos_batch[bi]
        for token_pos in range(last_pos + 1):
            norm_token_pos = norm_pos_map[token_pos]
            if (
                    token_pos not in prev_query_box_pos
                    and token_pos != last_pos - 2
                    and token_pos != last_pos
                    and ablate_non_vital_pos
            ):
                inputs[bi, token_pos, :] = mean_act[norm_token_pos, :]

            elif token_pos in prev_query_box_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[-1][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[norm_token_pos, head_idx]

            elif token_pos == last_pos - 2:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[2][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[norm_token_pos, head_idx]

            elif token_pos == last_pos:
                for head_idx in range(model.config.num_attention_heads):
                    if head_idx not in circuit_components[0][layer]:
                        inputs[bi, token_pos, head_idx] = mean_act[norm_token_pos, head_idx]

    inputs = rearrange(inputs,"batch seq_len n_heads d_head -> batch seq_len (n_heads d_head)",n_heads=model.config.num_attention_heads,)
    # w_o = model.state_dict()[f"{layer}.weight"]
    # output = einsum(inputs, w_o, "batch seq_len hidden_size, d_model hidden_size -> batch seq_len d_model")
    return inputs


def get_module(model: LanguageModel, layer: int, module_type:Literal["o_proj","resid"]="o_proj"):
    if model.config.architectures[0] in ["LlamaForCausalLM", "Gemma2ForCausalLM"]:
        if module_type == "o_proj":
            module_name = f"model.layers.{layer}.self_attn.{module_type}"
        elif module_type == "resid":
            module_name = f"model.layers.{layer}"
    else:
        if module_type == "o_proj":
            module_name = f"base_model.model.model.layers.{layer}.self_attn.{module_type}"
        elif module_type == "resid":
            module_name = f"base_model.model.model.layers.{layer}"
    return module_name

def compute_pair_drop_values(
    model: LanguageModel,
    heads: list,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int = 0,
):
    """
    Computes the pair drop values for the given heads.

    Args:
        model: model under investigation.
        heads: heads to compute the pair drop values for.
        circuit_components: circuit components.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        mean_activations: mean activations of the model.
        rel_pos: relative position of the query box label token.
    """

    greedy_res = defaultdict(lambda: defaultdict(float))

    for layer_idx_1, head_1 in tqdm(heads, total=len(heads), desc="Pair drop values"):
        layer_1 = get_module(model, layer=layer_idx_1)
        circuit_components[rel_pos][layer_1].remove(head_1)

        for layer_idx_2, head_2 in heads:
            layer_2 = get_module(model, layer=layer_idx_2)
            if greedy_res[(layer_2, head_2)][(layer_1, head_1)] > 0.0:
                continue
            if layer_1 is not layer_2 and head_1 is not head_2:
                circuit_components[rel_pos][layer_2].remove(head_2)

            pdb.set_trace()
            greedy_res[(layer_1, head_1)][(layer_2, head_2)] = eval_circuit_performance(
                model, dataloader, modules, circuit_components, mean_activations
            )
            if layer_1 is not layer_2 and head_1 is not head_2:
                circuit_components[rel_pos][layer_2].append(head_2)

        circuit_components[rel_pos][layer_1].append(head_1)

    res = defaultdict(lambda: defaultdict(float))
    for k in greedy_res:
        for k_2 in greedy_res[k]:
            if greedy_res[k][k_2] > 0.0:
                res[str(k)][str(k_2)] = greedy_res[k][k_2]
                res[str(k_2)][str(k)] = greedy_res[k][k_2]

    return res


def get_head_significance_score(
    model: LanguageModel,
    heads: list,
    ranked: dict,
    percentage: float,
    circuit_components: dict,
    dataloader: torch.utils.data.DataLoader,
    modules: list,
    mean_activations: dict,
    rel_pos: int,
):
    """
    Computes the head significance score for the given heads.

    Args:
        model: model under investigation.
        heads: heads to compute the pair drop values for.
        ranked: ranked pair drop values.
        percentage: percentage of heads to use for computing the head significance score.
        circuit_components: circuit components.
        dataloader: dataloader containing clean and corrupt inputs.
        modules: modules to patch.
        mean_activations: mean activations of the model.
        rel_pos: relative position of the query box label token.
    """

    res = {}

    for layer_idx, head in tqdm(
        heads, total=len(heads), desc="Head significance score"
    ):
        if model.config.architectures[0] in ["LlamaForCausalLM", "GemmaForCausalLM"]:
            layer = f"model.layers.{layer_idx}.self_attn.o_proj"
        else:
            layer = f"base_model.model.model.layers.{layer_idx}.self_attn.o_proj"

        for r in ranked[str((layer, head))][: math.ceil(percentage * len(ranked.values()))]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].remove(top_head)

        before = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        circuit_components[rel_pos][layer].remove(head)
        after = eval_circuit_performance(
            model, dataloader, modules, circuit_components, mean_activations
        )
        res[(layer, head)] = (before, after)

        for r in ranked[str((layer, head))][: math.ceil(percentage * len(ranked.values()))]:
            top_layer = r[0].split(",")[0][2:-1]
            top_head = int(r[0].split(",")[1][:-1])
            if r[1] <= 0:
                break
            circuit_components[rel_pos][top_layer].append(top_head)
        circuit_components[rel_pos][layer].append(head)

    return res


def compute_topk_components(
    patching_scores: torch.Tensor, k: Optional[int]=None, largest=True, return_values=False, top_p: Optional[float]=None
):
    """
    Computes the topk most influential components (i.e. heads) for patching.

    Args:
        patching_scores: patching scores for the components.
        k: number of components to return.
        largest: whether to return the largest or smallest components.
        return_values: whether to return the values of the components or not.
    """

    if top_p is not None:
        assert 0 < top_p <= 1.0, "top_p must be in (0, 1]"

        flat_scores = patching_scores.flatten()

        # Sort scores
        sorted_values, sorted_indices = torch.sort(flat_scores, descending=largest)

        # Use absolute values for mass if desired (comment out if not)
        # mass = sorted_values.abs()
        mass = sorted_values

        cumulative_mass = torch.cumsum(mass, dim=0)
        total_mass = mass.sum()
        # Find cutoff index
        cutoff = torch.searchsorted(cumulative_mass, top_p * total_mass).item() + 1

        # if largest:
        #     total_mass = cumulative_mass.max()
        #     # Find cutoff index (if positive then negative values, cum mass peaks then drop)
        #     cutoff = torch.searchsorted(cumulative_mass[:torch.argmax(cumulative_mass)], top_p * total_mass).item() + 1
        # else:
        #     total_mass = cumulative_mass.min()
        #     # Find cutoff index (if negative then positive values, cum mass decrease to a bottom then rise to the top)
        #     cutoff = torch.searchsorted(-cumulative_mass[:torch.argmin(cumulative_mass)], -top_p * total_mass).item() + 1

        if k is not None:
            cutoff = min(cutoff, k)
        top_indices = sorted_indices[:cutoff]
        top_values = sorted_values[:cutoff]
    else: # top_k
        top_indices = torch.topk(patching_scores.flatten(), k, largest=largest).indices
        top_values = torch.topk(patching_scores.flatten(), k, largest=largest).values

    # Convert the top_indices to 2D indices
    row_indices = top_indices // patching_scores.shape[1]
    col_indices = top_indices % patching_scores.shape[1]
    top_components = torch.stack((row_indices, col_indices), dim=1)
    # Get the top indices as a list of 2D indices (row, column)
    top_components = top_components.tolist()

    if return_values:
        return top_components, top_values.tolist()
    else:
        return top_components

def compute_topk_components_knee(
    patching_scores: torch.Tensor,
    largest: bool = True,
    return_values: bool = False,
    knee_kwargs: Optional[Dict[str, Any]] = None,
    return_knee: bool = False,
    increase_cutoff_n: int = 0
):
    """
    Select components by:
      - use_knee=True: cut at knee/elbow using kneed (recommended for your request)
      - else if top_p is not None: nucleus/top-p over "mass"
      - else: top-k

    Special behavior:
      - If largest=False, we filter to NEGATIVE scores only (drop >=0) because you only
        want the most negative heads.

    knee_kwargs are forwarded to KneeLocator, e.g.:
      dict(S=1.0, curve="convex", direction="decreasing", interp_method="interp1d", online=False, polynomial_degree=7)
    increase_cutoff controls how many additional heads you want to include.
    """
    if patching_scores.ndim != 2:
        raise ValueError(f"Expected 2D tensor (layers, heads), got shape {tuple(patching_scores.shape)}")

    flat = patching_scores.flatten()

    # For largest=False, drop non-negative values (your use case).
    if not largest:
        neg_mask = flat < 0
        if not torch.any(neg_mask):
            if return_knee and return_values:
                return [], [], None
            if return_knee:
                return [], None
            if return_values:
                return [], []
            return []
        candidate_scores = flat[neg_mask]
        candidate_flat_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
    else:
        candidate_scores = flat
        candidate_flat_indices = torch.arange(flat.numel(), device=flat.device)

    # -------------------------
    # Knee / elbow cutoff branch
    # -------------------------
    # Sort so "best" are first.
    # - largest=False -> most negative first (ascending)
    # - largest=True  -> most positive first (descending)
    sorted_scores, order = torch.sort(candidate_scores, descending=largest)
    sorted_flat_indices = candidate_flat_indices[order]

    # kneed runs on CPU/numpy
    # Use magnitudes of the selected direction so the curve is typically decreasing.
    # If largest=False: sorted_scores are negative increasing toward 0, so -sorted_scores decreases.
    y = (-sorted_scores).detach().cpu().numpy()

    n = y.shape[0]
    if n < 3:
        cutoff = n
    else:
        x = np.arange(1, n + 1, dtype=float)

        # sensible defaults for a decreasing magnitude curve
        kw = dict(S=1.0, curve="convex", direction="decreasing", interp_method="interp1d", online=False)
        if knee_kwargs:
            kw.update(knee_kwargs)

        kl = KneeLocator(x, y, **kw)

        knee_x = kl.knee if kl.knee is not None else kl.elbow  # elbow/knee are interchangeable in kneed
        if knee_x is None:
            cutoff = n  # fallback: keep all candidates
        else:
            cutoff = int(round(float(knee_x)))
            cutoff = max(1, min(cutoff, n))

    cutoff += increase_cutoff_n

    top_indices = sorted_flat_indices[:cutoff]
    top_values = flat[top_indices]

    # Convert to (layer, head)
    n_heads = patching_scores.shape[1]
    row_indices = top_indices // n_heads
    col_indices = top_indices % n_heads
    top_components = torch.stack((row_indices, col_indices), dim=1).tolist()

    if return_knee:
        knee_info = {
            "cutoff": cutoff,
            "n_candidates": int(candidate_scores.numel()),
            "knee_kwargs": knee_kwargs or {},
        }
        if return_values:
            return top_components, top_values.tolist(), knee_info
        return top_components, knee_info

    if return_values:
        return top_components, top_values.tolist()
    return top_components

def find_previous_query_box_pos(data: Dict[str, Any]) -> List[int]:
    """
    Compute the position of the previous query box label token (excluding query statement)

    Args:
        data: dictionary of ['base_tokens', 'base_last_token_indices', 'source_tokens', 'source_last_token_indices',
            'labels']
    """
    input_ids = data['base_tokens']
    query_box_token = input_ids[data["base_last_token_indices"] - 2]
    if isinstance(input_ids, np.ndarray):
        prev_query_box_token_pos = (input_ids[: data["base_last_token_indices"] - 2] == query_box_token).nonzero()[0]
    else:
        prev_query_box_token_pos = (input_ids[: data["base_last_token_indices"] - 2] == query_box_token).nonzero().squeeze(-1)
    return prev_query_box_token_pos


def get_model_and_tokenizer(model_name, tokenizer_only: bool=False, for_eap: bool=True):
    padding_side = "right"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if not tokenizer_only:
        if "llama-7b" in model_name:
            model_hf = AutoModelForCausalLM.from_pretrained("luodian/llama-7b-hf")
            tokenizer = AutoTokenizer.from_pretrained("luodian/llama-7b-hf", padding_side=padding_side)
            model = HookedTransformer.from_pretrained_no_processing(
                "llama-7b",
                hf_model=model_hf,
                tokenizer=tokenizer,
                device=device,
                dtype=torch.bfloat16
            )  # Same model used in previous work.
        elif model_name == "nikhil07prakash/float-7b":
            model_hf = AutoModelForCausalLM.from_pretrained("nikhil07prakash/float-7b")
            tokenizer = AutoTokenizer.from_pretrained("luodian/llama-7b-hf", padding_side=padding_side)
            model = HookedTransformer.from_pretrained_no_processing(
                "llama-7b",
                hf_model=model_hf,
                tokenizer=tokenizer,
                device=device,
                dtype=torch.bfloat16
            )
        elif model_name == "codellama/CodeLlama-13b-hf":
            model_hf = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
            model = HookedTransformer.from_pretrained_no_processing(
                "Llama-2-13b",
                hf_model=model_hf,
                tokenizer=tokenizer,
                # device=device,
                device_map=device,
                dtype=torch.bfloat16,
                n_devices=torch.cuda.device_count(),
            )
        else:
            # model = HookedTransformer.from_pretrained(model_name, dtype=torch.bfloat16, device="cuda")
            # with reduced precision, it's advised to use `from_pretrained_no_processing`

            # model = HookedTransformer.from_pretrained_no_processing(model_name, dtype=torch.bfloat16, device="cuda")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN,
                                                      padding_side=padding_side)
            model = HookedTransformer.from_pretrained_no_processing(
                model_name, dtype=torch.bfloat16, device=device, use_auth_token=HUGGINGFACE_TOKEN
            )
        if for_eap:
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True
    else:
        # fast loading tokenizer only for debugging dataset loader
        model = None
        if "float" in model_name:
            tokenizer = AutoTokenizer.from_pretrained("luodian/llama-7b-hf", padding_side=padding_side)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = padding_side
    torch.cuda.empty_cache()
    return model, tokenizer


def get_random_guess_baseline(dataset: Iterable[Tuple[str,str,np.array]]):
    acc = []
    for data in dataset:
        objs = get_objects(data[0])
        acc.append(1/len(set(objs)))
    return np.mean(acc)


def get_objects(s: str) -> List[str]:
    return remove_substrings(s.lower(), NON_OBJ_WORDS).split()


def get_token_semantic_type(t: str) -> str:
    t = t.strip().lower()
    if t.isnumeric():
        return "box_id"
    elif t == "box":
        return "box"
    elif t in {"move", "remove", "put"}:
        return "op"
    elif t in {",", "."}:
        return "punct"
    elif t == "the":
        return "the"
    elif t.startswith("contain"):
        return "contains"
    elif t in NON_OBJ_WORDS:
        return "others"
    else:
        return "objs"


def is_int_with_negatives(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    # If there's a dot, recursively get the nested object
    if pre:
        target_obj = rgetattr(obj, pre)
    else:
        target_obj = obj
    setattr(target_obj, post, val)

def rgetattr(obj, attr):
    # Helper to get nested attributes
    return functools.reduce(getattr, [obj] + attr.split('.'))


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)  # use start += 1 to find overlapping matches


def generate_counterfactual_cma_query_id(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None) -> Tuple[str, str, Optional[List[str]]]:
    query_box = prompt[prompt.rfind("Box") + 4]
    # change all mentions of query box to a new box id (8)
    locs = list(find_all(prompt, f"Box {query_box}"))
    if replace_indices is not None:
        # last occurance is special, always added (otherwise prompt is invalid)
        locs = locs[:-1][replace_indices] + [locs[-1]]

    new_prompt = prompt
    for loc in locs:
        new_prompt = new_prompt[:loc] + "Box 8" + new_prompt[loc + 5:]

    # change clean prompt to query 8 (but not in its context)
    prompt = prompt[:locs[-1]] + "Box 8" + prompt[locs[-1] + 5:]
    return new_prompt, prompt, None


def generate_counterfactual_cma_query_id_last_obj(prompt: str,tokenizer: AutoTokenizer) -> Tuple[str, str, Optional[List[str]]]:
    return generate_counterfactual_cma_query_id(prompt, replace_indices=[-1])


def generate_counterfactual_cma_query_id_first_obj(prompt: str,tokenizer: AutoTokenizer) -> Tuple[str, str, Optional[List[str]]]:
    return generate_counterfactual_cma_query_id(prompt, replace_indices=[0])


def generate_counterfactual_cma_remove_query_id(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Tuple[str, str, List[str]]:
    new_prompt, prompt, _ = generate_counterfactual_cma_query_id(prompt, replace_indices=replace_indices)
    # in addition, we need to measure the removed objects.
    # heuristically here we assume second to last occurance is removed obj
    assert len(list(find_all(prompt, f"Remove"))) == 1, "Currently only support 1 remove case"
    locs = list(find_all(new_prompt, f"Box 8"))
    last_remove_phrase = get_phrase_around_idx(new_prompt, locs[-2])
    removed_objs = get_objects(last_remove_phrase)

    return new_prompt, prompt, removed_objs

def generate_counterfactual_cma_remove_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Tuple[str, str, List[str], List[str]]:
    """
    replace all occurance of removed obj to a new objs
    (Cake -> Apple), book are in box 0. … Remove the (cake -> apple) from box 0. Box 0 contains the book
    target:
    """

    assert len(list(find_all(prompt, f"Remove"))) == 1, "Currently only support 1 remove case"
    query_box = prompt[prompt.rfind("Box") + 4]
    locs = list(find_all(prompt, f"Box {query_box}"))
    last_remove_phrase = get_phrase_around_idx(prompt, locs[-2])
    removed_objs = get_objects(last_remove_phrase)

    world_objs = get_objects(prompt)
    not_used_objs = [o for o in all_objects if o not in world_objs]
    replaced_objs = random.sample(not_used_objs, k=len(removed_objs))

    new_prompt = prompt
    for old_obj, new_obj in zip(removed_objs, replaced_objs):
        if replace_indices is None:
            new_prompt = re.sub(r"\b%s\b" % old_obj, new_obj, new_prompt)
            new_prompt = re.sub(r"\b%s\b" % old_obj.capitalize(), new_obj.capitalize(), new_prompt)
        elif replace_indices == [-1]: # replace just remove phrase
            new_prompt = re.sub(r"\b%s\b" % f"Remove the {old_obj}", f"Remove the {new_obj}", new_prompt)
        else:  # replace description but not remove phrase
            new_prompt = re.sub(r"\b%s\b" % old_obj, new_obj, new_prompt)
            new_prompt = re.sub(r"\b%s\b" % old_obj.capitalize(), new_obj.capitalize(), new_prompt)
            new_prompt = re.sub(r"\b%s\b" % f"Remove the {new_obj}", f"Remove the {old_obj}", new_prompt)
    return new_prompt, prompt, removed_objs, replaced_objs

def generate_counterfactual_cma_remove_obj_og_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on clean object (removed in clean data) """
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, replace_indices, all_objects)
    return {"new_prompt":new_prompt, "labels":removed_objs}

def generate_counterfactual_cma_remove_obj_og_obj_last_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on clean object (removed in clean data) """
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, [-1], all_objects)
    return {"new_prompt":new_prompt, "labels":removed_objs}

def generate_counterfactual_cma_remove_obj_og_obj_first_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on clean object (removed in clean data), ctf only differ in removal phrase"""
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, [0], all_objects)
    return {"new_prompt":new_prompt, "labels":removed_objs}

def generate_counterfactual_cma_remove_obj_new_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on ctf object (removed in ctf data), ctf only differ in removal phrase"""
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, replace_indices, all_objects)
    return {"new_prompt":new_prompt, "labels":replaced_objs}

def generate_counterfactual_cma_remove_obj_new_obj_last_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on ctf object (removed in ctf data), ctf only differ in description phrase """
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, [-1], all_objects)
    return {"new_prompt":new_prompt, "labels":replaced_objs}

def generate_counterfactual_cma_remove_obj_new_obj_first_obj(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """ Focused on ctf object (removed in ctf data), ctf only differ in removal phrase """
    new_prompt, prompt, removed_objs, replaced_objs = generate_counterfactual_cma_remove_obj(prompt, [0], all_objects)
    return {"new_prompt":new_prompt, "labels":replaced_objs}


def generate_counterfactual_cma_remove_op(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """
    change 1 remove to 1 put
    Apple, orange are in box 0. … (Put->Remove) the apple (into->from) box 0. Box 0 contains the orange
    target: apple
    """

    assert len(list(find_all(prompt, f"Remove"))) == 1, "Currently only support 1 remove case"
    assert replace_indices is None or len(replace_indices) == len(all_objects)
    query_box = prompt[prompt.rfind("Box") + 4]
    locs = list(find_all(prompt, f"Box {query_box}"))
    last_remove_phrase = get_phrase_around_idx(prompt, locs[-2])
    removed_objs = get_objects(last_remove_phrase)
    new_prompt = re.sub(r"\b%s\b" % "Remove", "Put", prompt)
    new_prompt = re.sub(r"\b%s\b" % "from", "into", new_prompt)
    return {"new_prompt":new_prompt, "labels":removed_objs}


def generate_counterfactual_cma_remove_op_reverse(prompt: str, replace_indices: Optional[Union[List[int], slice]] = None, all_objects: List[str]=None) -> Dict[str, Any]:
    """
    change 1 remove to 1 put
    Apple, orange are in box 0. … (Remove->put) the apple (from->into) box 0. Box 0 contains the orange
    target: apple
    """
    new_prompt, prompt, removed_objs = generate_counterfactual_cma_remove_op(prompt, replace_indices, all_objects)
    return {"new_prompt":prompt, "changed_input_prompt": new_prompt, "labels":removed_objs}




def generate_counterfactual_add_single_op(prompt: str, all_objs: Set[str], op: str, is_query: str, label:str="op", swap_description_order:bool=False) -> Dict[str, Any]:
    """
    Given old prompt add a single operation to the source prompt

    Args:
        prompt (str): the sentence that does not include target objects
        all_objs (set[str]): set of all objects in the world of the dataset
        tokenizer (AutoTokenizer): the AutoTokenizer used to tokenize the prompt
        op (str): the operation to add
        is_query (str): whether the operation affects the query box [query, irrelevant, irrelevant-illegal, irrelevant-swap]
            - irrelevant-illegal: for when we add operation but using the object of the query phrase
            - irrelevant-swap: for when we add operation and also swap the objects such that the irrelevant box now has the query box content
        label (str): ["op","correct"], set label to
            - op: the object involved in the operation
            - correct: the object that is actually in the query box
        swap_description_order (bool): whether to swap description phrase object order

    Returns:
        new_prompt: the counterfactual/new prompt
        labels: object we want to measure (e.g. if we remove something, we want to
            measure the logit of the removed object)
        changed_input_prompt: the changed input prompt (only valid for irrelevant-swap)
        changed_input_labels: the changed input labels (only valid for irrelevant-swap)
    """
    op = op.lower().strip()
    query_box = prompt[prompt.rfind("Box") + 4]
    assert all([o not in prompt for o in ["Move ", "Put ", "Remove "]]), "currently only supporting no-op datapoints"
    assert is_query in ["query", "irrelevant", "irrelevant-illegal", "irrelevant-swap"]
    if is_query == "query":
        box = query_box
    else:
        box = random.choice([str(b) for b in range(7) if str(b) != query_box])

    desc_phrases = prompt.split(". ")[0].split(", ")

    if is_query == "irrelevant-swap":
        # if swap, we change the content of the query and selected box
        desc_phrases[int(box)], desc_phrases[int(query_box)] = desc_phrases[int(query_box)], desc_phrases[int(box)]
        desc_phrases[int(box)] = desc_phrases[int(box)].replace(query_box, box)
        desc_phrases[int(query_box)] = desc_phrases[int(query_box)].replace(box, query_box)
        prompt = "T" + ", ".join(desc_phrases).replace(" The ", " the ")[1:] + ". " + ". ".join(prompt.split(". ")[1:])
        original_labels = get_objects(desc_phrases[int(query_box)])

    objects_per_box = [get_objects(p) for p in desc_phrases]
    global_used_objects = get_objects(prompt)
    global_unused_objects = [o for o in all_objs if o not in global_used_objects]

    labels = None
    if op == "put":
        obj = random.choice(global_unused_objects)
        op_phrase = f"Put the {obj} into Box {box}."
        if label == "ecorrect":
            labels = [objects_per_box[int(box)], obj]  # gold label for ctf data
        elif label == "op":
            labels = [obj]  # object added to the box

    elif op == "remove":
        box_objects = objects_per_box[int(query_box)] if is_query == "irrelevant-illegal" else objects_per_box[int(box)]
        # assert len(box_objects) > 1 or is_query!="query", "Don't want to predict empty box states"
        # obj = random.choice(box_objects)
        obj = box_objects[0]  # instead of random removing, remove 1st one, forcing model to change prediction
        op_phrase = f"Remove the {obj} from Box {box}."
        if label == "correct":
            labels = [o for o in box_objects if o!=obj]  # gold label for ctf data
        else:
            labels = [obj] # object removed

    elif op == "move_0":
        box2 = random.choice([str(b) for b in range(7) if str(b) != query_box and str(b) != box])
        box_objects = objects_per_box[int(query_box)] if is_query == "irrelevant-illegal" else objects_per_box[int(box)]
        # assert len(box_objects) > 1 or is_query!="query", "Don't want to predict empty box states"
        # obj = random.choice(box_objects)
        obj = box_objects[0]  # instead of random removing, remove 1st one, forcing model to change prediction
        op_phrase = f"Move the {obj} in Box {box} to Box {box2}."
        if label == "correct":
            labels = [o for o in box_objects if o!=obj] # gold label for ctf data
        elif label == "op":
            labels = [obj] # object removed

    else: # move_1
        box2 = random.choice([str(b) for b in range(7) if str(b) != query_box and str(b) != box])
        box_objects = objects_per_box[int(box2)]
        obj = random.choice(box_objects)
        op_phrase = f"Move the {obj} in Box {box2} to Box {box}."
        if label == "correct":
            labels = [objects_per_box[int(box)], obj]  # gold label for ctf data
        elif label == "op":
            labels = [obj] # object added

    if labels is None:
        raise NotImplementedError(f"label method={label} if not implemented")

    if swap_description_order:
        og_phrase = desc_phrases[int(box)]
        og_objs = get_objects(og_phrase)
        og_objs_swap = og_objs[::-1]
        swapped_phrases = " and ".join([f"the {o}" for o in og_objs_swap]).capitalize() + f" are in Box {box}"
        desc_phrases[int(box)] = swapped_phrases

    new_prompt = ", ".join(desc_phrases) + f". {op_phrase} " + prompt.split(". ")[1]
    out_dict = {"new_prompt":new_prompt, "labels":labels}
    if is_query == "irrelevant-swap":
        out_dict.update({"changed_input_prompt": prompt, "changed_input_labels": original_labels})
    return out_dict

def is_insertion_phrase(s: str, query_box_id: int) -> bool:
    if " is in Box " in s or " are in Box ":
        return True
    if "Put " in s:
        return True
    if "Remove " in s:
        return False
    if "Move " in s:
        if s.strip(".").endswith(f"Box {query_box_id}"):
            return True
        else:
            return False
    raise ValueError(f"Unable to determine insertion or not for phrase: {s}")

def capital_first_letter(s: str) -> str:
    return s[0].upper() + s[1:]

def decapital_first_letter(s: str) -> str:
    return s[0].lower() + s[1:]

def generate_counterfactual_dcm_obj_or_pos(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Given old prompt and list of valid objects, generate specific counterfactuals which we can test
    whether information about the object or the position of the object (or maybe something else)
     is being tracked.

    Args:
        prompt (str): the sentence that does not include target objects
        all_objects (Set[str]): set of all valid objects we can use for counterfactuals
        hypothesis (str): the hypothesis we are testing:
            obj: object information
            pos_phrase_ctf_op:

    Returns:
        new_prompt: the counterfactual prompt
        labels: list of expected target objects if hypothesis were to be true
        label_types: list of target object types (e.g. [description, op1_put, op2_move1]), for better behavior categorization
    """
    all_objects = kwargs["all_objects"]
    hypothesis = kwargs["hypothesis"]

    query_box_id = int(prompt[prompt.rfind("Box") + 4])
    # swap description phrases around
    desc_phrases = decapital_first_letter(prompt).split(".")[0].split(", ")
    new_desc_phrases = desc_phrases.copy()
    while desc_phrases[query_box_id] == new_desc_phrases[query_box_id]:
        random.shuffle(new_desc_phrases)
    new_prompt = ", ".join(new_desc_phrases) + ". " + ". ".join(prompt.split(". ")[1:])
    new_prompt = capital_first_letter(new_prompt)

    # swap operation phrases around
    op_phrases = new_prompt.split(". ")[1:-1]
    new_op_phrases = op_phrases.copy()
    if len(op_phrases) > 1:
        query_op_indices = [i for i, op in enumerate(op_phrases) if f"Box {query_box_id}" in op]
        while any(new_op_phrases[i] == op_phrases[i] for i in query_op_indices):
            random.shuffle(new_op_phrases)
        new_prompt = new_prompt.split(". ")[0] + ". " + ". ".join(new_op_phrases) + ". " + prompt.split(". ")[-1]

    # map old objects to new objects
    used_objects = get_objects(prompt)
    unused_objects = [o for o in all_objects if o not in used_objects]
    alt_objects = random.sample(unused_objects, k=len(used_objects))
    obj_map = {o: alt_objects[i] for i, o in enumerate(used_objects)}
    def replace_objects(p: str, obj_map: Dict[str, str]) -> str:
        for old_obj, new_obj in obj_map.items():
            p = p.replace(f" {old_obj} ", f" {new_obj} ").replace(f" {old_obj},", f" {new_obj},").replace(f" {old_obj}.", f" {new_obj}.")
        return p
    new_prompt = replace_objects(new_prompt, obj_map)
    # also map the new phrase lists with new objects
    new_desc_phrases = [replace_objects(p, obj_map) for p in new_desc_phrases]
    new_op_phrases = [replace_objects(p, obj_map) for p in new_op_phrases]

    # compute ground-truth object for counterfactual
    labels, label_types = [], []
    if hypothesis == "obj":
        # add description phrase object(s)
        old_desc_objs = get_objects(desc_phrases[query_box_id])
        new_desc_objs = [obj_map[o] for o in old_desc_objs]
        labels.extend(new_desc_objs)
        label_types.extend(["description"]*len(new_desc_objs))
        # add operation phrase object(s)
        for i, op in enumerate(new_op_phrases):
            if f"Box {query_box_id}" in op:
                op_objs = get_objects(op)
                # if object not in box, then it's move (into) or put, otherwise it's remove
                if op_objs[0] in labels:
                    [labels.remove(o) for o in op_objs]
                else:
                    [labels.append(o) for o in op_objs]
                    op_type = get_operation_type(op)
                    label_types.extend([f"op{i}_{op_type}"]*len(op_objs))

    elif hypothesis.startswith("pos_phrase"):
        # loop through all phrases, find phrase index of object, find corresponding object of that index in clean
        for pid, new_phrase in enumerate([*new_desc_phrases, *new_op_phrases]):
            if f"Box {query_box_id}" in new_phrase:
                old_phrase = desc_phrases[pid] if pid < 7 else op_phrases[pid-7]
                objs = get_objects(old_phrase)
                new_op_type = "description" if pid < 7 else f"op{pid-7}_{get_operation_type(new_phrase)}"
                old_op_type = "description" if pid < 7 else f"op{pid-7}_{get_operation_type(old_phrase)}"
                for obj in objs:
                    # TODO just for thoughts, objs in old box are different set of objs than new box,
                    # so they would never appear unless they were put there before with counterfactual patching
                    # if hypothesis is that operation follows the counterfactual's operation
                    if "ctf_op" in hypothesis:
                        if is_insertion_phrase(new_phrase, query_box_id) and obj not in labels:
                            labels.append(obj)
                            label_types.append(new_op_type)
                        elif not is_insertion_phrase(new_phrase, query_box_id) and obj in labels:
                            labels.remove(obj)
                    # if hypothesis is that operation follows the original prompt's operation
                    elif "og_op" in hypothesis:
                        if is_insertion_phrase(old_phrase, query_box_id) and obj not in labels:
                            labels.append(obj)
                            label_types.append(old_op_type)
                        elif not is_insertion_phrase(old_phrase, query_box_id) and obj in labels:
                            labels.remove(obj)
                    # if hypothesis is that operation is just whatever is legal
                    elif "legal_op" in hypothesis:
                        if obj in labels:
                            labels.remove(obj)
                            label_types.append(new_op_type)  # TODO not sure if this is right, but can't think of an alternative
                        else:
                            labels.append(obj)
    # elif hypothesis == "pos_obj":
    #     # for each object, append/remove the object of the same index in the clean sentence
    #     old_objs = get_objects(prompt)
    #     new_objs = [obj_map[o] for o in old_objs]
    #     for pid, new_phrase in enumerate([*new_desc_phrases, *new_op_phrases]):
    else:
        raise NotImplementedError(f"hypothesis {hypothesis} not implemented.")
    # pdb.set_trace()
    return {"new_prompt":new_prompt, "labels": labels, "label_types": label_types}


def get_operation_type(operation_phrase: str) -> str:
    operation_phrase = operation_phrase.strip().lower()
    return "put" if operation_phrase.startswith("put") else "remove" if operation_phrase.startswith("remove") else "move"


def generate_counterfactual_dcm_remove_second_oid(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Ctf: Apple, orange in box 0... Remove apple(*) from box 0. box 0 contains (orange)
    Org: Orange, Apple in box 0... Remove apple(*) from box 0. box 0 contains (orange)
    Patch target: apple (around middle layer)

    Args:
        prompt (str): the sentence that does not include target objects
        tokenizer (AutoTokenizer): model's tokenizer
        pos: (str): all, box, obj, which token position to patch

    Returns:
        new_prompt: the counterfactual prompt
        labels: list of expected target objects if hypothesis were to be true
        patch_locations: List of tuples (src, tgt) locations
    """
    assert prompt.count(". Remove the") == 1
    tokenizer = kwargs["tokenizer"]
    patch_location = kwargs.get("pos", "obj")
    # break_up_description = kwargs.get("break_up_description", False)

    query_box_id = prompt[prompt.rfind("Box") + 4]
    description_phrases = prompt.split(". ")[0].split(", ")
    query_description_phrase = description_phrases[int(query_box_id)]
    init_objects = get_objects(query_description_phrase)
    assert len(init_objects) == 2
    # "The wheel and the crown are in Box 0
    # if break_up_description:
    #     new_query_phrase = f"the {init_objects[1]} is in Box {query_box_id}, the {init_objects[0]} is in Box {query_box_id}"
    # else:
    new_query_phrase = f"the {init_objects[1]} and the {init_objects[0]} are in Box {query_box_id}"
    description_phrases[int(query_box_id)] = new_query_phrase
    new_prompt = ", ".join(description_phrases) + prompt[prompt.find(". "):]
    # if break_up_description:
    #     description_phrases[int(query_box_id)] = f"the {init_objects[0]} is in Box {query_box_id}, the {init_objects[1]} is in Box {query_box_id}"
    #     prompt = ", ".join(description_phrases) + prompt[prompt.find(". "):]

    removal_phrase = prompt.split(". ")[1]
    removed_objs = get_objects(removal_phrase)
    assert len(removed_objs) == 1

    obj_start= len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the {removed_objs[0]}")) - 1
    box_id_start = len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the {removed_objs[0]} from Box 0")) - 1
    patch_locations = []
    if patch_location in ["all", "obj"]:
        patch_locations.append((obj_start, obj_start))
    elif patch_location in ["all", "phrase"]:
        patch_locations.append((obj_start -2, obj_start - 2))  # Remove
        patch_locations.append((obj_start - 1, obj_start + 3))  # the
        patch_locations.append((obj_start, obj_start))
        patch_locations.append((obj_start + 1, obj_start + 1))  # from
        patch_locations.append((obj_start + 2, obj_start + 2))  # Box
        patch_locations.append((obj_start + 3, obj_start + 3))  # space
    if patch_location in ["all", "box", "phrase"]:
        patch_locations.append((box_id_start, box_id_start))
    if patch_location in ["all", "period", "phrase"]:
        patch_locations.append((box_id_start+1, box_id_start+1))
    if patch_location in ["all"]:
        # patch_locations.append((box_id_start + 2, box_id_start + 2))  # box
        # patch_locations.append((box_id_start + 3, box_id_start + 3))  # space
        # patch_locations.append((box_id_start + 4, box_id_start + 4))  # query id
        # patch_locations.append((box_id_start + 5, box_id_start + 5))  # contains
        patch_locations.append((box_id_start + 6, box_id_start + 6))  # the

    return {"new_prompt": new_prompt, "labels": removed_objs, "patch_locations": patch_locations}
    # return {"new_prompt": new_prompt,"changed_input_prompt":prompt, "labels": removed_objs, "patch_locations": patch_locations}

def generate_counterfactual_dcm_remove_second_oid_alt(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Ctf: Apple, orange in box 0... Remove orange(*) from box 0(*). box 0 contains (apple)
    Org: Apple, orange in box 0... Remove apple(*) from box 0(*). box 0 contains (orange)
    Patch target: apple (around middle layer)

    Args:
        prompt (str): the sentence that does not include target objects
        tokenizer (AutoTokenizer): model's tokenizer
        pos: (str): both, box, obj, which token position to patch
        force_removed_obj_order: ["forceLast", "forceFirst", "no"]

    Returns:
        new_prompt: the counterfactual prompt
        labels: list of expected target objects if hypothesis were to be true
        patch_locations: List of tuples (src, tgt) locations
    """
    assert prompt.count(". Remove the") == 1
    tokenizer = kwargs["tokenizer"]
    patch_location = kwargs.get("pos", "both")
    force_removed_obj_order = kwargs.get("force_removed_obj_order", "no")
    query_box_id = prompt[prompt.rfind("Box") + 4]
    description_phrases = prompt.split(". ")[0].split(", ")
    query_description_phrase = description_phrases[int(query_box_id)]
    init_objects = get_objects(query_description_phrase)
    assert len(init_objects) == 2

    removal_phrase = prompt.split(". ")[1]
    removed_objs = get_objects(removal_phrase)
    assert len(removed_objs) == 1
    new_removed_objs = [o for o in init_objects if o not in removed_objs]
    new_removal_phrase = f"Remove the {new_removed_objs[0]} from Box {query_box_id}"
    new_prompt = prompt.replace(removal_phrase, new_removal_phrase)

    # if the object removed is not of a particular order, change it so we are consistent
    # if ((force_removed_obj_order == "forceFirst" and removed_objs[0]==init_objects[1]) or
    #     (force_removed_obj_order == "forceLast" and removed_objs[0]==init_objects[0])):
    #     # need to swap description obj order
    #     prompt = prompt.replace(f"{init_objects[0]} and the {init_objects[1]}", f"{init_objects[1]} and the {init_objects[0]}")
    #     new_prompt = new_prompt.replace(f"{init_objects[0]} and the {init_objects[1]}",f"{init_objects[1]} and the {init_objects[0]}")
    # #

    obj_start= len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the {removed_objs[0]}")) - 1
    box_id_start = len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the {removed_objs[0]} from Box 0")) - 1
    patch_locations = []
    if patch_location in ["all", "obj"]:
        patch_locations.append((obj_start, obj_start))
    if patch_location in ["all", "box"]:
        patch_locations.append((box_id_start, box_id_start))
    if patch_location in ["all", "period"]:
        patch_locations.append((box_id_start+1, box_id_start+1))
    # if patch_location in ["all"]:
        # patch_locations.append((box_id_start + 2, box_id_start + 2))  # box
        # patch_locations.append((box_id_start + 3, box_id_start + 3))  # space
        # patch_locations.append((box_id_start + 4, box_id_start + 4))  # query id
        # patch_locations.append((box_id_start + 5, box_id_start + 5))  # contains
        # patch_locations.append((box_id_start + 6, box_id_start + 6))  # the

    return {"new_prompt": new_prompt, "changed_input_prompt": prompt, "labels": removed_objs, "patch_locations": patch_locations}


def generate_counterfactual_dcm_remove_across(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Ctf: Apple, orange in box 0, book, bag in box 1 ... (Remove bag from box 0). box 1 contains (book)
    Org: book, bag in box 1, apple, orange in box 0 ... (Remove apple from box 0). box 0 contains (orange)
    Patch target: apple (if idx)

    Args:
        prompt (str): the sentence that does not include target objects
        tokenizer (AutoTokenizer): model's tokenizer
        pos: (str): both, box, obj, which token position to patch

    Returns:
        new_prompt: the counterfactual prompt
        labels: list of expected target objects if hypothesis were to be true
        patch_locations: List of tuples (src, tgt) locations
    """
    assert prompt.count(". Remove the") == 1
    tokenizer = kwargs["tokenizer"]
    patch_location = kwargs.get("pos", "both")
    query_box_id = prompt[prompt.rfind("Box") + 4]
    description_phrases = prompt.split(". ")[0].split(", ")

    # shuffle description phrases
    new_description_phrases = description_phrases.copy()
    random.shuffle(new_description_phrases)

    # get new box id at the same box order id/index
    new_query_box_desc_phrase = new_description_phrases[int(query_box_id)]
    new_query_box_id = new_query_box_desc_phrase[new_query_box_desc_phrase.rfind("Box")+4]
    new_query_box_objs = get_objects(new_query_box_desc_phrase)

    # find clean removed obj index (within box) and remove the other indexed obj in new phrase
    desc_objs = get_objects(description_phrases[int(query_box_id)])
    removal_phrase = prompt.split(". ")[1]
    removed_obj = get_objects(removal_phrase)[0]
    removed_obj_idx = desc_objs.index(removed_obj)
    new_removed_obj_idx = 1 if removed_obj_idx == 0 else 0
    new_removed_obj = new_query_box_objs[new_removed_obj_idx]
    new_remove_phrase = f"Remove the {new_removed_obj} from Box {new_query_box_id}"
    new_query_phrase = f"Box {new_query_box_id} contains the"
    # new_query_phrase = f"Box {query_box_id} contains the"
    new_prompt = ", ".join(new_description_phrases)+f". {new_remove_phrase}. {new_query_phrase}"


    obj_start= len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the bag")) - 1
    box_id_start = len(tokenizer.encode(prompt[:prompt.find("Remove")] + f"Remove the bag from Box 0")) - 1
    patch_locations = []
    # if patch_location in ["all", "obj", "phrase"]:
    # patch_locations.append((obj_start -2, obj_start - 2))  # Remove
    # patch_locations.append((obj_start - 1, obj_start + 3))  # the
    # patch_locations.append((obj_start, obj_start))
    # patch_locations.append((obj_start + 1, obj_start + 1))  # from
    # patch_locations.append((obj_start + 2, obj_start + 2))  # Box
    # patch_locations.append((obj_start + 3, obj_start + 3))  # space

    # if patch_location in ["all", "box"]:
    #     patch_locations.append((box_id_start, box_id_start))
    # if patch_location in ["all", "period"]:
    #     patch_locations.append((box_id_start+1, box_id_start+1))
    if patch_location in ["all"]:
        # patch_locations.append((box_id_start + 2, box_id_start + 2))  # box
        # patch_locations.append((box_id_start + 3, box_id_start + 3))  # space
        # patch_locations.append((box_id_start + 4, box_id_start + 4))  # query id
        patch_locations.append((box_id_start + 5, box_id_start + 5))  # contains
        patch_locations.append((box_id_start + 6, box_id_start + 6))  # the

    # pdb.set_trace()
    return {"new_prompt": new_prompt, "labels": [removed_obj], "patch_locations": patch_locations}


## for now copied from entity-tracking-probing.src.utils, but need to merge it once we refactor the repos
def format_sentence(dat: Union[str, Dict[str, Any], List[int]], prompt_format: bool, prompt_prefix: Optional[str], chat_format: bool = False, tokenizer=None) -> str:
    if isinstance(dat, str):
        sent = dat
    elif isinstance(dat, list):
        sent = tokenizer.decode(dat, skip_special_tokens=True)
        pdb.set_trace()
    else:
        sent_field = "context" if "context" in dat else "prefix"
        sent = dat[sent_field]

    if prompt_format in ["PROMPT", "PROMPT_ALTFORM", "PROMPT_ALLBOX_ALTFORM", "INSTRUCTION", "default"]:
        # pdb.set_trace(header="formatting sentence with few-shot prompt")
        # just need to make sure if prompt_prefix is already in the example sentence
        # print(f"Formatting sentence with prompt_format {prompt_format} and prompt_prefix {prompt_prefix}")
        if prompt_prefix is None:
            # NOTE: this is only for llama70b, 2shot. The original PROMPT variable passed here are not altform, and the input dat already contains the full altform prompt.
            example_sent = sent
        else:
            example_sent = prompt_prefix + ". ".join(sent.split(". ")[:-1]) + ".\nStatement: " + sent.split(". ")[
                -1].removesuffix(" .")

    elif prompt_format:
        raise NotImplementedError()
    else:
        example_sent = sent.removesuffix(" .")

    if not chat_format:
        return example_sent

    assert prompt_format != False and tokenizer is not None
    instruction = example_sent.split("\n")[0]
    examples = []
    if "PROMPT" in prompt_format or prompt_format.startswith("INSTRUCTION"):  # 2 shots (no CoT)
        example_sents = example_sent.replace("\n\n", "\n").split("\n")
        curr_ex = {}
        for i, sent in enumerate(example_sents[1:]):
            if sent.startswith("Description"):
                curr_ex['input'] = sent
            elif sent.startswith("Statement"):
                curr_ex['output'] = sent
            if len(curr_ex) == 2:
                examples.append(curr_ex)
                curr_ex = {}

    # format with chat template
    messages = []
    if "llama" in tokenizer.name_or_path.lower() or "gemma" in tokenizer.name_or_path.lower():
        messages.append({"role": "system", "content": instruction})
    else:
        # for models that don't have system role
        messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": "Okay."})

    for example in examples:
        messages.append({"role": "user", "content": f"{example['input']}"})
        messages.append({"role": "assistant", "content": f"{example['output']}"})

    # messages = messages[:-1]  # last example is query
    prompt_string = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False, add_generation_prompt=True)
    # move end of turn for last turn and have model generate from that point on
    end_idx = prompt_string.rfind(examples[-1]['output']) + len(examples[-1]['output'])
    prompt_string = prompt_string[:end_idx]
    # pdb.set_trace()
    return prompt_string


def fix_fonts(title=20, label=20, xtick=15, ytick=15, default=15):
    # Set the global font family to 'Times New Roman'
    # keep running into
    plt.rc('font', family='serif', serif=['Times New Roman'])

    # Set the global default font size (e.g., to 14)
    plt.rcParams["font.size"] = default
    plt.rcParams["xtick.labelsize"] = xtick  # Optional: specific size for x-axis ticks
    plt.rcParams["ytick.labelsize"] = ytick  # Optional: specific size for y-axis ticks
    plt.rcParams["axes.labelsize"] = label  # Optional: specific size for axis labels
    plt.rcParams["axes.titlesize"] = title  # Optional: specific size for plot titles


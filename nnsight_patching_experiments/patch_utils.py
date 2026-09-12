import os
import sys
import json
import argparse
import pickle
from typing import Callable

import numpy as np

from transformers import BitsAndBytesConfig
import nnsight
from nnsight import LanguageModel, CONFIG

sys.path.append("..")
# from utils import get_model_and_tokenizer, load_dataloader, get_random_guess_baseline, fix_random_seed, str_to_bool, \
#     find_previous_query_box_pos, is_int_with_negatives, stupid_pad, PROMPT_ALTFORM, setup_nnsight
from utils import get_model_and_tokenizer, load_dataloader, get_random_guess_baseline, fix_random_seed, str_to_bool, \
    find_previous_query_box_pos, is_int_with_negatives, PROMPT_ALTFORM, setup_nnsight


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='experiment seed', type=int, default=42)
    parser.add_argument('--model', help='hf model name', type=str, default="luodian/llama-7b-hf")
    parser.add_argument('--load_in_8bit', help='load in 8bit or not', action='store_true')
    parser.add_argument('--load_in_4bit', help='load in 4bit or not', action='store_true')
    parser.add_argument('--remote', help='use NDIF remote to run nnsight code (necessary for big models)', action='store_true')
    parser.add_argument('--load_graph', help='if eval only, path to load the graph from', type=str, default=None)

    # data specific arguments
    parser.add_argument('-d', '--datafile', help='dataset file', type=str)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=1)
    parser.add_argument('-n', '--num_samples', help='number of samples', type=int, default=8)
    parser.add_argument('-s', '--ops_order', help='comma separated sequence of operations', type=str, default=None)
    parser.add_argument('--query_ops_order', help='comma separated sequence of operations (applied to query box only)',
                        type=str, default=None)
    parser.add_argument('--success_filter', help='whether to only consider successful/unsuccessful prompts',
                        default=None, type=str_to_bool)
    parser.add_argument('--operations_on_same_obj', help='whether operations should be applied to same object',
                        default=None, type=str_to_bool)
    parser.add_argument('--copy_filter',
                        help='whether to only consider prompts that cannot be solved by copy mechanism (where the previous mention of the query box state has the same first item as first label item)',
                        default=None, type=str_to_bool)
    parser.add_argument('--output_dir', help='output directory', type=str, default='../outputs/top_attentions')
    parser.add_argument('--max_initial_objects_per_box', help='max number of objects in any box in initial state',
                        type=int, default=None)
    parser.add_argument('--counterfactual_format', help='what kind of counterfactuals do we use',
                        # choices=["rand_obj", "rand_query_id", "rand_box_id", "rand_obj_rand_query_id",
                        #          "rand_obj_rand_box_id", "dcm_obj", "dcm_pos_ctf_op", "dcm_pos_og_op", "dcm_pos_legal_op"],
                        type=str, default='rand_obj_rand_query_id')
    parser.add_argument('--data_field', help='data field containing sentences we want to do investigation on', type=str,
                        default='sentence')
    parser.add_argument('--token_step',
                        help='token step on which we interpret the logit for. {pred, exp_1, exp_2, etc}', type=str,
                        default='pred')
    parser.add_argument('--num_query_object', help='filter on number of query objects', type=int, default=None)
    parser.add_argument('--sort_query_objects', help='sort label query objects by order in which they appear in the prompt',  action="store_true")
    parser.add_argument('--overwrite_cache', help='whether to recompute results', action="store_true")
    parser.add_argument('--prompt_format', help='Whether to use prompt and if so what prompt format', type=str, default=False, choices=[False, "INSTRUCTION", "PROMPT_ALTFORM", "PROMPT_ALLBOX_ALTFORM"])

    # circuit specifications TODO
    parser.add_argument('--top_p', help='top_p percentile of heads. Either set this or number of group of heads, but this is easier to justify.', type=float, default=None)
    parser.add_argument('--n_groupA', help='number of group A heads', type=int, default=7)
    parser.add_argument('--n_groupB', help='number of group B heads', type=int, default=10)
    parser.add_argument('--n_groupC', help='number of group C heads', type=int, default=6)
    parser.add_argument('--n_groupD', help='number of group D heads', type=int, default=5)

    # patching objective related args
    parser.add_argument('--score_source', help='Whether to use "logit","prob", or "logp" as the patching score', type=str, default='prob', choices=["logit", "prob", "logp"])
    parser.add_argument('--use_object_index', help='Which index(es) the target to calculate patching score for. Default uses all target objects', type=str, default=None)

    return parser


def post_arg_parse_fix(args):
    # "" is when we want specifically data with no operations, None (default) is no filter
    args.ops_order = None if args.ops_order is None else tuple() if args.ops_order == "" else tuple(
        args.ops_order.split(","))
    args.query_ops_order = None if args.ops_order is None else tuple() if args.query_ops_order == "" else tuple(
        args.query_ops_order.split(","))
    if args.use_object_index is not None:
        if is_int_with_negatives(args.use_object_index):
            args.use_object_index = [int(args.use_object_index)]
        elif ":" in args.use_object_index:
            assert not args.use_object_index.endswith(":"), "slicing can be of form 'a:b' or `:b`."
            args.use_object_index = slice(int(args.use_object_index[1:])) if args.use_object_index.startswith(":") else slice(*[int(a) for a in args.use_object_index.split(":")])
    if args.top_p is not None:
        args.n_groupA, args.n_groupB, args.n_groupC, args.n_groupD = None, None, None, None


def maybe_patch_or_load_cache(result_cache_path: str, patch_func: Callable, **kwargs):
    if os.path.exists(result_cache_path) and ("args" in kwargs and not kwargs["args"].overwrite_cache):
        print(f"Loading cached results from {result_cache_path}")
        if result_cache_path.endswith(".pkl"):
            with open(result_cache_path, "rb") as f:
                result = pickle.load(f)
        elif result_cache_path.endswith(".npy"):
            result = np.load(result_cache_path)
        elif result_cache_path.endswith(".json"):
            result = json.load(open(result_cache_path))
        else:
            raise NotImplementedError(f"result_cache_path {result_cache_path} type not supported")
        return result
    else:
        result = patch_func(**kwargs)
        if result_cache_path.endswith(".pkl"):
            with open(result_cache_path, "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        elif result_cache_path.endswith(".npy"):
            np.save(result_cache_path, result)
        elif result_cache_path.endswith(".json"):
            json.dump(result, open(result_cache_path, "w"), indent=2)
        else:
            raise NotImplementedError(f"result_cache_path {result_cache_path} type not supported")
        return result


def get_model_and_dataset(args):
    qcfg = None
    if args.load_in_8bit:
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        qcfg = BitsAndBytesConfig(load_in_4bit=True)
    model = LanguageModel(args.model, device_map="auto", dispatch=True, quantization_config = qcfg)
    model.tokenizer.padding_side = "right"
    if any([t in args.model for t in ["gemma", "Llama-3.", "santacoder"]]):
        prepend_space_to_answer = True
    else:
        prepend_space_to_answer = False
    print("MODEL LOADED")
    # load dataset
    dataloader, dataset = load_dataloader(
        model=model,
        tokenizer=model.tokenizer,
        datafile=args.datafile,
        num_samples=args.num_samples,
        num_boxes=7,  # args.num_boxes,
        ops_order=args.ops_order,
        query_ops_order=args.query_ops_order,
        success_filter=args.success_filter,
        operations_on_same_obj=args.operations_on_same_obj,
        copy_filter=args.copy_filter,
        batch_size=args.batch_size,
        return_dataset=True,
        max_initial_objects_per_box=args.max_initial_objects_per_box,
        counterfactual_format=args.counterfactual_format,
        data_field=args.data_field,
        token_step=args.token_step,
        prepend_space_to_answer=prepend_space_to_answer,
        num_query_object=args.num_query_object,
        sort_query_objects=args.sort_query_objects,
        seed=args.seed,
        prompt_format=args.prompt_format,
        remote=args.remote,
    )
    print(f"DATALOADER CREATED ({len(dataset)=})")
    print(f"max data length: {len(dataset['base_tokens'][0])}")
    return dataloader, dataset, model
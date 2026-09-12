import argparse
import csv
import json
import os
import pdb
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

import torch
import numpy as np
import pandas as pd

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.dataset import *
from src.probing_utils import format_sentence, get_quantization_config

import nnsight
import anyio

from nnsight import LanguageModel
import sys

MAX_NEW_TOKENS = 50
MAX_REMOTE_ATTEMPTS = 10

def find_last_nth_index(value, lst, n):
    matches = [(i, val) for i, val in enumerate(reversed(lst)) if val == value]
    if len(matches) >= n:
        return len(lst) - matches[n - 1][0] - 1  # Get index from last nth match
    else:
        return -1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to the entity tracking model.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model with 8-bit")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model with 4-bit")

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to a jsonl file of data.",
        required=True,
    )
    parser.add_argument(
        "--object_vocabulary_file",
        type=str,
        default="data/objects_with_bnc_frequency.csv",
        help='Path to a .csv file with a string field "object_names".'
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=22,
        help="Seed for random sampling."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/baseline_inference"
    )
    parser.add_argument(
        "--few_shot_prompt",
        type=str,
        default=False,
        choices=[False, "PROMPT", "PROMPT_ALTFORM", "PROMPT_ALLBOX_ALTFORM", "INSTRUCTION", "PROMPT_ALTFORM_SINGULAR", "INSTRUCTION_SINGULAR"]
    )
    parser.add_argument(
        "--chat", action="store_true",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--use_remote",
        action="store_true",
        help="Whether to use remote model.",
    )
        
    # distributed inference ( for caching embedding)
    parser.add_argument('--distributed',
                        dest="distributed",
                        action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int)

    args = parser.parse_args()

    # load object map
    object_map = {}
    object_list = []
    with open(args.object_vocabulary_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            object_map[row["object_name"]] = i
            object_list.append(row["object_name"])

    # set up distributed inference
    if args.distributed:
        rank = int(os.environ.get("LOCAL_RANK",0))
        print(f"{rank=}")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)  # https://github.com/pytorch/pytorch/issues/146767
        torch.distributed.init_process_group("nccl", device_id=device)
        tp_plan = "auto"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tp_plan = None
    model_kwargs = {"tp_plan": tp_plan}

    model_name = args.model_dir
    if args.distributed:
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif args.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["quantization_config"] =  get_quantization_config(args)
    if model_kwargs.get("quantization_config") is None and transformers.utils.is_torch_bf16_gpu_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
    from src.probing_utils import VLM_REGISTRY
    vlm_spec = next((v for k, v in VLM_REGISTRY.items() if k.lower() in args.model_dir.lower()), None)
    if args.use_remote:
        model = LanguageModel(args.model_dir, device_map="auto")
    elif vlm_spec is not None:
        # VLM checkpoint: text-only generation; loader/trust_remote_code per registry
        if vlm_spec["loader"] == "causal_trc":
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, **model_kwargs)
        else:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_dir, trust_remote_code=vlm_spec["trc"], **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    if model_kwargs.get("quantization_config") is None and not args.distributed and not args.use_remote:
        model = model.to(device)
    # InternVL's wrapper generate is broken text-only (img_context assert, completion-only
    # return); its language_model is a full CausalLM. Others must NOT be unwrapped (no lm_head).
    gen_model = model.language_model if (vlm_spec is not None and vlm_spec["unwrap_for_generate"]) else model

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=bool(vlm_spec and vlm_spec["trc"]))
    tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"inference_{os.path.basename(os.path.normpath(args.model_dir))}"
                                             f"{'_8bit' if args.load_in_8bit else '_4bit' if args.load_in_4bit else ''}"
                                             f"{'_fs'+args.few_shot_prompt if args.few_shot_prompt else ''}"
                                             f"{'_chat' if args.chat else ''}"
                                             f"_{os.path.basename(os.path.normpath(args.data_path))}")
    print(f"Writing output to {out_path}")

    # Data
    sampled_data = pd.read_json(args.data_path, orient='records', lines=True).to_dict("records")
    # read cached data from out_path
    if os.path.exists(out_path):
        with open(out_path, 'r') as rf:
            existing_data = [json.loads(line) for line in rf]
    n_cached = len(existing_data) if os.path.exists(out_path) else 0
    n_cached_batches = n_cached // args.batch_size
    assert n_cached % args.batch_size == 0, "Cached data is not a multiple of batch size"
    # TODO batch generation
    for bi in tqdm(range(0, len(sampled_data), args.batch_size)):
        batch_data = sampled_data[bi:bi+args.batch_size]
        batch_sent = []
        if bi // args.batch_size < n_cached_batches:
            print(f"Skipping batch {bi//args.batch_size} because it's already cached")
            continue
        # pdb.set_trace(header="entering batch processing")
        for dat in batch_data:
            target = dat.get("gold", dat["masked_content"])
            orig_items = [] if target == "nothing" else target.removeprefix("the ").split(" and the ")
            dat["orig_items"] = orig_items
            example_sent = format_sentence(dat, args.few_shot_prompt, globals().get(args.few_shot_prompt), chat_format=args.chat, tokenizer=tokenizer)
            batch_sent.append(example_sent)
        # pdb.set_trace(header="rdy to submit")
        batch_input = tokenizer(batch_sent, return_tensors="pt", padding=True, padding_side="left").to(device)
        with torch.no_grad():
            if not args.use_remote:
                output = gen_model.generate(
                    **batch_input,
                    max_new_tokens=50,
                    stop_strings=[".", "\n", "Box"],
                    tokenizer=tokenizer,
                    num_beams=1,
                    do_sample=False,  # greedy decoding
                )
                generation = tokenizer.batch_decode(output[:, batch_input.input_ids.shape[1]:], skip_special_tokens=True)
            else:
                # with model.generate(**batch_input, max_new_tokens=50, stop_strings=[".", "\n", "Box"], num_beams=1, do_sample=False, remote=True) as gen:
                # input_ids = batch_input["input_ids"]
                # attention_mask = batch_input["attention_mask"]
                # NOTE not sure why ndif raise errors when passing tokenized inputs directly, so use raw text for now and let ndif side do tokenization
                # Added retry logic to solve occasional connection issues
                for n_attempt in range(MAX_REMOTE_ATTEMPTS):
                    try:
                        with model.generate(batch_sent, max_new_tokens=50, num_beams=1, do_sample=False,remote=True) as gen:
                            # not sure ndif takes these args, but leave it for now
                            # output = model.generator.output.save()
                            output = model.generator.output.save()
                        break
                    except Exception as e:
                        print(f"Remote generation attempt {n_attempt+1} failed with error: {e}")
                        if n_attempt == MAX_REMOTE_ATTEMPTS - 1:
                            raise e
                        else:
                            print("Retrying...")
                # Cant use stop strings with ndif s  o I'll just truncate at first period
                generation = []
                for out in output:
                    # pdb.set_trace(header="checking raw output from remote generation")
                    
                    generated_tokens = out[-50:]  # skip input prompt
                    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    stop_words = [".", "\n", "Box"]
                    # find first stop_words in the seq.
                    pattern = re.compile("|".join(map(re.escape, stop_words)))
                    match = pattern.search(decoded_text)
                    decoded_text = decoded_text[:match.start()] if match else decoded_text
                    
                    generation.append(decoded_text)
                    
        
        for i, dat in enumerate(batch_data):
            pred_objs = list(set([o for o in generation[i].lower().replace(","," ").replace("."," ").split(" ") if o in object_map]))
            write_d = {
                'prefix': batch_sent[i],
                'original_answer': generation[i],
                'parsed_original_answer': pred_objs,
                'gold_items': dat['orig_items'],
                'gold_answer': dat["masked_content"],
                'numops': dat["numops"],
                'numops_global': dat["prefix"].count(". ")-1,
                'correct': set(pred_objs) == set(dat['orig_items']),
                'precision': np.mean([o in dat['orig_items'] for o in pred_objs]).item(),
                'recall': np.mean([o in pred_objs for o in dat['orig_items']]).item(),
            }
            with open(out_path, 'a') as wf:
                wf.write(json.dumps(write_d) + "\n")

    # for dat in tqdm(sampled_data):
    #     target = dat.get("gold", dat["masked_content"])
    #     orig_items = [] if target == "nothing" else target.removeprefix("the ").split(" and the ")
    #     dat["orig_items"] = orig_items
    #     example_sent = format_sentence(args, dat)
    #     input_example = tokenizer(example_sent, return_tensors="pt").to(device)
    #
    #     with torch.no_grad():
    #         output = model.generate(
    #             **input_example,
    #             max_new_tokens=50,
    #             stop_strings=[".", "\n", "Box"],
    #             tokenizer=tokenizer,
    #             num_beams=1,
    #             do_sample=False,  # greedy decoding
    #         )
    #     generation = tokenizer.batch_decode(output[:, input_example.input_ids.shape[1]:])[0]
    #
    #     # print(generation)
    #     # print()
    #     pred_objs = list(set([o for o in generation.lower().replace(","," ").replace("."," ").split(" ") if o in object_map]))
    #     write_d = {
    #         'prefix': example_sent,
    #         'original_answer': generation,
    #         'parsed_original_answer': pred_objs,
    #         'gold_items': dat['orig_items'],
    #         'gold_answer': dat["masked_content"],
    #         'numops': dat["numops"],
    #         'numops_global': example_sent.count(". ")-1,
    #         'correct': set(pred_objs) == set(dat['orig_items']),
    #         'precision': np.mean([o in dat['orig_items'] for o in pred_objs]).item(),
    #         'recall': np.mean([o in pred_objs for o in dat['orig_items']]).item(),
    #     }
    #     with open(out_path, 'a') as wf:
    #         wf.write(json.dumps(write_d) + "\n")




if __name__ == '__main__':
    main()
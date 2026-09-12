import csv
import pdb
from typing import List, Dict, Tuple, Any, Optional, Union
import regex as re
import random

import numpy as np
import torch
from transformers import BitsAndBytesConfig, set_seed

# VLMs supported text-only (probing + behavioral inference).
#   trc: repo requires trust_remote_code
#   loader: "imagetext" = AutoModelForImageTextToText, "causal_trc" = AutoModelForCausalLM(+trc)
#   unwrap: use .language_model for hidden-state caching (top-level forward needs pixel_values)
#   unwrap_for_generate: use .language_model.generate (only InternVL — its wrapper generate
#     asserts img_context_token_id and returns completion-only tokens; its language_model is a
#     full Qwen3ForCausalLM. Molmo2/LLaVA language_model lack lm_head — never unwrap for generate.)
VLM_REGISTRY = {
    "Qwen3-VL-8B-Instruct": {"trc": False, "loader": "imagetext", "unwrap": False, "unwrap_for_generate": False},
    "InternVL3_5-8B": {"trc": True, "loader": "causal_trc", "unwrap": True, "unwrap_for_generate": True},
    "Molmo2-8B": {"trc": True, "loader": "imagetext", "unwrap": True, "unwrap_for_generate": False},
    "llava-onevision-qwen2-7b-ov-hf": {"trc": False, "loader": "imagetext", "unwrap": True, "unwrap_for_generate": False},
}

NON_OBJ_WORDS={
    "put", "remove", "move",
    "contains", "the", "nothing",
    # "container",
    "are", "from", "into", "and",  "is", "in", "box", "to",
    ",", ".", ".\n", "\n",
    "<bos>","</bos>",  "<s>", "</s>", '', "<|begin_of_text|>", #  chat template special tokens too
    "Description", ":", "Statement", "Given", "description", "after", '"', "write", "a", "true", "statement", "about", "all", "boxes", "their", "contents", "according", "description", # from prompts
    "query", "its", # TODO from CoT
    "<|endoftext|>", "<|end_of_text|>", # # ADDED eos token since it's also the padding token
    *[str(i) for i in range(10)]
}

BOX_IDS = set([str(i) for i in range(10)])
OBJECT_END_TOKENS = set()

def find_all(text, substring):
    positions = []
    start_index = 0
    while True:
        try:
            # Find the next occurrence starting from start_index
            index = text.index(substring, start_index)
            positions.append(index)
            # Update start_index to search after the current match's start
            start_index = index + 1
        except ValueError:
            # Break the loop when no more occurrences are found
            break
    return positions


def get_objects(sentence: str):
    return re.findall(r'the ([^ ,.]+) ', sentence.lower())

def get_box_ids(sentence: str):
    return re.findall(r'Box (\d+)', sentence)

def is_object(word: str) -> bool:
    return word.strip().lower() not in NON_OBJ_WORDS

def is_box_id(word: str) -> bool:
    return word.strip().lower() in BOX_IDS


def get_token_pos_given_span_types(input_ids: torch.Tensor, tokenizer, span_type: str, objects:List[str]=None) -> List[int]:
    global OBJECT_END_TOKENS
    if len(OBJECT_END_TOKENS)==0 and objects is not None:
        print("pre-computing object end tokens")
        OBJECT_END_TOKENS.update(tokenizer.encode(f" {o}")[-1] for o in objects)

    sent_str = tokenizer.decode(input_ids, skip_special_tokens=True) # dont ignore special tokens to be compatible with padding tokens
    # if sentence has prompt attached, need to disregard tokens in shots and prompt
    if "\n\n" in sent_str:
        # find the last occurring 'Description:', and only count occurrences from there
        prompt_str = sent_str[:sent_str.rfind("Description:")+len("Description:")] # Decsription: is tokenized to two tokens
        
        start_idx = len(tokenizer.encode(prompt_str))
    else:
        start_idx = 0
        
    # super patchy fix: add padded length
    if input_ids[0] == 128001:
        pad_len = (input_ids==128001).sum().item()
        start_idx += pad_len    

    decoded = [tokenizer.decode(t) for t in input_ids]
    indices = []
    for i, token_str in enumerate(decoded):
        if i < start_idx:
            continue

        if "object" in span_type and token_str.strip().lower() not in NON_OBJ_WORDS:
            if "llama" in tokenizer.name_or_path.lower() or "gpt2" in tokenizer.name_or_path.lower():  # default behavior, expecting llama
                indices.append(i)
            else:  # for other tokenizers, objects maybe split into multiple tokens, so check if it's the last token of
                   # of any objects
                assert objects is not None
                if input_ids[i].cpu().item() in OBJECT_END_TOKENS:
                    indices.append(i)

        if "number" in span_type and token_str.strip().lower() in BOX_IDS:
            indices.append(i)
        if "period" in span_type and token_str.strip().lower() == ".":
            indices.append(i)
        if "comma" in span_type and token_str.strip().lower() == ",":
            indices.append(i)

    return indices


def format_sentence(dat: Union[str,Dict[str, Any], List[int]], prompt_format:bool, prompt_prefix:Optional[str], chat_format:bool=False, tokenizer=None) -> str:
    if isinstance(dat, str):
        sent = dat
    elif isinstance(dat, list):
        sent = tokenizer.decode(dat, skip_special_tokens=True)
        pdb.set_trace()
    else:
        sent_field = "context" if "context" in dat else "prefix"
        sent = dat[sent_field]

    if prompt_format in ["PROMPT", "PROMPT_ALTFORM","PROMPT_ALLBOX_ALTFORM", "INSTRUCTION", "PROMPT_ALTFORM_SINGULAR", "INSTRUCTION_SINGULAR", "default"]:
        # pdb.set_trace(header="formatting sentence with few-shot prompt")
        # just need to make sure if prompt_prefix is already in the example sentence
        # print(f"Formatting sentence with prompt_format {prompt_format} and prompt_prefix {prompt_prefix}")
        if prompt_prefix is None:
            # NOTE: this is only for llama70b, 2shot. The original PROMPT variable passed here are not altform, and the input dat already contains the full altform prompt.
            example_sent = sent
        else:
            example_sent = prompt_prefix + ". ".join(sent.split(". ")[:-1]) + ".\nStatement: " + sent.split(". ")[-1].removesuffix(" .")

    elif prompt_format:
        raise NotImplementedError()
    else:
        example_sent = sent.removesuffix(" .")

    if not chat_format:

        # print(f"Example sentence after formatting: {example_sent}")
        return example_sent

    assert prompt_format!=False and tokenizer is not None
    instruction = example_sent.split("\n")[0]
    examples = []
    if "PROMPT" in prompt_format or prompt_format.startswith("INSTRUCTION"): # 2 shots (no CoT)
        example_sents = example_sent.replace("\n\n","\n").split("\n")
        curr_ex = {}
        for i, sent in enumerate(example_sents[1:]):
            if sent.startswith("Description"):
                curr_ex['input'] = sent
            elif sent.startswith("Statement"):
                curr_ex['output'] = sent
            if len(curr_ex)==2:
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



def get_quantization_config(args):
    q_config = None
    if args.load_in_8bit:
        q_config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        q_config = BitsAndBytesConfig(load_in_4bit=True)
    return q_config


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

def get_object_mapping(object_vocabulary_file: str):
    object_map = {}
    object_list = []
    with open(object_vocabulary_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            object_map[row["object_name"]] = i
            object_list.append(row["object_name"])
    return object_map, object_list

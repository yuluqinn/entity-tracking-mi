import argparse
import os
import re
import pdb
import regex as re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable, Union

import h5py
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import json
from torch.utils.data import Dataset, DataLoader
import datasets
from .probing_utils import get_objects, get_box_ids, is_object, is_box_id, get_token_pos_given_span_types #format_sentence

import sys
sys.path.append(".")
# check all pathes
print("Current working directory:", os.getcwd())
print("Current paths:", sys.path)
from utils import format_sentence, PROMPT, PROMPT_ALTFORM, PROMPT_ALLBOX_ALTFORM, INSTRUCTION
from .state_evals import generate_state_matrix, detect_local_removals, detect_removals

_GPT_MAX_LENGTH = 512
NUM_BOXES = 7


def has_row_level_labels(raw_examples):
    """True for row-level-filtered files, whose rows carry a precomputed `global_state`
    (see scratch/correctonly_merged/build_dataset.py). Such files need not come in 7-row blocks."""
    return bool(raw_examples) and "global_state" in raw_examples[0]


def findall(s: str, substring: str) -> List[int]:
    return [m.start() for m in re.finditer(substring, s)]


def token_strs_to_str(token_strs: List[str], tokenizer) -> str:
    if "gemma" in tokenizer.name_or_path.lower() or "llama" in tokenizer.name_or_path.lower() or "gpt2" in tokenizer.name_or_path.lower():
        tokens = []
        for token_str in token_strs:
            token = tokenizer.encode(token_str, add_special_tokens=False)
            tokens.extend(token)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    # elif "llama" in tokenizer.name_or_path.lower():
    #     try:
    #         decoded = tokenizer.decode([tokenizer.encode(t, add_special_tokens=False)[0] for t in token_strs], skip_special_tokens=True)
    #     except Exception:
    #         pdb.set_trace()
    else:
        raise NotImplementedError("Double check tokenizer type manually!")
    return decoded


def str_to_token_strs(sentence: str, tokenizer) -> List[str]:
    return [tokenizer.decode(t) for t in tokenizer.encode(sentence)]


class LMDataloader(Dataset):
    """Loads LM (T5 denoising) dataset with masked input."""

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_field, target_field, include_empty=True, min_prev_objects=-1):
        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        if self.include_empty and self.min_prev_objects < 1:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe[target_field].str.contains("nothing") | dataframe[target_field].str.contains("is empty")
            self.data = dataframe[-f]
            if self.min_prev_objects > 0:
                f = dataframe[target_field].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]
            self.data = self.data.reset_index()

        self.source_len = source_len
        self.target_len = target_len
        self.source_text = self.data[source_field]
        self.target_text = self.data[target_field]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # Cleaning data so as to ensure data is in string type
        source_text = source_text.split()
        target_text = target_text.split()
        

        source = self.tokenizer.batch_encode_plus(
            [source_text], max_length=self.source_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.target_len, pad_to_max_length=True,
            is_split_into_words=True, padding="max_length", return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


class GPTDataloaderForInference(Dataset):
    """Loads LM dataset for inference."""

    def __init__(self, dataframe, tokenizer, max_length=_GPT_MAX_LENGTH, include_empty=True, condition_on="number", min_prev_objects=-1, include_prompt:Union[bool,str]=False, return_span=False, args=None):
        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.condition_on = condition_on
        self.return_span = return_span

        if self.include_empty:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe["masked_content"].str.contains("nothing") | dataframe["masked_content"].str.contains("is empty")
            self.data = dataframe[-f]
        
            if self.min_prev_objects > 0:
                f = dataframe["masked_content"].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]

            self.data = self.data.reset_index()


        # optional alternate model-input column (e.g. chat-template-wrapped prompts);
        # label parsing elsewhere keeps reading the canonical prefix/sentence fields
        cache_field = getattr(args, "cache_prefix_field", None) if args is not None else None
        prefix_src = self.data[cache_field] if cache_field and cache_field in self.data.columns else self.data["prefix"]
        self.prefix_text = prefix_src
        self.target_text = self.data["sentence"]
        self.max_length = max_length

        if self.min_prev_objects > 0:
            self.prefix_text = self.data.apply(lambda x: x["prefix"] + " " + " and ".join(x["masked_content"].split(" and ")[0:self.min_prev_objects]) + " and the", axis=1)

        elif self.condition_on == "contains":  # najoung's original data seems to not have "contains", but our data does
            # add " contains" to prefix
            self.prefix_text = prefix_src.apply(lambda x: x + " contains" if not x.endswith("contains") else x)
        elif self.condition_on == "the":
            # add " contains the" to prefix
            self.prefix_text = prefix_src.apply(lambda x: x + " contains the" if not x.endswith("contains") else x + " the")

        if isinstance(include_prompt, str):
            # self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            # self.target_text = self.target_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.apply(lambda x: format_sentence(x, prompt_format=include_prompt, prompt_prefix=globals()[include_prompt], chat_format=args.chat, tokenizer=self.tokenizer))
            self.target_text = self.target_text.apply(lambda x: format_sentence(x, prompt_format=include_prompt, prompt_prefix=globals()[include_prompt], chat_format=args.chat, tokenizer=self.tokenizer))

            print(self.prefix_text[0])
            print(self.prefix_text[1])
            print("---------")
            print(self.target_text[1])

        if return_span:
            self.add_phrase_spans()
            self.span = self.data["phrase_spans"]
            self.span_tokens = self.data["phrase_span_tokens"]
            self.local_op_span_indices = self.data["local_op_span_indices"]

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "right"
        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus([target_text], max_length=self.max_length, truncation=True, return_tensors='pt')

        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus([prefix_text], max_length=self.max_length, truncation=True, return_tensors='pt')

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()

        output =  {
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
        }

        if self.return_span:
            output["span"] = self.span[index]
            output["span_tokens"] = self.span_tokens[index]
            output["numops"] = self.data["numops"][index]
            output["local_op_span_indices"] = self.local_op_span_indices[index]
        return output
    
    def get_collate_fn(self):
        """
        Returns a collate function to be used in DataLoader, original impelmentation does not support batch size > 1;
        Need to use left padding here
        """
        def collate_fn(batch):
            padding_side = 'left' # right-align for the conveinence of auto-regressive generation, and indexing
            padding_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            prefix_ids = [item['prefix_ids'] for item in batch]
            prefix_attn_masks = [item['prefix_attn_masks'] for item in batch]
            target_ids = [item['target_ids'] for item in batch]
            padded_prefix_ids = torch.nn.utils.rnn.pad_sequence(prefix_ids, batch_first=True, padding_value=padding_value, padding_side=padding_side)
            padded_prefix_attn_masks = torch.nn.utils.rnn.pad_sequence(prefix_attn_masks, batch_first=True, padding_value=0, padding_side=padding_side)
            padded_target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=padding_value, padding_side=padding_side)
            start_of_effective_token = []
            # start of effective token
            for i in range(len(batch)): # assume left padding though
                assert padded_prefix_ids[i][-prefix_ids[i].shape[0]:].equal(prefix_ids[i])
                assert padded_target_ids[i][-target_ids[i].shape[0]:].equal(target_ids[i])
                start_idx = padded_prefix_ids.shape[1] - prefix_ids[i].shape[0]
                assert padded_prefix_attn_masks[i][-prefix_attn_masks[i].shape[0]:].equal(prefix_attn_masks[i])
                start_of_effective_token.append(start_idx)
                
                
            
            output = {
                'prefix_ids': padded_prefix_ids.to(dtype=torch.long),
                'prefix_attn_masks': padded_prefix_attn_masks.to(dtype=torch.long),
                'target_ids': padded_target_ids.to(dtype=torch.long),
                'start_of_effective_token': torch.tensor(start_of_effective_token, dtype=torch.long),
                'padding_side': padding_side
            }
            if self.return_span:
                # TODO need to check whether padding will affect span indices
                output["span"] = [item['span'] for item in batch]
                output["span_tokens"] = [item['span_tokens'] for item in batch]
                output["numops"] = [item['numops'] for item in batch]
                output["local_op_span_indices"] = [item['local_op_span_indices'] for item in batch]
            return output
        return collate_fn
            

    def add_phrase_spans(self):
        """
        for each example, compute span for each phrase, which include 7 description phrases,
        operation phrases, and query phrase.
        """
        def compute_spans(sentence:str) -> str:
            sentence_tokens = self.tokenizer.encode(sentence)
            spans = []
            # initial description phrases
            init_desc = sentence.split(". ")[0] + "."
            init_desc_tokens = self.tokenizer.encode(init_desc)
            # codellama for example use different token for ',' and 'key,'.
            period_token = self.tokenizer.encode("key.", add_special_tokens=False)[-1]
            comma_token = self.tokenizer.encode("key,", add_special_tokens=False)[-1]
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
            op_phrases_tokens = self.tokenizer.encode(op_phrases)
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


        def compute_phrase_span_tokens(row) -> List[str]:
            tokens = []
            s = self.tokenizer.encode(row["sentence"])
            for i in range(NUM_BOXES):
                tokens.append(f"DESC_{i}: {self.tokenizer.decode(s[row.phrase_spans[i][0]:row.phrase_spans[i][1]+1], skip_special_tokens=True)}")
            for i in range(row.sentence.strip(".").count(". ")-1):
                tokens.append(f"OP_{i}: {self.tokenizer.decode(s[row.phrase_spans[i+NUM_BOXES][0]:row.phrase_spans[i+NUM_BOXES][1]+1], skip_special_tokens=True)}")
            tokens.append(f"QUERY: {self.tokenizer.decode(s[row.phrase_spans[-1][0]:row.phrase_spans[-1][1]+1], skip_special_tokens=True)}")
            return tokens


        self.data["phrase_spans"] = self.data.sentence.apply(compute_spans)
        self.data["phrase_span_tokens"] = self.data.apply(compute_phrase_span_tokens, 1)

        def compute_local_op_span_indices(row) -> List[int]:
            op_phrases = row["sentence"].strip(".").split(". ")[1:-1]
            query_box = row["sentence"][row["sentence"].rfind("Box")+4]
            indices = []
            for i, op_phrase in enumerate(op_phrases):
                if f"Box {query_box}" in op_phrase:
                    indices.append(NUM_BOXES+i)
            return indices
        self.data["local_op_span_indices"] = self.data.apply(compute_local_op_span_indices, 1)


class ProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map, max_data=None, box_label_base=0):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices.
            box_label_base (int): lowest box number in the dataset (0 original, 1 vlm-data).
        """
        self.box_label_base = box_label_base
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.examples, self.num_ops, counts, self.mentioned_objects = self.load_examples(path_to_data, max_examples=max_data)

        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        
        # per-row files (rows carry a precomputed `global_state`) have one label per row;
        # the original block files have one label per 7 consecutive rows
        assert len(self.activations) == (len(self.examples) if self.per_row else NUM_BOXES * len(self.examples))

        self.n = len(self.activations)

    def get_weights(self):
        return self.weights

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        k = index if self.per_row else index // NUM_BOXES
        return self.activations[index], self.examples[k], torch.tensor(self.num_ops[k]).to(torch.long),  self.mentioned_objects[k]

    def load_examples(self, path_to_data, max_examples=None):

        raw_examples = []

        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))

        # Row-level-filtered files: every row carries `global_state` ({object: box_no}, the full
        # state at its timestep, precomputed while the 7-row block was still intact), so the
        # label no longer has to be assembled from sibling rows and blocks need not be complete.
        self.per_row = has_row_level_labels(raw_examples)
        if self.per_row:
            counts = np.zeros((NUM_BOXES + 1))
            examples, num_ops, all_mentioned_objects = [], [], []
            for ex in raw_examples:
                s_parts = ex["sentence"].strip(".").split(".")
                box_contents = torch.zeros(self.n_objects)
                for obj, box_no in ex["global_state"].items():
                    box_contents[self.oti[obj]] = box_no + 1 - self.box_label_base
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(NUM_BOXES + 1)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                mentioned_objects = torch.zeros(self.n_objects)
                for o in re.findall(r'the ([^ ,.]+)[ ,.]', " ".join(s_parts[:-1]) + " "):
                    mentioned_objects[self.oti[o]] = 1
                all_mentioned_objects.append(mentioned_objects)
                if max_examples is not None and len(examples) == max_examples:
                    break
            return examples, num_ops, counts, all_mentioned_objects

        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"

        counts = np.zeros((NUM_BOXES + 1))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            box_no = int(s[4]) # 4th character is the box number
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    oidx = self.oti[c]
                    box_contents[oidx] = box_no + 1 - self.box_label_base
            
            if (i % NUM_BOXES) == (NUM_BOXES - 1):
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(NUM_BOXES + 1)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                o_names = re.findall(r'the ([^ ,.]+)[ ,.]', " ".join(s_parts[:-1]) + " ")
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)

            if max_examples is not None and len(examples) == max_examples:
                break
        
        return examples, num_ops, counts, all_mentioned_objects

class BinaryProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing.
    Given state description, operations, and query phrase, predict whether the query box contain each of 100 objects
    """
    
    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1, max_data=None, local_operation_order=-1, subset_mask:Optional[Iterable[bool]]=None):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices.
            local_operation_order (int): The state after how many local operations do we want to prob.
        """
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())

        self.examples, self.num_ops, counts, self.mentioned_objects = self.load_examples(path_to_data, max_examples=max_data)
        if max_data is not None and len(activations) > len(self.examples):
            # cached activations cover the full split; align them with the truncated example list
            activations = activations[:len(self.examples)]
        if local_operation_order != -1:
            self.examples, self.num_ops, counts, self.mentioned_objects, activations = self.load_examples_prior_states(
                path_to_data, local_operation_order=local_operation_order, activations=activations,
                examples=self.examples, num_ops=self.num_ops, all_mentioned_objects=self.mentioned_objects,
                subset_mask=subset_mask
            )

        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        
        assert len(self.activations) ==  len(self.examples), f"find activation shape: {len(self.activations)}, example shape {len(self.examples)}"
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]
    
    def load_examples(self, path_to_data, max_examples=None):
        
        raw_examples = []
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))

        if not has_row_level_labels(raw_examples):  # labels here are per row anyway; only the block layout is assumed
            assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"

        counts = np.zeros((2))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            if (not is_empty or self.include_empty) and n_obj > self.min_prev_objects:
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                o_names = re.findall(r'the ([^ ,.]+)[ ,.]', " ".join(s_parts[:-1]) + " ")
                for o in o_names:
                    if o == "contents": # move content splits, not actual object
                        continue
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)

            if max_examples is not None and len(examples) == max_examples:
                break

        return examples, num_ops, counts, all_mentioned_objects

    def get_prior_operations(self, sentence: str, num_operations: int, local_operation_order: int) -> Tuple[bool, List[str], List[str]]:
        """
        This function previous local operation phrases and global operation phrases
        Args:
            sentence (str): The sentence to find the prior state for
            num_operations (str): The number of local operations on the query box
            local_operation_order (int): The state after how many local operations do we want to prob.
                if -1, the last state/operation (should not be this value). if -2, second to last state/operation.

        return
            (bool) whether datapoint contains enough operations for the local_operation_order
            (List[str]) list of local operation phrases needed to back track
            (List[str]) list of global operation phrases needed to back track
        """
        query_box = sentence[sentence.rfind("Box ") + 4]  # only works for single digit num boxes
        sentence_no_query = sentence[:sentence.removesuffix(".").rfind(".") + 1]

        # in these cases, there are not enough previous states for the query box, invalid data
        if np.abs(local_operation_order) > num_operations+1:
            return False, [], []

        # find previous local operation phrase previous state where query box has a different state
        op_indices = [m.start() for m in re.finditer(f"Box {query_box}", sentence_no_query)][1:]
        prior_local_op_pos_list = op_indices[local_operation_order + 1:]
        prior_local_ops = []
        for prior_local_op_pos in prior_local_op_pos_list:
            start = sentence_no_query[:prior_local_op_pos].rfind(".")+2
            end = sentence_no_query.find(".", start)
            prior_local_op = sentence_no_query[start:end+1]
            prior_local_ops.append(prior_local_op)

        prior_global_ops = sentence_no_query[sentence_no_query.find(prior_local_ops[0]):].strip(".").split(".")
        return True, prior_local_ops, prior_global_ops

    def back_track_box_content(self, sentence: str, box_content: torch.Tensor, prior_local_ops: List[str]) -> torch.Tensor:
        """
        Given list of operations on the box, and current content, calculate what was content before list of operations
        """
        query_box = sentence[sentence.rfind("Box ") + 4]
        for op in prior_local_ops[::-1]:
            o_names = re.findall(r'the ([^ ,.]+) ', op)
            for o in o_names:
                # since we are back-tracking from current state,
                # we flip the effects of each operation
                if "Remove" in op:
                    assert box_content[self.oti[o]] == 0
                    box_content[self.oti[o]] = 1
                elif "Put" in op:
                    assert box_content[self.oti[o]] == 1
                    box_content[self.oti[o]] = 0
                elif "Move" in op:
                    # Move into: move from X into <query_box>
                    if op[op.find(f"Box {query_box}")+5] == ".":
                        assert box_content[self.oti[o]] == 1
                        box_content[self.oti[o]] = 0
                    else: # Move out: move from <query_box> into X
                        assert box_content[self.oti[o]] == 0
                        box_content[self.oti[o]] = 1
        return box_content


    def load_examples_prior_states(
        self,
        path_to_data: str,
        local_operation_order: int,
        activations: List[torch.Tensor],
        examples: List[torch.Tensor],
        num_ops: List[List[int]],
        all_mentioned_objects: List[torch.Tensor],
        subset_mask: Optional[np.ndarray],
    ):
        """
        Instead of probing model for last query box state, we try to obtain n-th prior state of the query
        box, and prob whether model's intermediate layer contain those information, which would support
        a (layer-wise) sequential algorithm hypothesis.

        Additionally, we will discard datapoints that does not have enough prior states, and recalculate
        class counts

        if subset_mask is provided, we assume path to data is to the full dataset, meaning we can easily find
        previous box state by looking at the right previous datapoint

        Args:
            path_to_data (str):
            local_operation_order (int):
            activations (List[torch.Tensor]):
            examples (List[torch.Tensor]):
            num_ops (List[int]):
            all_mentioned_objects (List[torch.Tensor]):
            subset_mask (np.ndarray):
        """
        assert self.min_prev_objects < 1, "self.min_prev_objects > 0 not supported yet"
        # pdb.set_trace(header="debug binary probe load ps")
        # Threw indexing errors, might need to check compatability with altform, and the loading logics
        df_full = pd.read_json(path_to_data, lines=True, orient="records")
        assert subset_mask is None or len(subset_mask) == len(df_full), "subset_mask length should match full dataset size"
        if not self.include_empty:
            df = df_full[df_full.masked_content.apply(lambda s: "is empty" not in s and "nothing" not in s)]
        else:
            df = df_full
        
        if subset_mask is not None:
            df_emb = df_full[subset_mask.astype(bool)]
            if not self.include_empty:
                df_emb = df_emb[df_emb.masked_content.apply(lambda s: "is empty" not in s and "nothing" not in s)]

        new_examples = []
        new_num_ops = []
        new_all_mentioned_objects = []
        new_activations = []
        new_counts = np.zeros((2))

        for i, i_df_full in enumerate(df.index):
            # if subsetting, subsetted length should be the same as length of examples
            # otherwise, dataset loaded should be the subset dataset (same as number of examples)
            if subset_mask is not None and not subset_mask[i_df_full]:
                continue

            valid, prior_local_ops, prior_global_ops = self.get_prior_operations(df.iloc[i]["sentence"], df.iloc[i]["numops"], local_operation_order)
            if not valid:
                continue

            # labels: previous query box state
            if subset_mask is None:
                # brute-force computing iteratively based on previous operations
                box_content = self.back_track_box_content(df.iloc[i]["sentence"], examples[i], prior_local_ops)
            else:
                # find out which of the previous datapoint contains the state we want
                prev_i_df_full = i_df_full - len(prior_global_ops)*7
                # if the previounes state is empty, and the state not pre-calculated, we know the content (empty!)
                if prev_i_df_full not in df.index and not self.include_empty:
                    box_content = torch.zeros(self.n_objects)
                else:
                    prev_i = df.index.tolist().index(prev_i_df_full)
                    box_content = examples[prev_i]

            new_examples.append(box_content)
            
            # feature: latest state embedding
            if subset_mask is not None:
                # if subset has to be applied, we need to find the index of the activation,
                # which was calculated on subset + (optionally) filtering on empty
                subset_i = df_emb.index.tolist().index(i_df_full)
                new_activations.append(activations[subset_i])
            else:
                new_activations.append(activations[i])

            # for now keeping the other ones the same as last box state as they don't affect probe prediction signal
            new_num_ops.append(num_ops[i])
            new_all_mentioned_objects.append(all_mentioned_objects[i])
            new_counts += np.array(
                [torch.sum((examples[i] == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in
                 range(2)]).astype(float)

        print(f"prior state={local_operation_order} data loaded: total of {len(new_examples)} examples")
        return new_examples, new_num_ops, new_counts, new_all_mentioned_objects, new_activations


class SpanProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing.
    Given state description and operation phrases, we use hidden states of tuples of tokens (e.g. [box,obj])
      to predict whether the box is bound to the object through a "removal" or "exist" tag.

      There are three cases:
      - If the box contains the object, removal is False, exist is True
      - If the box does not contain the object, removal is False, exist is False
      - If the box previously contains the object but it got removed, removal is True, exist is False

    """

    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1, max_data=None, tokenizer=None, expand_query_box=False, balance_label_sampling=True, span_probe_type:str="numer-object-remove", args=None,split="train",same_phrase_only=False):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices.
            local_operation_order (int): The state after how many local operations do we want to prob.
        """
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.tokenizer = tokenizer
        self.span_probe_type = span_probe_type
        # caching should be layer specific, but that defeats the purpose of caching, so moving on
        # exploded_data_cache_dir = f"{args.model_representation_path}/exploded_{span_probe_type}{'_balanced' if balance_label_sampling else ''}_samePhrase={same_phrase_only}_{split}.pkl"
        # if os.path.exists(exploded_data_cache_dir):
        #     self.activations, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings = self.load_cached_examples(exploded_data_cache_dir)
        # else:
        self.activations, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings = self.load_examples(path_to_data, activations=activations, max_examples=max_data, expand_query_box=expand_query_box, balance_label_sampling=balance_label_sampling, same_phrase_only=same_phrase_only)
            # self.cache_examples(exploded_data_cache_dir, self.activations, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings)
        # self.activations = [act.to(torch.float16) for act in self.activations]
        self.weights = torch.tensor(counts, dtype=torch.float32) / torch.tensor(counts.sum(), dtype=torch.float32)

        # TODO not sure where this is needed
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]

        assert len(self.activations) == len(self.examples)

        self.n = len(self.activations)

    def get_weights(self):
        return self.weights

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]

    @staticmethod
    def load_cached_examples(path:str):
        print(f"Loading cached exploded activations and labels from {path} ...")
        with open(path, "rb") as f:
            output = pickle.load(f)
        exploded_activations, labels, num_ops, counts, all_mentioned_objects, analysis_strings = output
        return exploded_activations, labels, num_ops, counts, all_mentioned_objects, analysis_strings

    @staticmethod
    def cache_examples(path, activations, examples, num_ops, counts, mentioned_objects, analysis_strings):
        with open(path, "wb") as f:
            pickle.dump((activations, examples, num_ops, counts, mentioned_objects, analysis_strings), f)
        print(f"Cached exploded activations and labels to {path} ...")

    def load_examples(self, path_to_data, activations: List[torch.Tensor], max_examples=None, expand_query_box=False, balance_label_sampling=True, same_phrase_only=False):

        raw_examples = []
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
        if max_examples is not None:
            raw_examples = raw_examples[:max_examples]
        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"

        counts = np.zeros((2))  # count of yes vs no across dataset
        exploded_activations = []  # list of activations, concatenation of [box id, object]
        labels = []  # list of labels, each with shape [#tasks] (which is 1 for span prob)
        num_ops = []  # number of global operations so far
        all_mentioned_objects = []  # whether obj is mentioned in prompt or not, not meaningful for span prob because all objects are mentioned,
                                    # instead we are repurposing this for whether obj and boxid are mentioned in the same phrase
        analysis_strings = [] # just for error analysis, save the sentence, with span highlighted, so we know when models are predicting correctly
        box_contents = torch.zeros(self.n_objects)  # vector with object positions, void = 0
        for i, ex in tqdm(enumerate(raw_examples), total=len(raw_examples)):
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            if (not is_empty or self.include_empty) and n_obj > self.min_prev_objects:
                # num_ops.append([len(s_parts) - 2] * self.n_objects)  # TODO
                box_contents = torch.zeros(self.n_objects)

                # here we start our explosion, expanding pairwise (box id, object) embeddings
                if same_phrase_only:
                    activation_index_tuples, label_list, token_strs, same_phrase_labels = self.get_same_phrase_box_id_object_pairs(ex["sentence"], expand_non_query_box=i % 7 == 0)
                else:
                    activation_index_tuples, label_list, token_strs, same_phrase_labels = self.get_all_box_id_object_pairs(ex["sentence"], expand_non_query_box=i%7 == 0, expand_query_box=expand_query_box)
                activation_index_tuples, label_list, same_phrase_labels = np.array(activation_index_tuples), np.array(label_list), np.array(same_phrase_labels)
                if same_phrase_only and len(label_list)>0:
                    same_phrase_idx = np.where(same_phrase_labels)[0]
                    activation_index_tuples = activation_index_tuples[same_phrase_idx]
                    label_list = label_list[same_phrase_idx]
                    same_phrase_labels = same_phrase_labels[same_phrase_idx]

                # subset the resulting pairs if needed
                if balance_label_sampling and len(label_list)>0:
                    positive_data_idx = np.where(label_list)[0]
                    negative_data_idx = np.where(~np.array(label_list))[0]
                    if len(negative_data_idx) > len(positive_data_idx):
                        negative_data_idx = np.random.choice(negative_data_idx, size=len(positive_data_idx), replace=False)
                    else:
                        positive_data_idx = np.random.choice(positive_data_idx, size=len(negative_data_idx), replace=False)

                    sampled_idx = [*positive_data_idx, *negative_data_idx]
                    activation_index_tuples = activation_index_tuples[sampled_idx]
                    label_list = label_list[sampled_idx]

                # because activations are only cached for object/box_ids, we need a index mapper
                relevant_indices = get_token_pos_given_span_types(self.tokenizer.encode(ex["sentence"]), self.tokenizer, "number-object")
                full_to_act_idx = {idx:i for i, idx in enumerate(relevant_indices)}
                for act_tuple, label, same_phrase_label in zip(activation_index_tuples, label_list, same_phrase_labels):
                    if self.span_probe_type.startswith("number-object"):
                        exploded_activations.append(activations[i][0,[full_to_act_idx[t] for t in act_tuple]].view(-1))
                    elif self.span_probe_type.startswith("number-"):
                        exploded_activations.append(activations[i][0, full_to_act_idx[act_tuple[0]]])
                    elif self.span_probe_type.startswith("object-"):
                        exploded_activations.append(activations[i][0, full_to_act_idx[act_tuple[1]]])
                    else:
                        raise NotImplementedError(f"Span probe type {self.span_probe_type} is not implemented.")

                    labels.append(torch.Tensor([label]))
                    counts[label] += 1
                    num_ops.append([self.get_num_ops_by_idx(token_strs, max(act_tuple))])
                    # all_mentioned_objects.append(self.get_mentioned_objects_by_idx(token_strs, max(act_tuple)))
                    # all_mentioned_objects.append(torch.Tensor([1]))  # same shape as label. not meaningful
                    # here we repurpose all_mentioned_objects field as whether object and box-id is cross phrase boundary
                    all_mentioned_objects.append(torch.Tensor([same_phrase_label]))

                    # format the analysis string for error analysis
                    a_string = " ".join([f"**{t}**" if i in act_tuple else t for i,t in enumerate(token_strs)])
                    analysis_strings.append(a_string)
        print(f"Processed {len(raw_examples)} raw examples, exploded to {len(labels)} tuple pairs. Expansion rate = {len(labels) / len(raw_examples) * 100:.2f}%")
        return exploded_activations, labels, num_ops, counts, all_mentioned_objects, analysis_strings

    def get_all_box_id_object_pairs(self, sentence: str, expand_non_query_box: bool, expand_query_box: bool) -> Tuple[List[Tuple[int, int]], List[int], List[str], List[str]]:
        """
        Extract pairs of token indices (box id, object) to form the training set for our span prob.
        For every object occurring in the sentence (until the next occurrence of the same object), we
        form a pair.
        For remove tag, if the object was ever removed from the box, we label it True, otherwise False.
        For exist tag, if the object exists in the box currently, we label it True, otherwise False.

        Args:
            sentence (str): Sentence to extract pairs from.
            expand_non_query_box (bool): Whether to expand the non-query boxes from non-query phrase.
            expand_query_box (bool): Whether to for pairs for query box (at query phrase) against previous
                objects.
            same_phrase_label (bool): Whether to expand all labels according to the same phrase.
            condition_on (str): ['number-object', 'number-*', 'object-*']. which, or both of the pair we want
        """
        token_strs = str_to_token_strs(sentence, self.tokenizer)
        if not expand_query_box and not expand_non_query_box:
            return [], [], token_strs, []

        x_list = []
        y_list = []
        same_phrase_labels = []

        phrases = sentence.strip(".").split(". ")
        query_box_id = sentence[sentence.rfind("Box ")+4]
        query_id_idx = len(self.tokenizer.encode(sentence[:sentence.rfind("Box ") + 5])) - 1
        if expand_query_box:
            prev_query_id_idx = self.get_previous_occurrence(token_strs, query_box_id, query_id_idx-1)
            for token_idx in range(prev_query_id_idx+1, query_id_idx):
                token_str = token_strs[token_idx]
                # if there is an object between the previous occurance of query box id and query phrase, add this data
                if is_object(token_str):
                    x_list.append((query_id_idx, token_idx))
                    y_list.append(self.get_tag_label(token_strs, query_id_idx, token_idx))
                    same_phrase_labels.append(0)

        # remove the query phrase
        token_strs = str_to_token_strs(". ".join(sentence.strip(".").split(". ")[:-1]), self.tokenizer)
        if expand_non_query_box:
            # loop through every object
            for obj_token_idx, obj_token_str in enumerate(token_strs):

                # if token is not an object or exceeds query phrase, ignore
                if not is_object(obj_token_str) or obj_token_idx >= query_id_idx:
                    continue

                # search until the next occurrence of this object, form all tuples
                # starting from the beginning if no previous occurrence or current obj index if there is the previous occurrence
                obj_token_start_idx = 0 if 0==self.get_previous_occurrence(token_strs, obj_token_str, obj_token_idx-1) else obj_token_idx
                obj_token_end_idx = self.get_next_occurrence(token_strs, obj_token_str, obj_token_idx+1)

                if self.span_probe_type.startswith("number-object"):
                    for box_id_idx in range(obj_token_start_idx, obj_token_end_idx):
                        if is_box_id(token_strs[box_id_idx]):
                            x_list.append((box_id_idx, obj_token_idx))
                            y_list.append(self.get_tag_label(token_strs, box_id_idx, obj_token_idx))
                            same_phrase_labels.append(self.get_same_phrase_label(token_strs, box_id_idx, obj_token_idx))
                elif self.span_probe_type.startswith("object-"):
                    x_list.append((-1, obj_token_idx))
                    y_list.append(self.get_tag_label(token_strs, -1, obj_token_idx))
                    same_phrase_labels.append(1)
        return x_list, y_list, token_strs, same_phrase_labels
    
    def get_same_phrase_box_id_object_pairs(self, sentence: str, expand_non_query_box: bool):
        """
        Instead of using above complicated logic, simply go through every phrase and append
        """
        token_strs = str_to_token_strs(sentence, self.tokenizer)
        if not expand_non_query_box:
            return [], [], token_strs, []

        x_list = []
        y_list = []
        same_phrase_labels = []

        phrases = sentence.strip(".").split(". ")
        query_box_id = sentence[sentence.rfind("Box ")+4]
        query_id_idx = len(self.tokenizer.encode(sentence[:sentence.rfind("Box ") + 5])) - 1

        # remove the query phrase
        token_strs = str_to_token_strs(". ".join(sentence.strip(".").split(". ")[:-1]), self.tokenizer)
        # loop through every object
        for obj_token_idx, obj_token_str in enumerate(token_strs):

            # if token is not an object or exceeds query phrase, ignore
            if not is_object(obj_token_str) or obj_token_idx >= query_id_idx:
                continue

            # search until the end of phrase, form all tuples
            # starting from the beginning if no previous occurrence or current obj index if there is the previous occurrence
            obj_token_start_idx = obj_token_idx  # assumption here is box id always come after in a phrase
            obj_token_end_idx = self.get_next_occurrence(token_strs, [",", "."], obj_token_idx+1)

            if self.span_probe_type.startswith("number-object"):
                for box_id_idx in range(obj_token_start_idx, obj_token_end_idx):
                    if is_box_id(token_strs[box_id_idx]):
                        x_list.append((box_id_idx, obj_token_idx))
                        y_list.append(self.get_tag_label(token_strs, box_id_idx, obj_token_idx))
                        same_phrase_labels.append(1)
            # elif self.span_probe_type.startswith("object-"):
            #     x_list.append((-1, obj_token_idx))
            #     y_list.append(self.get_tag_label(token_strs, -1, obj_token_idx))
            #     same_phrase_labels.append(1)
        return x_list, y_list, token_strs, same_phrase_labels
        
        
    def get_next_occurrence(self, token_strs: List[str], token: Union[List[str], str], start_idx: int) -> int:
        for idx in range(start_idx, len(token_strs)):
            if isinstance(token, str) and token == token_strs[idx]:
                return idx
            elif isinstance(token, list) and token_strs[idx] in token:
                return idx
        return len(token_strs)

    def get_previous_occurrence(self, token_strs: List[str], token: Union[List[str], str], start_idx: int) -> int:
        for idx in range(start_idx, -1,-1):
            if isinstance(token, str) and token == token_strs[idx]:
                return idx
            elif isinstance(token, list) and token_strs[idx] in token:
                return idx
        return 0

    def get_tag_label(self, token_strs: List[str], query_id_idx: int, obj_idx: int) -> bool:
        if self.span_probe_type.startswith("number-object"):
            if self.span_probe_type.endswith("remove"):
                return self.get_removal_tag_label(token_strs, query_id_idx, obj_idx)
            elif self.span_probe_type.endswith("exist"):
                return self.get_exist_tag_label(token_strs, query_id_idx, obj_idx)
        elif self.span_probe_type.startswith("object-"):  # object only
            if self.span_probe_type.endswith("remove"):
                return self.get_removal_tag_label_object_only(token_strs, obj_idx)
            elif self.span_probe_type.endswith("exist"):
                pass
        raise NotImplementedError

    def get_removal_tag_label(self, token_strs: List[str], query_id_idx: int, obj_idx: int):
        """Given sentence and token positions of the box id token and object token, determine the removal label
        (if the object has been removed from the box before)"""
        partial_sentence = token_strs_to_str(token_strs[:max(query_id_idx, obj_idx)+1], self.tokenizer) + "."
        obj_str, query_id = token_strs[obj_idx].strip().lower(), token_strs[query_id_idx].strip().lower()
        if re.findall(rf'Remove the (?=[^.]*{obj_str})([^.]*) from Box {query_id}\.', partial_sentence):
            return True
        elif re.findall(rf"Move the (?=[^.]*{obj_str})([^.]*) (in|from) Box {query_id}\.", partial_sentence):
            return True
        else:
            return False

    def get_exist_tag_label(self, token_strs: List[str], query_id_idx: int, obj_idx: int):
        """Given sentence and token positions of the box id token and object token, determine the exist label
        (if the object exist in the box)"""
        partial_sentence = token_strs_to_str(token_strs[:max(query_id_idx, obj_idx)+1], self.tokenizer) + "."
        obj_str, query_id = token_strs[obj_idx].strip().lower(), token_strs[query_id_idx].strip().lower()
        phrases = partial_sentence.split(", ")
        if "." in phrases[-1]:
            phrases = [*phrases[:-1], *phrases[-1].split(". ")]
        exist = False
        for i, phrase in enumerate(phrases):
            if f" {obj_str} " in phrase and f"Box {query_id}" in phrase:
                if i < NUM_BOXES:  # descriptions
                    exist = True
                else:  # operations
                    if "Put" in phrase:
                        assert not exist
                        exist = True
                    elif "Remove" in phrase:
                        assert exist
                        exist = False
                    elif f"Move" in phrase:
                        if f"in Box {query_id}" in phrase or f"from Box {query_id}" in phrase:
                            assert exist
                            exist = False
                        else:
                            assert not exist
                            exist = True
                    else:
                        raise Exception("Weird operations found")
        # print(f"sentence:\n{partial_sentence}\nObject: {obj_str}, around {token_strs[max(obj_idx - 4, 0):obj_idx + 4]}\nBox ID: {query_id}, around {token_strs[max(query_id_idx - 4, 0):query_id_idx + 4]}\nExist: {exist}")
        return exist

    def get_num_ops_by_idx(self, token_strs:List[str], idx: int) -> int:
        """Given sentence, check how many global operations are there by an index"""
        partial_sentence = token_strs_to_str(token_strs[:idx+1], self.tokenizer)
        return partial_sentence.count(". Move ")+partial_sentence.count(". Remove ")+partial_sentence.count(". Put ")

    def get_mentioned_objects_by_idx(self, token_strs:List[str], idx: int) -> torch.Tensor:
        """Given sentence, check how many objects have been mentioned by an index"""
        partial_sentence = token_strs_to_str(token_strs[:idx + 1], self.tokenizer)
        o_names = get_objects(partial_sentence)
        mentioned_objects = torch.zeros(self.n_objects)  # vector with mentioned objects
        for o in o_names:
            if o == "contents":  # move content splits, not actual object
                continue
            oidx = self.oti[o]
            mentioned_objects[oidx] = 1
        return mentioned_objects

    def get_same_phrase_label(self,token_strs, box_id_idx, obj_idx) -> int:
        """ Return 1 if box id and object are mentioned in the same phrase, 0 otherwise"""
        phrase_between = " ".join(token_strs[min(box_id_idx, obj_idx): max(box_id_idx, obj_idx)+1])
        if "." in phrase_between or "," in phrase_between:
            return 0  # they are in different phrases
        else:
            return 1  # they are in the same phrase

    def get_removal_tag_label_object_only(self, token_strs: List[str], obj_idx: int) -> List[str]:
        # TODO hummmm doesn't make sense because Move case you can't determine the tag..
        cur_idx = obj_idx-1
        while cur_idx >= 0:
            if "Remove" == token_strs[cur_idx]:
                return True
            if token_strs[cur_idx].strip() in [",", "."]:
                return False
        return False


class PhraseProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing.
    Given end of the phrase, predict whether the current mentioned box contain each of 100 objects
    could be cumulative state or just current phrase state. Three labels possible corresponds to
    {0: non-exist, 1: exist, 2: removed}
    """

    mask_fields=["local_box", "local_obj", "local_box_obj", "cum_box", "cum_obj", "cum_box_obj"]

    def __init__(self, activations: Optional[List[torch.Tensor]], path_to_data:str, object_to_index_map, include_empty=True, max_data=None, tokenizer=None, args=None, split="train", activation_h5_path:Optional[str] = None):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe. If activation file is too big
                (would be None), then we expect loading activation index, which we load using h5 during loader dynamically
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices.

        """
        self.include_empty = include_empty
        self.oti = object_to_index_map
        self.oti_box = lambda obj, box_id: int(box_id)*self.n_objects + self.oti[obj]
        self.n_objects = len(self.oti.keys())
        self.tokenizer = tokenizer
        self.args = args
        self.split = split
        self.activation_h5_path = activation_h5_path

        if self.activation_h5_path is not None:
            layer_activation_h5_path = f"{self.activation_h5_path}/representations_l{args.layer}.h5"
            if os.path.exists(layer_activation_h5_path):
                print("found layer specific activation h5 files, using those instead!")
                self.activation_h5_path = layer_activation_h5_path
        self.activation_h5_file = None
        subset_str = "_subset" if (args.dataset_subset or (self.split == "test" and args.dataset_subset_test_only)) else ""
        exploded_data_cache_dir = f"{args.model_representation_path}/exploded_{args.condition_on}_{split}{subset_str}.pkl"
        # pdb.set_trace(header="phrase probe dataloader init")
        if os.path.exists(exploded_data_cache_dir):
            self.activation_index_tuples, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings = self.load_cached_examples(exploded_data_cache_dir)
        else:
            self.activation_index_tuples, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings = self.load_examples(path_to_data, max_examples=max_data)
            self.cache_examples(exploded_data_cache_dir, self.activation_index_tuples, self.examples, self.num_ops, counts, self.mentioned_objects, self.analysis_strings)
        self.activations = activations if activation_h5_path is None else []

        # following the weight calculation in ProbeDataloader, where 3 is #classes, maybe 3 is too extreme # (NUM_BOXES + 1) also doesn't work local-obj-box goes down
        self.weights = torch.tensor(counts, dtype=torch.float32).sum() / torch.tensor(counts*3, dtype=torch.float32)
        print(f"phrase probe dataset CE weights={self.weights}")
        self.masks = self.get_data_masks(self.analysis_strings, self.oti_box, self.args.condition_on)
        self.mask_ds = datasets.Dataset.from_dict(self.masks)

        # cache some results
        mask_cache_path = f"{self.args.model_representation_path}/{split}_masks{subset_str}.pt"
        if not os.path.exists(mask_cache_path):  # beware of not refreshed caches
            torch.save(self.masks, mask_cache_path)
        del self.masks
        self.mask_ds = self.mask_ds.remove_columns([c for c in self.mask_ds.column_names if c not in self.mask_fields])

        analysis_str_cache_path = f"{self.args.model_representation_path}/{split}_inputs{subset_str}.txt"
        if not os.path.exists(analysis_str_cache_path):  # beware of not refreshed caches
            with open(analysis_str_cache_path, "w") as f:
                f.writelines("\n".join(self.analysis_strings))

        assert len(self.activation_index_tuples) == len(self.examples)
        self.n = len(self.examples)

    def get_weights(self):
        return self.weights

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if self.activation_h5_path is not None:
            if self.activation_h5_file is None:
                self.activation_h5_file = h5py.File(self.activation_h5_path, "r", swmr=True, libver='latest')
            data_idx, token_idx = self.activation_index_tuples[index]
            activation = torch.from_numpy(self.activation_h5_file['activations'][f'activations_{data_idx}'][()])
            if activation.dim == 3:
                activation = activation[self.args.layer - 1, token_idx]
            else:  # per-layer cache
                activation = activation[token_idx]
        else: 
            # activations is still just list of torch tensors
            # activation = self.activations[index]   # old behavior
            data_idx, token_idx = self.activation_index_tuples[index]
            activation = self.activations[data_idx][0, token_idx]
        return activation, self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mask_ds[index]

    @staticmethod
    def load_cached_examples(path:str):
        print(f"Loading cached exploded activations and labels from {path} ...")
        with open(path, "rb") as f:
            output = pickle.load(f)
        exploded_activations, labels, num_ops, counts, all_mentioned_objects, analysis_strings = output
        return exploded_activations, labels, num_ops, counts, all_mentioned_objects, analysis_strings

    @staticmethod
    def cache_examples(path, activations, examples, num_ops, counts, mentioned_objects, analysis_strings):
        with open(path, "wb") as f:
            pickle.dump((activations, examples, num_ops, counts, mentioned_objects, analysis_strings), f)
        print(f"Cached exploded activations and labels to {path} ...")

    def load_examples(self, path_to_data, max_examples=None):
        raw_examples = []
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))

        assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"

        counts = np.zeros((3))  # count of {non-exist, exist, removed} no across dataset
        activation_index_tuples = []  # which datapoint is activation from
        labels = []  # list of labels, each with shape [#tasks] (which is 7 * 100), and range from 0-2: exist, non-exist, removed
        num_ops = []  # number of global operations so far
        all_mentioned_objects = []  # whether obj is mentioned up to the point in prompt or not
        analysis_strings = []  # just for error analysis, save the sentence, with span highlighted, so we know when models are predicting correctly
        for i, ex in tqdm(enumerate(raw_examples), total=len(raw_examples)):
            
            # pdb.set_trace(header="inside load_examples loop")
            
            is_empty = True
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False

            if not is_empty or self.include_empty:
                # extract all previous positions of comma or period
                sentence_no_query = ex["sentence"][:ex["sentence"].strip(".").rfind(".")+1]
                token_strs = str_to_token_strs(sentence_no_query, self.tokenizer)
                if i % 7 == 0:  # expanding 1st example of the 7 is enough
                    activation_idx_list, label_list = self.get_index_and_label_pairs(token_strs)
                    if self.args.condition_on.startswith("period_comma_prior"):
                        activation_idx_list = [i-1 for i in activation_idx_list]
                else:
                    activation_idx_list, label_list = [], []

                # now append each activation and label pairs
                relevant_indices = get_token_pos_given_span_types(self.tokenizer.encode(sentence_no_query), self.tokenizer, self.args.condition_on)
                if self.args.condition_on.startswith("period_comma_prior"):
                    relevant_indices = [pos - 1 for pos in relevant_indices]
                full_to_act_idx = {idx: j for j, idx in enumerate(relevant_indices)}
                for activation_idx, label in zip(activation_idx_list, label_list):
                    activation_index_tuples.append((i, full_to_act_idx[activation_idx]))
                    labels.append(label)
                    counts += np.array([torch.sum((label == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(3)]).astype(float)
                    str_so_far = token_strs_to_str(token_strs[:activation_idx], self.tokenizer)
                    num_ops.append([str_so_far.count(".")] * self.n_objects * NUM_BOXES)  # global ops for now
                    mentioned_objects = torch.zeros(self.n_objects)  # vector with mentioned objects
                    o_names = get_objects(str_so_far)
                    for o in o_names:
                        if o == "contents":  # move content splits, not actual object
                            continue
                        oidx = self.oti[o]
                        mentioned_objects[oidx] = 1
                    all_mentioned_objects.append(torch.tile(mentioned_objects, (NUM_BOXES,)))
                    # now save until end of the phrase for analysis string
                    suffix_idx = activation_idx+1
                    while token_strs[suffix_idx].strip() not in [",", "."]:
                        suffix_idx += 1
                    suffix_strs = token_strs_to_str(token_strs[activation_idx+1:suffix_idx], self.tokenizer).strip()
                    analysis_strings.append(f"{str_so_far}**{token_strs[activation_idx].strip()}**{suffix_strs}")
                    # NOTE: in GPT2, this looks like: "The bag is in Box**A**", where the {token_strs[activation_idx]} has a prepended space; But in Llama 70B, the space is tokenized into a separate token, so it looks like "The bag is in **Box**A**". So need to check whether token_strs[activation_idx] correspond to the correct ID token; need to verify whether this is the reason why id cannot be recognized in get_box_ids function;

            if max_examples is not None and i + 1 == max_examples:
                break
        print(f"exploded activations len = {len(labels)} from {len(raw_examples)} original examples ({len(labels)/len(raw_examples)*100:.2f}% expansion rate)")
        # pdb.set_trace(header="after load_examples")
        return activation_index_tuples, labels, num_ops, counts, all_mentioned_objects, analysis_strings

    def get_index_and_label_pairs(self, token_strs: List[str]):
        indices = []
        labels = []  # each label is size [# obj X # boxes], ranging from 0 (non-exist), 1 (exist), and 2 (removed)
        for token_idx, token_str in enumerate(token_strs):
            if token_str.strip().lower() in [".", ","]:
                if self.args.condition_on.endswith("_local"):
                    label = self.get_local_label(token_strs, token_idx, self.tokenizer, self.oti_box)
                elif self.args.condition_on.endswith("_cumulative"):
                    label = self.get_cumulative_label(token_strs, token_idx, self.tokenizer, self.oti_box, prev_label=labels[-1] if labels else None)
                else:
                    raise NotImplementedError

                if self.args.condition_on.startswith("number_all"):
                    # move cases, need to split table and indices
                    box_token_len = len(self.tokenizer.encode("Box 3", add_special_tokens=False))
                    if is_box_id(token_strs[token_idx - box_token_len - 2]):
                        # first, we need to append the move out label
                        indices.append(token_idx - box_token_len - 2)
                        if self.args.condition_on.endswith("_local"):
                            label = self.get_local_label(token_strs, token_idx, self.tokenizer, self.oti_box, both_move_labels="move_out")
                        elif self.args.condition_on.endswith("_cumulative"):
                            label = self.get_cumulative_label(token_strs, token_idx, self.tokenizer, self.oti_box, prev_label=labels[-1] if labels else None, both_move_labels="move_out")
                        labels.append(label)
                        # then, we append the move into label
                        indices.append(token_idx - 1)
                        if self.args.condition_on.endswith("_local"):
                            label = self.get_local_label(token_strs, token_idx, self.tokenizer, self.oti_box, both_move_labels="move_in")
                        elif self.args.condition_on.endswith("_cumulative"):
                            label = self.get_cumulative_label(token_strs, token_idx, self.tokenizer, self.oti_box, prev_label=labels[-1] if labels else None, both_move_labels="move_in")
                        labels.append(label)
                    else:  # non-move operations
                        indices.append(token_idx - 1)
                        labels.append(label)
                elif self.args.condition_on.startswith("object_all"): # for object_all, we want to separate each object with its own labels
                    # first get the object indices
                    obj_indices = []
                    i = token_idx-1
                    while i >= 0:
                        if is_object(token_strs[i].strip().lower()):
                            obj_indices.insert(0, i)
                        elif token_strs[i].strip().lower() in [".", ","]:
                            break
                        i -= 1
                    
                    # for each object, isolate that object's label and append
                    for obj_idx in obj_indices:
                        obj_label = self.isolate_object_label(label, token_strs[obj_idx].strip().lower())
                        indices.append(obj_idx)
                        labels.append(obj_label)
                        
                else: # period_comma ones, period_comma_prior is fixed later
                    indices.append(token_idx)
                    labels.append(label)
        return indices, labels
    
    def isolate_object_label(self, label: torch.Tensor, obj:str) -> torch.Tensor:
        """
        zero-out all label positions other than the object position
        """
        obj_idx = self.oti[obj]
        mask = torch.zeros_like(label)
        for box_i in range(NUM_BOXES):
            mask[box_i*100 + obj_idx] = 1
        obj_label = label.clone() * mask
        return obj_label
        
    @staticmethod
    def get_local_label(token_str: List[str], idx: int, tokenizer, oti_box: Callable[Tuple[str,str],int], both_move_labels:str="both") -> torch.Tensor:
        """
        calculate the state label regarding the last phrase only
        """
        for i in range(idx-1, 0, -1):
            if token_str[i].lower().strip() in [",", "."]:
                break

        last_phrase = token_strs_to_str(token_str[i: idx], tokenizer)
        label = PhraseProbeDataLoader.update_label_after_phrase(last_phrase, oti_box, both_move_labels=both_move_labels)
        return label

    @staticmethod
    def get_cumulative_label(token_str: List[str], idx: int, tokenizer, oti_box:Callable[Tuple[str,str],int], prev_label: torch.Tensor=None, n_objects=100, both_move_labels:str="both") -> torch.Tensor:
        """
        calculate the cumulative state label up to the last phrase
        somewhat inefficient for now because we update phrase by phrase
        """
        if prev_label is not None:  # if
            for i in range(idx - 1, 0, -1):
                if token_str[i].lower().strip() in [",", "."]:
                    break

            last_phrase = token_strs_to_str(token_str[i: idx], tokenizer)
            label = PhraseProbeDataLoader.update_label_after_phrase(last_phrase, oti_box, label=prev_label, n_objects=n_objects, both_move_labels=both_move_labels)
        else:  # go through all phrases and accumulate label
            label = torch.zeros(n_objects * NUM_BOXES)
            phrases = token_strs_to_str(token_str[:idx], tokenizer).strip(".").split(".")
            if len(phrases) > 1:
                phrases = [*phrases[0].split(","), *phrases[1:]]
            else:
                phrases = phrases[0].split(",")
            for phrase in phrases:
                label = PhraseProbeDataLoader.update_label_after_phrase(phrase, oti_box, n_objects=n_objects, both_move_labels=both_move_labels)

        return label

    @staticmethod
    def update_label_after_phrase(phrase:str, oti_box:Callable[Tuple[str,str],int], label:Optional[torch.Tensor]=None, n_objects=100, both_move_labels:str="both") -> torch.Tensor:
        """
        given a single phrase, update/return the label after the phrase.
        label mapping: {0: non-exist, 1: exist, 2: removed}
        """
        if label is None:
            label = torch.zeros(n_objects * NUM_BOXES)

        box_ids = get_box_ids(phrase)
        objects = get_objects(phrase)
        if "Remove " in phrase:
            for o in objects:
                label[oti_box(o, box_ids[0])] = 2  # removed
        elif ("Put " in phrase) or (" contains " in phrase) or (" is in " in phrase) or (" are in " in phrase):
            for o in objects:
                label[oti_box(o, box_ids[0])] = 1  # exist
        elif "Move " in phrase:
            if both_move_labels in ["both", "move_out"]:
                for o in objects:
                    label[oti_box(o, box_ids[0])] = 2  # remove from first box
            if both_move_labels in ["both", "move_in"]:
                for o in objects:
                    label[oti_box(o, box_ids[1])] = 1  # exist in second box
        else:
            raise NotImplementedError("Unrecognizable phrase type")

        return label

    @staticmethod
    def get_data_masks(analysis_strings: List[str], oti_box: Callable[Tuple[str,str],int], condition_on:str="period_comma_prior_local"):
        masks = []
        prev_sent = None
        # pdb.set_trace(header="inside get_data_masks")
        example_str = analysis_strings[0]
        # detect prepending space before **
        space_prepended = False
        replacement_str = ' '
        if example_str[example_str.find("**") - 1] == " ":
            space_prepended = True
            replacement_str = ''
            
        for line in tqdm(analysis_strings):
            mask = {}
            # get model name from tokenizer
            
            
            sent = line.strip().replace("**", replacement_str).strip()
            new_context = prev_sent is None or prev_sent not in sent
            last_phrase = sent if "," not in sent.strip(",") else sent.strip(",").split(",")[-1] if "." not in sent.strip(".") else sent.strip(".").split(".")[-1]

            # get groundtruth labels
            # if condition_on.startswith("number_all") and "Move " in last_phrase:
            #     if len(get_box_ids(last_phrase))==1:  # move out of box
            #         both_move_labels = "move_out"
            #     else:  # move into box
            #         both_move_labels = "move_in"
            #     local_label = PhraseProbeDataLoader.update_label_after_phrase(last_phrase, oti_box, both_move_labels=both_move_labels)
            #     cum_label = local_label if new_context else PhraseProbeDataLoader.update_label_after_phrase(last_phrase, oti_box, label=masks[-1]["cum_label"],both_move_labels=both_move_labels)
            # else:
            #     local_label = PhraseProbeDataLoader.update_label_after_phrase(last_phrase, oti_box)
            #     cum_label = local_label if new_context else PhraseProbeDataLoader.update_label_after_phrase(last_phrase,oti_box,label=masks[-1]["cum_label"])
            # mask["local_label"] = local_label.clone()
            # mask["cum_label"] = cum_label.clone()
            prev_sent = sent

            # get the masks for each of these condition
            box_ids = get_box_ids(last_phrase)
            objs = get_objects(last_phrase)
            if condition_on.startswith("number_all"):
                box_ids = [box_ids[-1]]
            if condition_on.startswith("object_all"):
                objs = [line.split("**")[1].strip()]

            # for the local state of the last phrase
            # Accuracy within the box mentioned in the phrase (out of 100*#boxes)
            mask["local_box"] = torch.zeros(700)
            for box_id in box_ids:
                mask["local_box"][int(box_id) * 100:int(box_id) * 100 + 100] = 1

            # Accuracy within the objs mentioned in the phrase (out of 7*#obj)
            mask["local_obj"] = torch.zeros(700)
            for obj in objs:
                for box_id in range(7):
                    mask["local_obj"][oti_box(obj, box_id)] = 1

            # Accuracy of the box, obj mentioned in the phrase (out of #obj*#box in this phrase)
            mask["local_box_obj"] = torch.zeros(700)
            for obj in objs:
                for box_id in box_ids:
                    mask["local_box_obj"][oti_box(obj, box_id)] = 1

            # for the global states
            if new_context:
                mask["cum_box"] = mask["local_box"]
                mask["cum_obj"] = mask["local_obj"]
                mask["cum_box_obj"] = mask["local_box_obj"]
            else:
                # Accuracy within the all box mentioned in previous contexts (out of max 700)
                mask["cum_box"] = torch.clamp(mask["local_box"] + masks[-1]["cum_box"], min=0, max=1)
                # Accuracy within the all obj mentioned in the phrase (original non-triv acc score)
                mask["cum_obj"] = torch.clamp(mask["local_obj"] + masks[-1]["cum_obj"], min=0, max=1)
                # Accuracy of all box, obj pairs mentioned in previous context (out of #obj*#box in all previous context)
                mask["cum_box_obj"] = torch.clamp(mask["local_box_obj"] + masks[-1]["cum_box_obj"], min=0, max=1)

            masks.append(mask)

        mask_tensors = {}
        for mask_field in masks[0].keys():
            mask = torch.stack([m[mask_field] for m in masks])
            mask_tensors[mask_field] = mask
            if mask_field in PhraseProbeDataLoader.mask_fields:
                mean_cnt = mask.sum(dim=1).mean()
                std_cnt = mask.sum(dim=1).std()
                print(f"{mask_field} mean count = {mean_cnt:.1f} +- {std_cnt:.1f}")
            del mask
        del masks
        return mask_tensors
    
    def get_collate_fn(self):
        pass 

class ObjectLocationProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, max_data=None):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices. 
        """
        self.examples, self.num_ops, counts = self.load_examples(path_to_data, max_examples=max_data)

        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)        

        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]

        
        assert len(self.activations) == len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], torch.tensor([self.examples[index]]), torch.tensor([self.num_ops[index]]).to(torch.long)
    
    def load_examples(self, path_to_data, max_examples=None):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))
                
        y = []
        num_ops = []
        counts = np.zeros(NUM_BOXES + 1)
        for ex in raw_examples:
            s_parts = ex["sentence"].strip(".").split(".")
            s = s_parts[-1].strip()
            if "no box" not in ex["masked_content"]:
                box_no = int(s[-1]) # 4th character is the box number
                y.append(box_no + 1)
                counts[box_no + 1] += 1
            else:
                y.append(0)
                counts[0] += 1

            num_ops.append(len(s_parts) - 2)
            # pdb.set_trace()
            if max_examples is not None and len(y) == max_examples:
                break

        return y, num_ops, counts



class GPTDataloaderForIncrementalLocalState(Dataset):
    """Loads LM dataset for inference of Incremental Local States."""

    def __init__(self, dataframe, tokenizer, max_length=_GPT_MAX_LENGTH, include_empty=True, condition_on="number", min_prev_objects=-1, include_prompt=False, object_map = {}):

        self.tokenizer = tokenizer
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.condition_on = condition_on
        self.n_objects = len(object_map)
        self.oti = object_map
        def get_numops_global(s):
            s = "Description" +  s.split("Description")[-1]
            seq = [st.strip() for st in s.split('.') if st]
            if len(seq) <= 1:
                return 0
            else:
                return len(seq) - 1
        if self.include_empty:
            self.data = dataframe
        else:
            # filter all examples with empty boxes if include_empty is set to False
            f = dataframe["masked_content"].str.contains("nothing") | dataframe["masked_content"].str.contains("is empty")
            self.data = dataframe[-f]
        
            if self.min_prev_objects > 0:
                f = dataframe["masked_content"].str.split(" and ").apply(lambda x: len(x) > self.min_prev_objects)
                self.data = self.data[f]

            self.data = self.data.reset_index()

        # for the original t5 dataset, prefix ends with box number, while for the few-shot dataset, it ends with "contains"
        if any(self.data["prefix"].str.endswith("contains")):
            # a patchy solution to remove the contains from the prefix
            self.data["prefix"] = self.data["prefix"].apply(lambda x: x[:-9] if x.endswith("contains") else x)
            self.data['masked_content'] = self.data['masked_content'].apply(lambda x: x.replace("contains ", "").replace("<extra_id_0> ", ""))
        
        
        self.prefix_text = self.data["prefix"] 
        self.target_text = self.data["sentence"]
        self.max_length = max_length

        if self.min_prev_objects > 0:
            self.prefix_text = self.data.apply(lambda x: x["prefix"] + " " + " and ".join(x["masked_content"].split(" and ")[0:self.min_prev_objects]) + " and the", axis=1)

        elif self.condition_on == "contains":
            # add " contains" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains")
        elif self.condition_on == "the":
            # add " contains the" to prefix
            self.prefix_text = self.data["prefix"].apply(lambda x: x + " contains the")
            
        # Already ends with the box id when using this dataloader. Need to remove the query completely
        

        if include_prompt:
            # self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".")
            self.target_text = self.target_text.apply(lambda x: PROMPT + ". ".join(x.split(". ")[:-1]) + ".\nStatement: " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.iloc[::NUM_BOXES].reset_index(drop=True)
            self.target_text = self.target_text.iloc[::NUM_BOXES].reset_index(drop=True)
            mask = self.prefix_text.apply(get_numops_global) > 0
            self.prefix_text = self.prefix_text[mask].reset_index(drop=True)
            self.target_text = self.target_text[mask].reset_index(drop=True)
            # also drop all the zero-ops samples with no operations
            

            print(self.prefix_text[0])
            print(self.prefix_text[1])
            print("---------")
            print(self.target_text[1])
        else:
            self.prefix_text = self.prefix_text.apply(lambda x:  ". ".join(x.split(". ")[:-1]) + ".")
            self.target_text = self.target_text.apply(lambda x:  ". ".join(x.split(". ")[:-1]) + ". " + x.split(". ")[-1])
            self.prefix_text = self.prefix_text.iloc[::NUM_BOXES].reset_index(drop=True)
            self.target_text = self.target_text.iloc[::NUM_BOXES].reset_index(drop=True)
            mask = self.prefix_text.apply(get_numops_global) > 0
            self.prefix_text = self.prefix_text[mask].reset_index(drop=True)
            self.target_text = self.target_text[mask].reset_index(drop=True)
            
        
            

            

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        self.tokenizer.padding_side = "right"
        target_text = str(self.target_text[index])

        targ = self.tokenizer.batch_encode_plus(
            [target_text], max_length=self.max_length, return_tensors='pt')

        prefix_text = str(self.prefix_text[index])

        pref = self.tokenizer.batch_encode_plus(
            [prefix_text], max_length=self.max_length, return_tensors='pt')

        target_ids = targ['input_ids'].squeeze()
        prefix_ids = pref['input_ids'].squeeze()
        prefix_attn_masks = pref['attention_mask'].squeeze()
        # only keep the last example. Remove the few shot examples.
        if "Description" in prefix_text:
            clean_prefix = "Description" +  prefix_text.split("Description")[-1] # This is used to generate the state matrix, so it should not affect the tokenization or saving activations with the LMs.
        else:
            clean_prefix = prefix_text
            
        start_of_task = None # Find the start of the task description in the tokenized sequence
        
        
        # Generate Mentioned Vectors from clean_prefix
        mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
        # pdb.set_trace()
        s_parts = clean_prefix.strip(".").split(".")
        o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
        for o in o_names: 
            oidx = self.oti[o]
            mentioned_objects[oidx] = 1


        state_matrix, _, _ = generate_state_matrix(clean_prefix, object_map=self.oti, contains_query=False)
        state_matrix = torch.tensor(state_matrix, dtype=torch.float32)
        if "Description" in prefix_text:
            
            start_of_task = [i for i, tkn in enumerate(self.tokenizer.convert_ids_to_tokens(prefix_ids)) if 'Description' in tkn][-1]
        else:
            start_of_task = 0
        
        # 1. Find all the box ids after the start_of_task
        # 2. parse them according to the operations 
        # 3. remove the inital state -- at that time BOX_ID position does not see any contents yet.
        # 4. We shuold have something like [[box ids in op1], [box ids in op2], ...], outer list length = num_ops, and a list of state matrix like [[content of boxes in op1], [content of boxes in op2], ...], outer list length = num_ops, inner list length = length of box ids in that operation
        
        # 5 remove 3. now we can have box_id before contents
        seq = [st.strip() for st in clean_prefix.split('.') if st]
        ori_state = seq[0]
        # pdb.set_trace(header="111")
        op_seq = seq[1:] 
        box_ids_in_ops = []
        for op in op_seq:
            box_ids = re.findall(r'Box (\d+)', op)
            if len(box_ids) > 0:
                box_ids_in_ops.append([int(bid) for bid in box_ids])
            else:
                box_ids_in_ops.append([])
        box_states_in_ops = []
        # if there's no operation at all, we should not remove the initial state
        for t in range(1, state_matrix.shape[0]): # would accidentally remove the final state when there's no operation at all
            box_states_in_ops.append([state_matrix[t, bid, :] for bid in box_ids_in_ops[t-1]])
        
        box_positions_in_ops = []
        flattened_box_ids = []

        for bids in box_ids_in_ops:
            flattened_box_ids.extend(bids)

        box_id_positions = []
        num_id_encountered = 0
        for i, tkn in enumerate(self.tokenizer.convert_ids_to_tokens(prefix_ids)):
            # just find all the box ids (integers)
            # if is interger and i > start_of_task, need to check token prefix
            clean_token = tkn.replace("Ġ", "").replace("▁", "") # for different tokenizers
            if re.match(r'^\d+$', clean_token) and i > start_of_task:
                num_id_encountered += 1
                if num_id_encountered > NUM_BOXES: # skip the first 7 box ids, which should correspond to the initial state description
                    box_id_positions.append(i)
                    
        
        assert len(box_id_positions) == len(flattened_box_ids), f"Number of box ids {len(flattened_box_ids)} does not match number of box id positions {len(box_id_positions)}!"

        idx = 0
        for bids in box_ids_in_ops:
            box_positions_in_ops.append(box_id_positions[idx:idx+len(bids)])
            idx += len(bids)
        assert idx == len(box_id_positions), f"Index {idx} does not match number of box id positions {len(box_id_positions)}!"
        
        # should use the flattened version of box box id position and box local states -- easier for indexing in the model activations
        box_positions_flattened = []
        box_states_flattened = []
        for bpos, bstate in zip(box_positions_in_ops, box_states_in_ops):
            box_positions_flattened.extend(bpos)
            box_states_flattened.extend(bstate)
        
            
        
        
                
        
        return {
            'target_ids': target_ids.to(dtype=torch.long),
            'prefix_ids': prefix_ids.to(dtype=torch.long),
            'prefix_attn_masks': prefix_attn_masks.to(dtype=torch.long),
            'box_id_positions': box_positions_in_ops, 
            'box_local_states': box_states_in_ops,
            'box_id_positions_flattened': box_positions_flattened,
            'box_local_states_flattened': torch.stack(box_states_flattened),
            'mentioned_objects': mentioned_objects
        }
        
    def build_dummy_activations(self, hidden_dim=5120):
        """Build dummy activations for testing, according to box_id_positions_flattened length."""
        dummy_activations = []
        for i in range(len(self)):
            data = self[i]
            n_box_ids = len(data['box_id_positions_flattened'])
            dummy_activations.append(torch.zeros((n_box_ids, hidden_dim), dtype=torch.float32))
        return dummy_activations
    

class IncrementalLocalStateProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing. This class mostly serves as a unwrapper for the GPTDataloaderForInference class -- for each example, the activations are saved one tensor. But we need to unroll them according to the box ids and local states."""
    def __init__(self, activations, dataset):
        self.activations = activations # shape: N_Prompts x [boxids x hidden_dim]
        self.dataset = dataset # shape: (N_Prompts x N_Boxes)
        self.oti = dataset.oti
        # pdb.set_trace()
        self.expanded_activations, self.labels, self.num_ops, self.counts, self.all_mentioned_objects = self.expand_examples(dataset, None)


    def expand_examples(self, dataset, path_to_data):
        """
        dataset: the GPTDataloaderForInference dataset
        path_to_data: path to the original data file, used to compute the mentioned vectors (non-trivial cases, but probably we can just focus on the recall rate (acc for positive examples) for now.)
        
        """
        expanded_activations = []
        labels = []
        num_ops = []
        counts = np.zeros(2)
        all_mentioned_objects = []
        # mentioned_objects = [] # ignored for now
        
        for i in range(len(dataset)):
            # pdb.set_trace(header="debug expand examples")
            data = dataset[i]
            act = self.activations[i].squeeze(0) # to shape: [N_box_ids, hidden_dim]
            print(i, act.shape, len(data['box_id_positions_flattened']), data['box_local_states_flattened'].shape)
            assert act.shape[0] == len(data['box_id_positions_flattened']) == data['box_local_states_flattened'].shape[0], f"Activation shape {act.shape[0]} does not match number of box ids {len(data['box_id_positions_flattened'])} or number of box states {data['box_local_states_flattened'].shape[0]}!"
            num_boxes_to_unroll = len(data['box_id_positions_flattened'])
            for i in range(num_boxes_to_unroll):
                box_id = data['box_id_positions_flattened'][i]
                box_state = data['box_local_states_flattened'][i]
                expanded_activations.append(act[i])
                labels.append(box_state)
                
                num_ops.append([len(data['box_id_positions']) - 1] * len(self.oti)) # all objects share the same number of operations
                all_mentioned_objects.append(data['mentioned_objects'])
                counts += np.array([torch.sum((box_state == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)

        return expanded_activations, labels, num_ops, counts, all_mentioned_objects

    def __getitem__(self, index):
        return self.expanded_activations[index], self.labels[index], torch.tensor(self.num_ops[index]).to(torch.long), self.all_mentioned_objects[index]
    def __len__(self):
        return len(self.expanded_activations)
    
    def get_weights(self):
        weights = torch.tensor([np.sum(self.counts)], dtype=torch.float32) / torch.tensor(self.counts * (NUM_BOXES + 1), dtype=torch.float32) # why nan?
        return weights
    
    
    
class MentionedProbeDataLoader(Dataset):
    """Loads box dataset into format used for probing."""
    
    def __init__(self, activations, path_to_data, object_to_index_map, include_empty=True, min_prev_objects=-1, box_label_base=0):
        """Initialize ProbeDataLoader.

        Args:
            activations (list): List of activations from LM to use as input for the probe.
            path_to_data (str): Path to corresponding dataset.
            object_to_index_map (dict[str,int]): Mapping from object names to indices.
            box_label_base (int): lowest box number in the dataset (0 original, 1 vlm-data).
        """
        self.box_label_base = box_label_base
        self.include_empty = include_empty
        self.min_prev_objects = min_prev_objects
        self.oti = object_to_index_map
        self.n_objects = len(self.oti.keys())
        self.examples, self.num_ops, counts, self.mentioned_objects, self.removed_objects = self.load_examples(path_to_data)
       
        self.weights = torch.tensor([np.sum(counts)], dtype=torch.float32) / torch.tensor(counts * (NUM_BOXES + 1), dtype=torch.float32)
        self.activations = activations
        
        if len(self.activations) == 0:
            self.examples = self.examples[0:0]
            self.num_ops = self.num_ops[0:0]
            self.mentioned_objects = self.num_ops[0:0]
        print(len(self.activations), len(self.examples))
        assert len(self.activations) ==  len(self.examples)
        
        self.n = len(self.activations)
    
    def get_weights(self):
        return self.weights
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, index):
        return self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index], self.removed_objects[index]
    
    def load_examples(self, path_to_data):
        
        raw_examples = []
        
        with open(path_to_data, "r", encoding="UTF-8") as data_f:
            for line in data_f:
                raw_examples.append(json.loads(line))


        if not has_row_level_labels(raw_examples):  # labels here are per row anyway; only the block layout is assumed
            assert len(raw_examples) % NUM_BOXES == 0, f"Number of examples is not a multiple of {NUM_BOXES}!"

        counts = np.zeros((2))
        examples = []
        num_ops = []
        all_mentioned_objects = []
        all_removed_objects = []
        box_contents = torch.zeros(self.n_objects) #vector with object positions, void = 0
        for i, ex in enumerate(raw_examples):
            s_parts = ex["sentence"].strip(".").split(".")
            state_mat, removed_objs, _ = generate_state_matrix(ex["sentence"], self.oti, num_boxes=NUM_BOXES, num_obj=self.n_objects, box_base=self.box_label_base)
            s = s_parts[-1].strip()
            is_empty = True
            n_obj = 0
            if "is empty" not in ex["masked_content"] and "nothing" not in ex["masked_content"]:
                is_empty = False
                contents = [_.replace("the ", "") for _ in ex["masked_content"].replace("<extra_id_0> ", "").replace("contains ", "").split(" and ")]
                for c in contents:
                    n_obj += 1
                    # only consider objects that haven't been output already
                    if self.min_prev_objects < 1 or n_obj > self.min_prev_objects:
                        oidx = self.oti[c]
                        box_contents[oidx] = 1

            
            if (not is_empty or self.include_empty) and n_obj >= self.min_prev_objects: # just to include empty boxes
                counts += np.array([torch.sum((box_contents == j) * torch.tensor([1.0], dtype=torch.float32)).item() for j in range(2)]).astype(float)
                examples.append(box_contents)
                num_ops.append([len(s_parts) - 2] * self.n_objects)
                box_contents = torch.zeros(self.n_objects)
                mentioned_objects = torch.zeros(self.n_objects) #vector with mentioned objects
                removed_objects = torch.zeros(self.n_objects) #vector with removed objects
                for oidx in removed_objs:
                    removed_objects[oidx] = 1
                # o_names = re.findall(r'the ([^ ,.]+) ', " ".join(s_parts[:-1]) + " ")
                o_names = re.findall(r'\bthe ([^\s,.\n]+)', " ".join(s_parts[:-1]) + " ") # this is a regex to find all object names in the sentence
                for o in o_names: 
                    oidx = self.oti[o]
                    mentioned_objects[oidx] = 1
                all_mentioned_objects.append(mentioned_objects)
                all_removed_objects.append(removed_objects)
                

      
        counts = np.zeros((2))
        counts[0] = torch.sum(torch.stack(all_mentioned_objects) == 0).item()
        counts[1] = torch.sum(torch.stack(all_mentioned_objects) == 1).item()
        # may need to divide by NUM BOXES but lets leave it for now
        
        print("Class distribution:", counts)
        print("Number of examples:", len(examples))
        print("Number of Non-Trivial examples:", np.sum(all_mentioned_objects)) # number of mentioned objects
        print("Number of positve samples", counts[1])
        print("Number of negative samples", counts[0])
        
        return examples, num_ops, counts, all_mentioned_objects, all_removed_objects
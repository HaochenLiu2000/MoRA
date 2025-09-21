import json
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
import random
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
import random
from typing import List, Tuple, Dict, Set

def save_dict_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

dict = {}
with open(f"webqsp/documents.json", 'r', encoding='utf-8') as f:
    i=0
    en=0
    j=0
    for line in tqdm(f):
        item = json.loads(line)
        for entity in item["document"]["entities"]:
            en+=1
            if entity["text"] not in dict:
                dict[entity["text"]] = entity["name"]
                i+=1
            else:
                j+=1
    save_dict_to_json(dict, f"webqsp/entity2name.json")
    
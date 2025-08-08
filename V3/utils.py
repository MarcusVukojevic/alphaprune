
from __future__ import annotations
import matplotlib.pyplot as plt
import math
import os
from typing import List, Dict, Union, Optional
import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,)
from datasets import load_dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn.functional as F

def plot_loss():
    pass

def plot_reward():
    pass

def plot_kl():
    pass


def load_model( name_or_path: str, device: str = "cuda", eightbit: bool = True,):
    
    if device == "mps":         
        model = AutoModelForCausalLM.from_pretrained(name_or_path,torch_dtype=torch.float32,    ).to("mps")
    else:
        if eightbit:
            quant_conf = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained( name_or_path, device_map="auto", quantization_config=quant_conf, torch_dtype=torch.float16,)
        else:
            model = AutoModelForCausalLM.from_pretrained(name_or_path,device_map="auto",torch_dtype=torch.float16,low_cpu_mem_usage=True,)
    
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.tokenizer = tokenizer 

    
    return model



def build_calib_dataset(ds_name: str,tokenizer,split: str = "validation",nsamples: int = 100,seq_len: int = 128,pad_short: bool = False,):
    """
    ds_name:
        • "wikitext"                  → wikitext-2-raw-v1 di default
        • "wikitext-2-raw-v1"         → esplicito
        • "wikitext/wikitext-2-raw-v1"
        • qualunque altro dataset HF con campo 'text'
    """
    # carico il dataset
    if ds_name.startswith("wikitext"):
        _, *cfg = ds_name.split("/")
        cfg = cfg[0] if cfg else "wikitext-2-raw-v1"
        raw = load_dataset("wikitext", cfg, split=split)
    else:
        raw = load_dataset(ds_name, split=split)

    bos = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.eos_token_id

    samples = []
    for txt in raw["text"]:
        if not txt.strip():
            continue

        ids = [bos] + tokenizer(txt, add_special_tokens=False).input_ids + [eos]

        # spezzetta in chunk
        for i in range(0, len(ids), seq_len):
            chunk = ids[i : i + seq_len]

            # --- garantisci lunghezza fissa ------------------
            if len(chunk) < seq_len:
                if pad_short:
                    chunk = F.pad(torch.tensor(chunk),
                                  (0, seq_len - len(chunk)),
                                  value=pad).tolist()
                else:
                    continue        # scarta i pezzi corti

            samples.append(torch.tensor(chunk))
            if len(samples) == nsamples:
                return samples

    return samples
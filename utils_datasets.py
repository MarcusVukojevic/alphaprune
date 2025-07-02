import os
import random
from tqdm import trange
from typing import Iterable, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
def build_calib_dataset(
        ds_name: str,
        tokenizer,
        split: str = "validation",
        nsamples: int = 100,
        seq_len: int = 128,
        pad_short: bool = False,          # ← True = pad, False = scarta chunk corti
):
    """
    ds_name:
        • "wikitext"                  → wikitext-2-raw-v1 di default
        • "wikitext-2-raw-v1"         → esplicito
        • "wikitext/wikitext-2-raw-v1"
        • qualunque altro dataset HF con campo 'text'
    """
    # ──────────────────── carica dataset ─────────────────────
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


# We advise against using this data loading method as it introduces bias towards shorter sequences
def collect_samples_with_join(
    data_iter: Iterable, tokenizer: AutoTokenizer, num_samples: int, sequence_length: int, text_key: str = "text"
):
    data = []

    pbar = trange(num_samples, total=num_samples, desc="Preparing calibration data")
    samples_collected = 0
    current_sample = torch.tensor([], dtype=torch.int64)
    for sample in data_iter:
        tokenized_sample = tokenizer(sample[text_key], return_tensors="pt", add_special_tokens=False).input_ids
        current_sample = torch.cat([current_sample, tokenized_sample], dim=1)
        if current_sample.numel() >= sequence_length:
            samples_collected += 1
            pbar.update()
            data.append(current_sample[:, :sequence_length])  # trim to sequence length; this introduces bias
            current_sample = torch.tensor([], dtype=torch.int64)  # reset current sample
        else:
            # add 2 new lines to the current sample
            current_sample = torch.cat([current_sample, tokenizer("\n\n", return_tensors="pt", add_special_tokens=False).input_ids], dim=1) 
        # Stop if sufficient number of samples are collected
        if samples_collected >= num_samples:
            break
    return data


# Load and process WikiText2 dataset
def get_wikitext2(
    num_samples: int,
    sequence_length: int,
    tokenizer: AutoTokenizer,
    train: bool = True,
):
    print("Loading WikiText2")
    if train:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        train_tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
        data = []
        for _ in trange(num_samples, total=num_samples, desc="Preparing calibration data"):
            i = random.randint(0, train_tokens.shape[1] - sequence_length - 1)
            data.append(train_tokens[:, i : i + sequence_length])
    else:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
        test_samples = test_tokens.numel() // sequence_length
        data = []
        for i in range(test_samples):
            data.append(test_tokens[:, i * sequence_length : (i + 1) * sequence_length])
    return data


# Load and process FineWeb-Edu v2 dataset
def get_fineweb_edu(num_tokens: int, sequence_length: int, tokenizer: AutoTokenizer, train: bool = True):
    print("Loading FineWeb-Edu v2")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    tokens_to_load = num_tokens
    if train:
        dataset = dataset.select(range(dataset.num_rows//2))
    else:
        dataset = dataset.select(range(dataset.num_rows//2, dataset.num_rows))
    dataset = dataset.shuffle(seed=0)
    data_iter = iter(dataset)
    data = []
    while tokens_to_load > 0:
        sample = next(data_iter)
        tokenized_sample = tokenizer(sample["text"], return_tensors="pt", add_special_tokens=False).input_ids
        tokenized_sample = tokenized_sample[:, :min(tokenized_sample.shape[1], tokens_to_load)]
        # Split the sequence into multiple samples if it is too long
        # Just throwing away extra tokens would introduce bias to the dataset
        while tokenized_sample.shape[1] > sequence_length:
            data.append(tokenized_sample[:, :sequence_length])
            tokenized_sample = tokenized_sample[:, sequence_length:]
            tokens_to_load -= sequence_length
        data.append(tokenized_sample)
        tokens_to_load -= tokenized_sample.shape[1]
    print(f"Total tokens loaded: {sum([sample.shape[1] for sample in data])}")
    return data

# Load and process C4 dataset
def get_c4(
    num_samples: int,
    sequence_length: int,
    tokenizer: AutoTokenizer,
    train: bool = True,
):
    print("Loading C4")
    if train:
        dataset = load_dataset(
            "allenai/c4",
            "default",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",  # pin revision
        )
        data = []
        data_iter = iter(dataset)
        data = collect_samples_with_join(data_iter, tokenizer, num_samples, sequence_length)
    else:
        dataset = load_dataset(
            "allenai/c4",
            "default",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation[:1100]",
            revision="607bd4c8450a42878aa9ddc051a65a055450ef87",  # pin revision
        )
        test_tokens = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt", add_special_tokens=False).input_ids
        test_samples = test_tokens.numel() // sequence_length
        data = []
        for i in range(test_samples):
            data.append(test_tokens[:, i * sequence_length : (i + 1) * sequence_length])
    return data


def get_data(data_name_or_path: str,num_tokens: int,sequence_length: int,tokenizer: AutoTokenizer,train: bool = True,) -> List[torch.Tensor]:
    # For legacy reasons only fineweb_edu is loaded on a per token granularity
    if os.path.isfile(data_name_or_path):
        data = torch.load(data_name_or_path)[:num_tokens // sequence_length]  # load data
        data = [sample[:, :sequence_length] for sample in data]  # trim to sequence length
    elif data_name_or_path == "wikitext2":
        data = get_wikitext2(num_tokens // sequence_length, sequence_length, tokenizer, train) 
    elif data_name_or_path == "c4":
        data = get_c4(num_tokens // sequence_length, sequence_length, tokenizer, train)
    elif data_name_or_path == "fineweb_edu":
        data = get_fineweb_edu(num_tokens, sequence_length, tokenizer, train)
    else:
        print(data_name_or_path)
        raise ValueError("Unknown dataset.")
    return data
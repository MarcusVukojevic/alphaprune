from __future__ import annotations
import math
import os
from typing import List, Dict, Union, Optional

import torch
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,)
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader
from utils_datasets import get_data

_DEFAULT_BLOCK = 64            # granularità hardware-friendly, posso cambiarla anche in 32 o 16
_MAX_EVAL_TOK  = 512           # lunghezza massima prompt per calcolo ppl
_EVAL_BATCH    = 8             # batch inferenza ppl (ci sta in 24 GB int8)


def load_model( name_or_path: str, device: str = "cuda", block: int = _DEFAULT_BLOCK, eightbit: bool = True,):
    if device == "mps":         
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.float32,    
        ).to("mps")
    else:
        if eightbit:
            quant_conf = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map="auto",
                quantization_config=quant_conf,
                torch_dtype=torch.float16,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name_or_path,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()  # nessun fine-tune

    #  costruzione indice-blocco
    n_blocks = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            n_blocks += math.ceil(m.out_features / block)
    model.tokenizer = tokenizer 
    model.n_blocks  = n_blocks 

    print(f"[utils] {name_or_path}:    {n_blocks} blocchi da {block} neuroni")
    return model


def load_dataset(name: str,tokenizer: AutoTokenizer,split: str = "validation",nsamples: int = 512,seq_len: int = 128,) -> List[torch.Tensor]:
    """
    Restituisce SEMPRE una lista di Tensor (1, seq_len) pronta per evaluate_ppl_seq.
    • se `name` è file locale / {"c4","wikitext2","fineweb_edu"}  → usa get_data
    • altrimenti carica da Hugging-Face dataset   → tokenizza ogni riga di testo
    """
    # -------- token-based (usa get_data) -----------------------------------
    token_based = os.path.isfile(name) or name in {"c4", "wikitext2", "fineweb_edu"}
    if token_based:
        num_tokens = nsamples * seq_len
        data = get_data(
            data_name_or_path=name,
            num_tokens=num_tokens,
            sequence_length=seq_len,
            tokenizer=tokenizer,
            train=(split == "train"),
        )
        print(f"[utils] token dataset {name} – {len(data)} sequenze da {seq_len}")
        return data

    if name == "wikitext":                      
        name = ("wikitext", "wikitext-2-raw-v1") #  tuple (dataset, config)

    # ---------- HF text dataset ----------
    if isinstance(name, tuple):
        ds = hf_load_dataset(name[0], name[1], split=split)
    else:
        ds = hf_load_dataset(name, split=split)


    if nsamples is not None and len(ds) > nsamples:
        ds = ds.shuffle(seed=0).select(range(nsamples))
    data: List[torch.Tensor] = []
    for row in ds:
        # 1) riga è dict { "text": str }
        if isinstance(row, dict) and "text" in row:
            txt = row["text"]

        # 2) riga è singola stringa
        elif isinstance(row, str):
            txt = row

        # 3) riga è tuple/list  (es. ds.iter(batch_size=…) restituisce lista)
        else:
            txt = row[0]          # prima colonna

        tokens = tokenizer(
            txt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=seq_len,
        ).input_ids
        if tokens.shape[1] < seq_len:          # pad a destra con EOS
            pad = torch.full(
                (1, seq_len - tokens.shape[1]),
                tokenizer.eos_token_id,
                dtype=torch.long,
            )
            tokens = torch.cat([tokens, pad], dim=1)
        data.append(tokens[:, :seq_len])
    print(f"[utils] text dataset {name} – {len(data)} sequenze da {seq_len}")
    return data



def create_board(device: str, n_blocks: int | None = None, model=None):
    if n_blocks is None:
        if model is None or not hasattr(model, "n_blocks"):
            raise ValueError("Specificare n_blocks oppure passare model con attr .n_blocks")
        n_blocks = model.n_blocks
    return torch.ones(n_blocks, dtype=torch.bool, device=device)


def reset_game(device: str, n_blocks: int | None = None, model=None):
    return create_board(device, n_blocks=n_blocks, model=model).clone()


torch.no_grad()
def evaluate_ppl(
    data: List[torch.Tensor],
    model,
    batch_size: int = 8,
    device: str | None = None,
) -> float:
    """
    • `data`  deve essere una lista di Tensor (1, seq_len) INT64
    • Calcola la ppl media sul modello autoregressivo
    """
    if device is None:
        device = next(model.parameters()).device
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    total_nll, total_tok = 0.0, 0
    for batch in loader:
        inp = batch.to(device, dtype=torch.long)                      # (B, seq_len)
        att = torch.ones_like(inp, device=device)  # full-mask
        out = model(input_ids=inp, attention_mask=att, labels=inp)
        nll = out.loss.float() * inp.numel()       # mean loss * token
        total_nll  += nll.item()
        total_tok  += inp.numel()

    ppl = math.exp(total_nll / total_tok)
    print(f"[utils] perplexity = {ppl:.3f}  ({total_tok} token)")
    return ppl


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image, ImageDraw

def save_mask_png(
    mask: torch.Tensor | np.ndarray,
    fname: str = "board_with_spacing.png",
    cell_size: int = 20,
    spacing: int = 4
):
    """
    Salva la mask 1-D come immagine a scacchiera con spaziatura visibile e sfondo scuro.
    mask: torch.Tensor o np.ndarray con valori {1=on, 0=off, -1=derank}.
    """
    # 1) Assicuriamoci di avere un array NumPy di int
    if isinstance(mask, torch.Tensor):
        mask_np = mask.clone().cpu().numpy().astype(int)
    else:
        mask_np = mask.flatten().astype(int)

    n    = mask_np.size
    side = math.ceil(math.sqrt(n))
    pad  = side * side - n
    if pad > 0:
        mask_np = np.concatenate([mask_np, np.zeros(pad, dtype=int)])

    # 2) Dimensioni immagine
    img_size = side * cell_size + (side + 1) * spacing
    img = Image.new("RGB", (img_size, img_size), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)

    # 3) Disegna i blocchi
    for idx, val in enumerate(mask_np):
        row = idx // side
        col = idx % side
        x0 = spacing + col * (cell_size + spacing)
        y0 = spacing + row * (cell_size + spacing)
        x1 = x0 + cell_size
        y1 = y0 + cell_size

        if val == 1:
            fill = (255, 255, 255)  # on = bianco
        elif val == -1:
            fill = (160, 160, 160)  # derank = grigio chiaro
        else:
            fill = (0, 0, 0)        # off = nero

        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # 4) Salva
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    img.save(fname)
    print(f"[utils_vis] board con spacing salvata in {fname} ({side}×{side})")


# save_mask_png(game.state, "figures/board_round1.png")
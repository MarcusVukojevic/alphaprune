import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


def load_model(name_or_path: str, device: str = "cuda", eightbit: bool = True):
    """Carica un modello HF causal LM + tokenizer.
    - su MPS forza float32
    - su CUDA può usare 8bit se disponibile
    """
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=torch.float32,
        ).to("mps")
    else:
        if eightbit and _HAS_BNB:
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
    model.tokenizer = tokenizer
    return model


def build_calib_dataset(
    ds_name: str,
    tokenizer,
    split: str = "validation",
    nsamples: int = 100,
    seq_len: int = 128,
    pad_short: bool = False,
):
    """Costruisce una lista di tensori Long di shape [seq_len] per calibrazione.
    Supporta:
      • "wikitext" → default "wikitext-2-raw-v1"
      • qualsiasi dataset HF con campo 'text'
    """
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
        for i in range(0, len(ids), seq_len):
            chunk = ids[i : i + seq_len]
            if len(chunk) < seq_len:
                if pad_short:
                    import torch as _t
                    chunk = F.pad(_t.tensor(chunk), (0, seq_len - len(chunk)), value=pad).tolist()
                else:
                    continue
            import torch as _t
            samples.append(_t.tensor(chunk))
            if len(samples) == nsamples:
                return samples

    return samples
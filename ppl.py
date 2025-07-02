from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, math, tqdm, numpy as np

device = "cuda"
model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True,
    add_bos_token=True,         # abilita il BOS
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

seq_len  = 512           # non superare 2048
buffer   = []            # accumula token qui
tot_nll  = tot_tok = 0

with torch.no_grad():
    for text in tqdm.tqdm(ds["text"]):
        if len(text.strip()) == 0:
            continue

        # ① aggiungi BOS manualmente (id 1) — semplice e sicuro
        buffer.append(tokenizer.bos_token_id)

        # ② tokenizza senza special tokens
        ids = tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids
        buffer.extend(ids)
        # ③ facoltativo: separa gli “articoli” con EOS
        buffer.append(tokenizer.eos_token_id)

        # ④ forma batch fissi di seq_len
        while len(buffer) >= seq_len:
            seq = buffer[:seq_len]
            buffer = buffer[seq_len:]

            inp = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

            # calcola logits in fp32 per stabilità numerica
            outputs = model(inp)
            logits  = outputs.logits.float()        # (1, L, V)

            # cross-entropy manuale (shift a destra) in fp32 e clamp
            log_probs = torch.log_softmax(logits.clamp(-50, 50), dim=-1)

            # target = token successivo
            target = inp[:, 1:].clone()
            logp_t = log_probs[:, :-1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)

            nll = (-logp_t).sum().item()
            tok = target.numel()

            tot_nll += nll
            tot_tok += tok

ppl = math.exp(tot_nll / tot_tok)
print(f"Perplexity Llama-2-7B @512 tok: {ppl:.2f}")

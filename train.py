""" python train.py --out lora_out --epochs 2
Then: python infer.py --split val --out preds_val_ft.jsonl --adapter lora_out/epoch2 --max_pixels <SAME as training>
Needs infer.py in the same folder (reuses its prompt so train/inference match).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, argparse, random, time
from pathlib import Path
from infer import PAGE_PROMPT


def page_local(rows):
    """Rename speakers by first appearance ON THIS PAGE. NARRATION kept; UNKNOWN -> unk."""
    m, out = {}, []
    for r in rows:
        s = r["speaker"]
        name = "NARRATION" if s == "NARRATION" else "unk" if s in ("UNKNOWN", "unk") else m.setdefault(s, f"char{len(m)+1}")
        out.append({"speaker": name, "text": r["text"]})
    return out


def build_examples(path):
    """prepared/*.jsonl (sequence-level, from prep_data.py) -> one example per page."""
    ex = []
    for line in open(path):
        r = json.loads(line)
        for k, (img, rows) in enumerate(zip(r["images"], r["target"])):
            ex.append({"image": img, "k": k + 1,
                       "answer": json.dumps({"lines": page_local(rows)}, ensure_ascii=False)})
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="prepared/train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="lora_out")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model

    random.seed(a.seed); torch.manual_seed(a.seed)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", quantization_config=bnb)
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=128 * 28 * 28, max_pixels=a.max_pixels)

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                     # language-model layers only (vision tower uses "blocks", not "layers")
                     target_modules=r".*layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    model.train()

    data = build_examples(a.train)
    print(len(data), "page examples")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(data) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.05, 1 - s / max(total, 1)))
    log = open(Path(a.out + "_log.jsonl"), "a")

    step = micro = 0; run_loss = 0.0; t0 = time.time()
    for ep in range(1, a.epochs + 1):
        random.shuffle(data)
        for ex in data:
            img = Image.open(ex["image"]).convert("RGB")
            user = [{"type": "image"}, {"type": "text", "text": PAGE_PROMPT.format(k=ex["k"], ctx="")}]
            msgs = [{"role": "user", "content": user}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            full = prompt + ex["answer"] + "<|im_end|>\n"
            enc = proc(text=[full], images=[img], return_tensors="pt")
            n_prompt = proc(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
            ids = enc["input_ids"]; n_ans = ids.shape[1] - n_prompt
            enc = {k: v.to("cuda") for k, v in enc.items()}
            # logits only for the answer tokens (+1 position before them): avoids a huge full-vocab logits tensor
            logits = model(**enc, logits_to_keep=n_ans + 1).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits, ids[0, n_prompt:].to("cuda"))
            (loss / a.accum).backward()
            run_loss += loss.item(); micro += 1
            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                rec = {"epoch": ep, "step": step, "loss": run_loss / a.accum, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0)}
                print(rec, flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); run_loss = 0.0
        model.save_pretrained(f"{a.out}/epoch{ep}")
        print("saved", f"{a.out}/epoch{ep}", flush=True)

if __name__ == "__main__":
    main()

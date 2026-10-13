"""Run a Qwen-VL model on 3-page manga sequences -> predictions JSONL.
  Val : python infer.py --split val  --out preds_val.jsonl  [--adapter lora_dir]
  Test: python infer.py --split test --out preds_test.jsonl [--adapter lora_dir]
Then score val: python score.py --references development/labels.jsonl --predictions preds_val.jsonl --output scores.json
"""
import json, re, argparse, time
from pathlib import Path

PROMPT = """You are reading three consecutive English manga pages (Page 1, Page 2, Page 3).
Transcribe the story text in reading order and say who spoke each line.

INCLUDE: dialogue, internal thoughts, narration boxes, spoken screams/grunts, and punctuation-only balloons such as "..." or "?!".
EXCLUDE: sound effects (and their captions), tiny unclear reaction text, titles/logos/credits/page numbers/ads/watermarks, character intro labels, translator/editor notes, text on signs/clothes/objects, and text inside letters, phones, newspapers or other documents.
Reading order: usually right-to-left, top-to-bottom, following panels and balloon connections. Keep stutters and repeats. Do not keep printed line wrapping.
Speakers: name each character consistently across all three pages as char1, char2, ... (numbered by first appearance). Thoughts belong to the thinking character. Use NARRATION only for narrator boxes. If you cannot tell who speaks, use "unk".

Output ONLY JSON, no other text, in this form:
{"pages": [[{"speaker": "char1", "text": "..."}], [], [{"speaker": "char2", "text": "..."}]]}
"pages" must have exactly 3 lists, one per page; use [] for a page with no story text."""


PAGE_PROMPT = """You are transcribing ONE manga page. The LAST image is the page to transcribe (Page {k} of 3).{ctx}

INCLUDE: dialogue, internal thoughts, narration boxes, spoken screams/grunts, and punctuation-only balloons such as "..." or "?!".
EXCLUDE: sound effects (and their captions), tiny unclear reaction text, titles/logos/credits/page numbers/ads/watermarks, character intro labels, translator/editor notes, text on signs/clothes/objects, and text inside letters, phones, newspapers or other documents.
Reading order: usually right-to-left, top-to-bottom, following panels and balloon connections. Keep stutters and repeats once. Do not keep printed line wrapping. Copy the text exactly as printed.
Speakers: name each character consistently as char1, char2, ... Thoughts belong to the thinking character. Use NARRATION only for narrator boxes. If you cannot tell who speaks, use "unk".

Output ONLY JSON of this form, with no other text:
{{"lines": [{{"speaker": "char1", "text": "..."}}]}}
Use {{"lines": []}} if the page has no story text."""


def parse_page(raw):
    """Parse one-page output -> list of {speaker,text}. Never raises."""
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    obj = None
    try: obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try: obj = json.loads(s[i:j + 1])
            except Exception: obj = None
    rows = []
    if isinstance(obj, dict) and isinstance(obj.get("lines"), list):
        for it in obj["lines"]:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                rows.append({"speaker": str(it.get("speaker", "unk")), "text": it["text"]})
        return rows
    for a, b in re.findall(r'\{\s*"speaker"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', s):
        try: t = json.loads('"' + b + '"', strict=False)
        except Exception: t = b
        rows.append({"speaker": a, "text": t})
    return rows


def degenerate(t):
    return len(t) > 400 or re.search(r"(.{1,6})\1{5,}", t) is not None


def parse_output(raw):
    """Best-effort JSON parse -> 3 lists of {speaker,text}. Never raises."""
    pages = [[], [], []]
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.M).strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            try: obj = json.loads(s[i:j + 1])
            except Exception: obj = None
    if obj is None:  # truncated output: salvage complete line objects, split pages by "[]"-less heuristic is unsafe -> page 1 only
        items = re.findall(r'\{\s*"speaker"\s*:\s*"([^"]*)"\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}', s)
        for a, b in items:
            try: t = json.loads('"' + b + '"', strict=False)
            except Exception: t = b
            pages[0].append({"speaker": a, "text": t})
        return pages
    raw_pages = obj.get("pages", []) if isinstance(obj, dict) else obj
    for k, pg in enumerate(raw_pages[:3] if isinstance(raw_pages, list) else []):
        if not isinstance(pg, list): continue
        for it in pg:
            if isinstance(it, dict) and isinstance(it.get("text"), str):
                pages[k].append({"speaker": str(it.get("speaker", "unk")), "text": it["text"]})
    return pages


def clean(pages):
    """unk -> unique label per line; drop empty/duplicate-adjacent lines; trim."""
    out, n = [], 0
    for pg in pages:
        rows = []
        for r in pg:
            t = re.sub(r"\s+", " ", r["text"]).strip()
            if not re.search(r"\w|[.!?]", t) or degenerate(t): continue
            sp = r["speaker"].strip() or "unk"
            if sp.lower() in ("unk", "unknown"):
                n += 1; sp = f"unk{n}"
            if any(x["text"] == t for x in rows): continue  # drop repeated lines (model looping)
            rows.append({"speaker": sp, "text": t})
        out.append(rows)
    return out


def load_items(split, root):
    if split == "val":
        return [(r["sequence_id"], r["images"]) for r in map(json.loads, open("prepared/val.jsonl"))]
    seqs = json.load(open(root / "sequences.json"))
    return [(s["sequence_id"], [str(root / p) for p in s["images"]]) for s in seqs if s["split"] == split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test", "development"], default="val")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max_pixels", type=int, default=640 * 28 * 28, help="per page; lower if out of memory")
    ap.add_argument("--max_new_tokens", type=int, default=900)
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--context", action="store_true", help="page mode: also show earlier pages + transcript (was copying; off by default)")
    ap.add_argument("--mode", choices=["page", "joint"], default="page",
                    help="page: one page per call with earlier pages as context (default); joint: all 3 pages in one call")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    kw = {}
    if not a.no_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", **kw)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    proc = AutoProcessor.from_pretrained(a.model, min_pixels=128 * 28 * 28, max_pixels=a.max_pixels)

    items = load_items(a.split, Path(a.root))[: a.limit]
    done = {}
    if Path(a.out).exists():  # resume
        for l in open(a.out):
            if l.strip(): r = json.loads(l); done[r["sequence_id"]] = r
    with open(a.out, "a") as f:
        for sid, paths in items:
            if sid in done: continue
            t0 = time.time()
            imgs = [Image.open(p).convert("RGB") for p in paths]

            def run(images, prompt_text, labels=None):
                content = []
                for n, lab in enumerate(labels or [None] * len(images)):
                    if lab: content.append({"type": "text", "text": lab})
                    content.append({"type": "image"})
                content.append({"type": "text", "text": prompt_text})
                text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
                inp = proc(text=[text], images=images, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    gen = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
                out = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                return out, inp["input_ids"].shape[1]

            if a.mode == "joint":
                raw, ntok = run(imgs, PROMPT, [f"Page {k+1}:" for k in range(3)])
                parsed = parse_output(raw)
            else:
                parsed, raws, ntok, so_far = [], [], 0, []
                for k in range(3):
                    ctx = ""
                    use = imgs[: k + 1] if a.context else imgs[k : k + 1]
                    labs = [f"Page {m+1}:" for m in range(k + 1)] if a.context else [None]
                    if so_far and a.context:
                        ctx = (" The earlier pages are shown first for character context. Transcript so far "
                               "(reuse the same speaker names for the same characters): " + json.dumps(so_far, ensure_ascii=False))
                    out, nt = run(use, PAGE_PROMPT.format(k=k + 1, ctx=ctx), labs)
                    rows = parse_page(out)
                    parsed.append(rows); raws.append(out); ntok = max(ntok, nt)
                    so_far += [{"page": k + 1, **r} for r in rows if not degenerate(r["text"])][:40]
                raw = json.dumps(raws, ensure_ascii=False)
            try: pages = clean(parsed)
            except Exception as e:
                print("parse failed:", e); pages = [[], [], []]
            f.write(json.dumps({"sequence_id": sid, "pages": pages}, ensure_ascii=False) + "\n"); f.flush()
            Path(a.out + ".raw").open("a").write(json.dumps({"sequence_id": sid, "raw": raw}, ensure_ascii=False) + "\n")
            print(f"{sid} {sum(map(len, pages))} lines, {ntok} max-in-tokens, {time.time()-t0:.0f}s", flush=True)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

"""Build train/val split (by sequence) and normalised VLM training targets.
Usage: python prep_data.py --data DATA_ROOT   (DATA_ROOT contains development/ and sequences.json)
"""
import json, random, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=".")
ap.add_argument("--seq", default="sequences.json")
ap.add_argument("--val_frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="prepared")
a = ap.parse_args()

seqs = {s["sequence_id"]: s["images"] for s in json.load(open(a.seq))}
labels = [json.loads(l) for l in open(Path(a.data) / "development/labels.jsonl") if l.strip()]

def normalise(pages):
    """Rename speakers to char1.. by first appearance; UNKNOWN -> 'unk' (see README note)."""
    m, out = {}, []
    for page in pages:
        rows = []
        for r in page:
            s = r["speaker"]
            if s == "NARRATION": name = "NARRATION"
            elif s == "UNKNOWN": name = "unk"
            else: name = m.setdefault(s, f"char{len(m)+1}")
            rows.append({"speaker": name, "text": r["text"]})
        out.append(rows)
    return out

recs = [{"sequence_id": r["sequence_id"],
         "images": [str(Path(a.data) / p) for p in seqs[r["sequence_id"]]],
         "target": normalise(r["pages"])} for r in labels]
random.Random(a.seed).shuffle(recs)
nv = round(len(recs) * a.val_frac)
Path(a.out).mkdir(exist_ok=True)
for name, part in (("val", recs[:nv]), ("train", recs[nv:])):
    with open(Path(a.out) / f"{name}.jsonl", "w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(part))

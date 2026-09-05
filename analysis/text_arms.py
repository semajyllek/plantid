"""Embed 20 Newsgroups arms for the modality test (TEXT_PREREG.md).

Deliberately outside narrowcast. The tool takes a dataset and does not know how
the dataset was produced -- that boundary is the reason a change of modality
costs an embedding script rather than a change to the tool.

Writes three `--embeddings` files per arm layout: crowded, varied, background.
"""

import argparse

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, AutoTokenizer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Two groups with several members each, mirroring the image arms (8 Sedum + 6
# Trifolium; 7 Larus + 6 Calidris). A first attempt used one group of five
# against five groups of one -- neither exercises the group fallback, so the
# fitted cascade never answered at group rank and the coverage mechanism under
# test could not appear. Matched at seven labels per arm.
CROWDED = ["comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
           "comp.sys.mac.hardware",
           "rec.autos", "rec.motorcycles", "rec.sport.hockey"]
VARIED = ["comp.graphics", "rec.sport.hockey", "sci.space", "talk.politics.guns",
          "soc.religion.christian", "misc.forsale", "alt.atheism"]


def group_of(label):
    """Top-level newsgroup. Supplied explicitly: the default first-token rule is a
    binomial convention and would make every label its own group here."""
    return label.split(".")[0]


def mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


def embed(texts, tok, model, device, batch=64, max_len=256):
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s + batch], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        out.append(mean_pool(h, enc["attention_mask"]).float().cpu().numpy())
    return np.vstack(out).astype("float32")


def write(path, texts, labels, tok, model, device):
    X = embed(texts, tok, model, device)
    np.savez_compressed(path, descriptor=X, label=np.asarray(labels, dtype=str),
                        group=np.asarray([group_of(l) for l in labels], dtype=str))
    print(f"{path}: {X.shape}, {len(set(labels))} labels", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-prefix", default="/tmp/news")
    ap.add_argument("--per-label", type=int, default=400)
    a = ap.parse_args()

    d = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    names = d.target_names
    by = {n: [] for n in names}
    for text, t in zip(d.data, d.target):
        if len(text.strip()) > 80:            # drop near-empty posts
            by[names[t]].append(text)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    print(f"{MODEL} on {device}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params\n", flush=True)

    for tag, labels in (("crowded", CROWDED), ("varied", VARIED)):
        texts, ls = [], []
        for lab in labels:
            take = by[lab][: a.per_label]
            texts += take
            ls += [lab] * len(take)
        write(f"{a.out_prefix}_{tag}.npz", texts, ls, tok, model, device)

    # negatives: newsgroups in neither arm
    others = [n for n in names if n not in set(CROWDED) | set(VARIED)]
    texts, ls = [], []
    for lab in others:
        take = by[lab][:120]
        texts += take
        ls += ["__OTHER__"] * len(take)
    write(f"{a.out_prefix}_bg.npz", texts, ls, tok, model, device)
    print(f"  negatives drawn from {len(others)} unused newsgroups")


if __name__ == "__main__":
    main()

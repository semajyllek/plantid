"""Distil BioCLIP-2's embedding function into a deployable ViT-B.

BioCLIP-2 answers 72% of captures at 95.6% precision and is a 304M ViT-L that
cannot ship. BioCLIP v1 ships at 46 MB and answers 53%. That 19pp of coverage is
the gap this tries to close (`ONDEVICE_FINDINGS.md`).

The job is smaller than it sounds, for three reasons:

- **The teacher never runs.** 82k BioCLIP-2 embeddings are already cached, so
  training regresses onto vectors that exist on disk rather than doing forward
  passes through a ViT-L.
- **The student's backbone is not random.** It starts as BioCLIP v1, already a
  biology-trained ViT-B at genus 0.931. (Its *output* still starts uncorrelated
  with the teacher's — see Architecture — but the features it computes are
  already good ones.)
- **Nothing new is downloaded.** The transfer set is images already fetched.

## The leak this must not have

The student must never see an image we later evaluate on. Training on the
catalogue test split or on the iNaturalist observations would fit the student to
the exact photographs it is then scored with — invisible in the results and worse
than any leak found in this project so far. `build_transfer_set` therefore takes
**catalogue `split == "train"` plus the background pool, and nothing else**:
val is excluded because `calibration.py` fits temperature on it, test because the
per-organ numbers come from it, and all of iNaturalist because that is the
evaluation set. There is a test pinning this.

## Architecture

The student is BioCLIP v1's visual tower with its 512-d projection removed, so it
emits the raw 768-wide pooled token — no bottleneck below the teacher's 768 — plus
a trainable 768->768 map.

The valuable part of the initialisation is the *backbone*: it starts as a
biology-trained ViT-B rather than noise. The head is a different matter — measured
at step zero, held-out cosine to the teacher is **−0.004**, i.e. nothing. Two
independently-trained encoders have unrelated bases, so no initialisation of a
linear map makes them agree; the head has to learn the change of basis from
scratch either way. Identity is chosen for being neutral and reproducible, not
because it starts anywhere good.

Loss is cosine distance against the L2-normalised teacher embedding, because
`inat_fusion._l2` normalises before the head sees anything: this optimises the
quantity that is actually consumed downstream.

## Augmentation, and what it costs

Targets are precomputed on centre-cropped images, so an augmented view no longer
strictly corresponds to its target. That is deliberate and standard — it teaches
invariance and stops the student memorising 48k images — but it does mean the
objective is "map any view of this plant to the teacher's canonical embedding"
rather than pure function matching. Keep augmentation mild. A held-out slice of
the transfer set is scored every epoch; if train cosine keeps rising while
held-out cosine plateaus, it is memorising and the transfer set needs to grow
(PlantNet has ~244k more images not yet downloaded).

Usage:
    python -m plantid.train.distil --epochs 40 --batch-size 256 --out distil_v1.pt
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS

TEACHER = "bioclip2"
STUDENT = "bioclip1"
SIDE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
OUT_DIR = DATA_PROCESSED / "distil"


def build_transfer_set(cache_dir=DATA_PROCESSED, teacher=TEACHER) -> pd.DataFrame:
    """(local_path, teacher embedding) for every image we are allowed to train on.

    Catalogue *train* split plus the background pool. Excluding val, test and all
    of iNaturalist is the whole correctness argument of this module — see the
    module docstring.
    """
    from plantid.features.embed_background import cache_path as bg_path
    from plantid.features.embed_catalog import cache_path as cat_path

    frames = []
    cat = pd.read_parquet(cache_dir / "catalog_index.parquet")
    bg = pd.read_parquet(cache_dir / "plantnet_background.parquet")
    for manifest, path_fn, train_only in ((cat, cat_path, True), (bg, bg_path, False)):
        lookup = (manifest.assign(image_id=manifest["image_id"].astype(str))
                  .dropna(subset=["local_path"])
                  .set_index("image_id")["local_path"])
        for organ in ORGANS:
            p = path_fn(organ, teacher, cache_dir)
            if not p.exists():
                continue
            z = np.load(p)
            # the background pool has no splits: its images are training-only
            # negatives by construction, so all of them are fair game
            keep = (z["split"] == "train") if train_only else np.ones(len(z["image_id"]), bool)
            ids = z["image_id"][keep].astype(str)
            emb = z["descriptor"][keep]
            have = np.isin(ids, lookup.index.values)
            frames.append(pd.DataFrame({
                "local_path": lookup.loc[ids[have]].to_numpy(),
                "teacher": list(emb[have]),
            }))
    return pd.concat(frames, ignore_index=True)


class TransferSet:
    """Images plus their cached teacher embeddings."""

    def __init__(self, df, cache_dir=DATA_PROCESSED, train=True):
        import torch
        import torchvision.transforms as T

        self.paths = [str(cache_dir / p) for p in df["local_path"]]
        self.targets = torch.tensor(np.stack(df["teacher"].values), dtype=torch.float32)
        self.targets = self.targets / self.targets.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(SIDE, scale=(0.7, 1.0),
                                    interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
        else:
            self.tf = T.Compose([
                T.Resize(SIDE, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(SIDE), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.targets[i]


def build_student(out_dim=768, checkpoint=None, device="cpu"):
    """BioCLIP v1's tower without its 512-d projection, plus a 768->768 identity map.

    Dropping `visual.proj` keeps the full 768-wide pooled token, so the student is
    never squeezed through a space narrower than the teacher's output.
    """
    import torch
    from torch import nn

    import open_clip

    clip, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    visual = clip.visual
    width = visual.proj.shape[0] if getattr(visual, "proj", None) is not None else out_dim
    visual.proj = None

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = visual
            # Neutral init. It does *not* start the student near the teacher --
            # the two encoders have unrelated bases, so cosine at step zero is
            # ~0 whatever this is set to. The backbone is what carries prior
            # knowledge; this map is learned.
            self.head = nn.Linear(width, out_dim, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(torch.eye(out_dim, width))

        def forward(self, x):
            e = self.head(self.visual(x))
            return e / e.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    model = Student()
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device)


def cosine_loss(student, target):
    return (1.0 - (student * target).sum(-1)).mean()


def evaluate(model, loader, device, amp_dtype=None):
    """Mean cosine to the teacher on held-out transfer images."""
    import torch

    model.eval()
    tot = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                e = model(x)
            tot += float((e.float() * y).sum(-1).sum())
            n += len(x)
    model.train()
    return tot / max(n, 1)


def train(epochs=40, batch_size=256, lr=1e-4, head_lr=1e-3, workers=8, val_frac=0.05,
          out=OUT_DIR / "student.pt", limit=None, device=None, seed=0, log_every=50,
          resume=True):
    """`out` should point at durable storage on a preemptible box.

    A Colab VM's disk dies with the session, so saving there means a run that is
    interrupted at epoch 39 is worth exactly nothing. Point `out` at a mounted
    Drive path and every improvement survives; `resume` then picks the run back up
    at the epoch it reached rather than starting over.
    """
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(device or ("cuda" if torch.cuda.is_available()
                                     else "mps" if torch.backends.mps.is_available() else "cpu"))
    amp = torch.bfloat16 if device.type in ("cuda", "cpu") else None

    df = build_transfer_set()
    if limit:
        df = df.sample(min(limit, len(df)), random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    is_val = rng.rand(len(df)) < val_frac
    print(f"transfer set: {len(df):,} images ({(~is_val).sum():,} train / "
          f"{is_val.sum():,} held out)  device {device}", flush=True)

    tr = DataLoader(TransferSet(df[~is_val].reset_index(drop=True), train=True),
                    batch_size=batch_size, shuffle=True, num_workers=workers,
                    drop_last=True, pin_memory=device.type == "cuda")
    va = DataLoader(TransferSet(df[is_val].reset_index(drop=True), train=False),
                    batch_size=batch_size, num_workers=workers)

    model = build_student(out_dim=int(df["teacher"].iloc[0].shape[0]), device=device)
    model.train()
    opt = torch.optim.AdamW([
        {"params": model.visual.parameters(), "lr": lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[lr, head_lr], total_steps=epochs * len(tr), pct_start=0.1)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out.with_suffix(".ckpt")

    hist, best, start = [], -1.0, 1
    base = None
    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        hist, best, start, base = ck["history"], ck["best"], ck["epoch"] + 1, ck["baseline_cos"]
        print(f"resuming from {ckpt_path} at epoch {start} "
              f"(best held-out so far {best:.4f})", flush=True)
        # OneCycle is defined over the whole run; rebuild it for what is left
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[lr, head_lr], total_steps=max((epochs - start + 1) * len(tr), 1),
            pct_start=0.1)
    if base is None:
        base = evaluate(model, va, device, amp)
        print(f"epoch  0 (init, = BioCLIP v1 pooled): held-out cosine {base:.4f}", flush=True)
    if start > epochs:
        print(f"already trained {epochs} epochs; nothing to do")
        return best

    def save(ep, val):
        """Model weights for `load_encoder`, plus the state needed to resume.

        Written on every epoch rather than only on improvement: the point is to
        survive the machine going away, and an epoch that did not improve still
        represents real progress through the schedule.
        """
        torch.save({"model": model.state_dict(), "epoch": ep, "best": best,
                    "history": hist, "baseline_cos": base, "val_cos": val}, ckpt_path)
        json.dump({"baseline_cos": base, "history": hist},
                  open(out.with_suffix(".json"), "w"), indent=1)

    for ep in range(start, epochs + 1):
        t0, run = time.time(), 0.0
        for i, (x, y) in enumerate(tr):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                loss = cosine_loss(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += float(loss.detach())
            if log_every and (i + 1) % log_every == 0:
                print(f"  ep{ep} step {i+1}/{len(tr)} loss {run/(i+1):.4f}", flush=True)
        val = evaluate(model, va, device, amp)
        hist.append({"epoch": ep, "train_cos": 1 - run / len(tr), "val_cos": val,
                     "secs": time.time() - t0})
        flag = ""
        if val > best:
            best, flag = val, "  *best"
            torch.save(model.state_dict(), out)   # plain state_dict for load_encoder
        save(ep, val)
        print(f"epoch {ep:2d}: train cosine {1-run/len(tr):.4f}  held-out {val:.4f}"
              f"  ({time.time()-t0:.0f}s){flag}", flush=True)

    print(f"\nbest held-out cosine {best:.4f} (started {base:.4f}) -> {out}")
    print("next: embed with variant 'bioclip1_distil', then "
          "`python -m plantid.eval.rejection --variant bioclip1_distil`")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="subsample, for smoke tests")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT_DIR / "student.pt"),
                    help="put this on durable storage (a mounted Drive path) if the "
                         "machine can be preempted — the VM disk dies with the session")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing .ckpt and start from scratch")
    args = ap.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, head_lr=args.head_lr,
          workers=args.workers, limit=args.limit, device=args.device, out=args.out,
          resume=not args.no_resume)


if __name__ == "__main__":
    main()

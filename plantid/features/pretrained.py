"""Extract embeddings from frozen pretrained vision encoders.

v2 architecture (see ROADMAP.md): rather than fine-tuning a backbone, run a
strong pretrained encoder in inference mode once per image and train only a
small head on the cached vectors. The encoders are never trained, so this is
pure batched inference and runs locally on Apple Silicon (MPS).

Output matches `features/store.py`'s format exactly — each encoder becomes a
`descriptors_{organ}_{variant}.npz` that drops straight into
`store.load_descriptors(organ, variant=...)` and therefore into
`match_eval.evaluate_organ` / `fusion.evaluate_fusion`, directly comparable to
the classical and fine-tuned-CNN numbers.

Requires the `.venv-mps` environment (torch/timm/transformers), not the base
project venv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS

# variant name -> how to build it. `loader` is one of:
#   "timm"      - timm.create_model(spec, pretrained=True, num_classes=0)
#   "open_clip" - open_clip via HF hub ("hf-hub:<repo>"), image tower only
ENCODERS = {
    "dinov3_s": {"loader": "timm", "spec": "vit_small_patch16_dinov3.lvd1689m"},
    "dinov3_b": {"loader": "timm", "spec": "vit_base_patch16_dinov3.lvd1689m"},
    "mobileclip2_s0": {"loader": "open_clip", "spec": "hf-hub:timm/MobileCLIP2-S0-OpenCLIP"},
    "bioclip2": {"loader": "open_clip", "spec": "hf-hub:imageomics/bioclip-2"},
    # In-domain: DINOv2 ViT-B/14 (reg4) fine-tuned on PlantCLEF 2024 (7,806
    # species). Published as a bare safetensors checkpoint with no timm config,
    # so the architecture is constructed explicitly and the weights loaded in.
    # Note img_size=518 (37x37 patches of 14px) — ~5x the pixels of a 224 model.
    "plantclef24": {
        "loader": "timm_checkpoint",
        "spec": "vit_base_patch14_reg4_dinov2.lvd142m",
        "repo": "vincent-espitalier/dino-v2-reg4-with-plantclef2024-weights",
        "filename": "vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all.safetensors",
        "img_size": 518,
    },
}

BATCH_SIZE = 64


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(variant: str, device: str | None = None):
    """Return (model, preprocess, device). Model is eval-mode and frozen."""
    import torch

    cfg = ENCODERS[variant]
    device = device or _device()

    if cfg["loader"] == "timm":
        import timm

        model = timm.create_model(cfg["spec"], pretrained=True, num_classes=0)
        data_cfg = timm.data.resolve_model_data_config(model)
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)
    elif cfg["loader"] == "timm_checkpoint":
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Build with the checkpoint's own head size so the weights load strictly,
        # then discard the head — we want the pooled embedding, not the
        # 7,806-way PlantCLEF logits.
        state = load_file(hf_hub_download(cfg["repo"], cfg["filename"]))
        n_classes = state["head.weight"].shape[0]
        model = timm.create_model(
            cfg["spec"], pretrained=False, num_classes=n_classes, img_size=cfg["img_size"]
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"plantclef checkpoint mismatch: missing={missing} unexpected={unexpected}")
        model.reset_classifier(0)
        data_cfg = timm.data.resolve_model_data_config(model)
        data_cfg["input_size"] = (3, cfg["img_size"], cfg["img_size"])
        preprocess = timm.data.create_transform(**data_cfg, is_training=False)
    elif cfg["loader"] == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(cfg["spec"])
        model = _ImageTower(model)
    else:  # pragma: no cover - guarded by ENCODERS
        raise ValueError(f"unknown loader {cfg['loader']}")

    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, device


class _ImageTower:
    """Adapt an open_clip model to a plain callable returning image embeddings."""

    def __init__(self, clip_model):
        self.clip = clip_model

    def __call__(self, x):
        return self.clip.encode_image(x)

    def eval(self):
        self.clip.eval()
        return self

    def to(self, device):
        self.clip.to(device)
        return self

    def parameters(self):
        return self.clip.parameters()


def embed_images(paths, model, preprocess, device, batch_size=BATCH_SIZE, desc=""):
    """Batched inference over a list of image paths -> (n, dim) float32 array."""
    import torch
    from PIL import Image
    from tqdm import tqdm

    out = []
    for i in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch = paths[i : i + batch_size]
        tensors = []
        for p in batch:
            with Image.open(p) as im:
                tensors.append(preprocess(im.convert("RGB")))
        x = torch.stack(tensors).to(device)
        with torch.no_grad():
            emb = model(x)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def extract_organ(index: pd.DataFrame, organ: str, variant: str, cache_dir: Path = DATA_PROCESSED) -> dict:
    """Embed every image of `organ` with the named encoder, in store.py format."""
    from plantid.features import store

    model, preprocess, device = load_encoder(variant)
    sub = index[(index["organ"] == organ) & index["local_path"].notna()].reset_index(drop=True)
    paths = [str(cache_dir / p) for p in sub["local_path"]]

    emb = embed_images(paths, model, preprocess, device, desc=f"{variant}[{organ}]")
    # Cast to numpy unicode, not object: np.load(allow_pickle=False) — which is
    # what store.load_descriptors uses — rejects object arrays, and pandas
    # string columns become object arrays via .to_numpy().
    data = {
        "image_id": np.asarray(sub["image_id"], dtype=str),
        "species_id": np.asarray(sub["species_id"], dtype=str),
        "species_name": np.asarray(sub["species_name"], dtype=str),
        "split": np.asarray(sub["split"], dtype=str),
        "descriptor": emb,
    }
    store.save_descriptors(data, organ, variant=f"{variant}_emb", cache_dir=cache_dir)
    return data


def main(variants=("dinov3_s",), organs=ORGANS, cache_dir: Path = DATA_PROCESSED):
    index = pd.read_parquet(cache_dir / "plantnet_index.parquet")
    for variant in variants:
        for organ in organs:
            data = extract_organ(index, organ, variant, cache_dir=cache_dir)
            print(f"{variant}/{organ}: {data['descriptor'].shape}")


if __name__ == "__main__":
    import sys

    main(variants=tuple(sys.argv[1:]) or ("dinov3_s",))

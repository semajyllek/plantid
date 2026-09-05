"""Is the final layer the right place to read a frozen encoder?

Every number in this project reads BioCLIP-2's *output* embedding --
proj(ln_post(x[:, 0])) after all 24 blocks. Meta's Perception Encoder reports
that for contrastively-trained encoders the best downstream representations sit
in the *middle* of the stack, not at the end: the last layers specialise toward
the contrastive objective and discard detail a downstream head could use.

If that holds here it is free accuracy -- no compression, no training, no change
in model size. If it does not, we have ruled out a cheap win before spending a
GPU on pruning.

Captures the CLS token after selected blocks, plus mean-pooled patch tokens at
the same depths (the pooling choice is confounded with depth otherwise).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from plantid.features.pretrained import load_encoder

B = Path("/private/tmp/claude-501/-Users-jameskelly-Documents-plantid/"
         "8f28d0ec-b6fd-4bd9-9ee3-a9ec10992c6b/scratchpad")
LAYERS = [12, 16, 18, 20, 21, 22, 23, 24]      # 24 == after the last block
BATCH = 32


def main(setname="common"):
    out = B / setname
    df = pd.read_parquet(out / "manifest.parquet").drop_duplicates("local_path").reset_index(drop=True)
    paths = [str(p) for p in df.local_path]
    print(f"{setname}: {len(paths)} images", flush=True)

    model, preprocess, dev = load_encoder("bioclip2")
    visual = model.clip.visual
    blocks = visual.transformer.resblocks

    grabbed = {}
    hooks = [blocks[i - 1].register_forward_hook(
        lambda mod, inp, o, i=i: grabbed.__setitem__(i, o.detach()))
        for i in LAYERS]

    acc = {f"cls{i}": [] for i in LAYERS}
    acc.update({f"mean{i}": [] for i in LAYERS})
    acc["final"] = []

    from PIL import Image
    for s in range(0, len(paths), BATCH):
        batch = []
        for p in paths[s:s + BATCH]:
            with Image.open(p) as im:
                batch.append(preprocess(im.convert("RGB")))
        with torch.no_grad():
            final = model(torch.stack(batch).to(dev))
        acc["final"].append(final.float().cpu().numpy())
        for i in LAYERS:
            h = grabbed[i]                       # (batch, 257, 1024)
            acc[f"cls{i}"].append(h[:, 0].float().cpu().numpy())
            acc[f"mean{i}"].append(h[:, 1:].mean(1).float().cpu().numpy())
        if s % (BATCH * 40) == 0:
            print(f"  {s}/{len(paths)}", flush=True)

    for h in hooks:
        h.remove()
    np.savez_compressed(out / "layers_bioclip2.npz",
                        label=df.species_name.to_numpy().astype(str),
                        cluster=df.obs_id.to_numpy().astype(str),
                        **{k: np.vstack(v).astype("float32") for k, v in acc.items()})
    print(f"wrote {out/'layers_bioclip2.npz'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "common")

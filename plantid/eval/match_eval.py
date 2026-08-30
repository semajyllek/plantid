"""Top-k species-retrieval accuracy for the classical k-NN matcher.

For a given organ, queries every image in a split against the train-split
gallery (`matching/classical.py`) and reports top-1/top-5/top-10 species
accuracy, alongside the random-guess baseline (k/n_species).
"""

import numpy as np
import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.features import store
from plantid.matching.classical import ClassicalMatcher, DEFAULT_K, build_matcher

TOP_KS = (1, 5, 10)
K_SWEEP = (1, 5, 10, 15, 20, 30)


def evaluate(matcher: ClassicalMatcher, data: dict, split: str, k: int, top_ks=TOP_KS) -> dict:
    mask = data["split"] == split
    queries = data["descriptor"][mask]
    true_species = data["species_id"][mask]

    rankings = matcher.rank_species(queries, k=k)

    hits = {tk: 0 for tk in top_ks}
    for ranked, true_sp in zip(rankings, true_species):
        ranked_species = [sp for sp, _ in ranked]
        for tk in top_ks:
            if true_sp in ranked_species[:tk]:
                hits[tk] += 1

    n = len(true_species)
    out = {f"top{tk}": hits[tk] / n for tk in top_ks}
    out["n"] = n
    return out


def per_species_top1(matcher: ClassicalMatcher, data: dict, split: str, k: int) -> pd.DataFrame:
    mask = data["split"] == split
    queries = data["descriptor"][mask]
    true_species = data["species_id"][mask]
    species_names = data["species_name"][mask]

    rankings = matcher.rank_species(queries, k=k)
    correct = [ranked[0][0] == true_sp for ranked, true_sp in zip(rankings, true_species)]

    df = pd.DataFrame({"species_id": true_species, "species_name": species_names, "correct": correct})
    return (
        df.groupby(["species_id", "species_name"])["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "top1_accuracy", "count": "n"})
        .reset_index()
        .sort_values("top1_accuracy")
    )


def sweep_k(matcher: ClassicalMatcher, data: dict, split: str = "val", ks=K_SWEEP) -> dict:
    return {k: evaluate(matcher, data, split, k, top_ks=(1,))["top1"] for k in ks}


def evaluate_organ(organ: str, cache_dir=DATA_PROCESSED, variant: str | None = None, standardize: bool = True) -> dict:
    matcher = build_matcher(organ, cache_dir=cache_dir, variant=variant, standardize=standardize)
    data = store.load_descriptors(organ, variant=variant, cache_dir=cache_dir)

    sweep = sweep_k(matcher, data, split="val")
    best_k = max(sweep, key=sweep.get)

    test_metrics = evaluate(matcher, data, split="test", k=best_k)
    n_species = len(set(matcher.species_ids))

    return {
        "organ": organ,
        "best_k": best_k,
        "val_sweep": sweep,
        "n_species": n_species,
        **test_metrics,
        "matcher": matcher,
        "data": data,
    }


# CNN embedding variants exported by notebooks/finetune_colab.ipynb
# (variant name -> standardize). supcon_proj is already L2-normalized.
CNN_VARIANTS = {
    "ce_emb": True,
    "supcon_emb": True,
    "supcon_proj": False,
}


def main():
    print(f"{'organ':8s} {'variant':12s} {'best_k':6s} {'top1':>6s} {'top5':>6s} {'top10':>6s}  random@1  random@5")
    for organ in ORGANS:
        result = evaluate_organ(organ)
        n_sp = result["n_species"]
        print(
            f"{organ:8s} {'classical':12s} {result['best_k']:<6d} "
            f"{result['top1']:6.3f} {result['top5']:6.3f} {result['top10']:6.3f}  "
            f"{1/n_sp:7.3f}  {min(5/n_sp, 1):7.3f}"
        )

        worst = per_species_top1(result["matcher"], result["data"], split="test", k=result["best_k"])
        print(f"  worst 5 species ({organ}):")
        for _, row in worst.head(5).iterrows():
            print(f"    {row.species_name:40s} top1={row.top1_accuracy:.2f} (n={row.n})")

        for variant, standardize in CNN_VARIANTS.items():
            cnn_result = evaluate_organ(organ, variant=variant, standardize=standardize)
            print(
                f"{organ:8s} {variant:12s} {cnn_result['best_k']:<6d} "
                f"{cnn_result['top1']:6.3f} {cnn_result['top5']:6.3f} {cnn_result['top10']:6.3f}"
            )


if __name__ == "__main__":
    main()

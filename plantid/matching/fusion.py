"""Late-fusion of per-organ classical matchers via reciprocal rank fusion.

Combines per-organ species rankings (`matching/classical.py`) for synthetic
{leaf, bark, flower, plant_id} observation groups (`data/groups.py`) into a
single fused ranking, using reciprocal rank fusion (Cormack, Clarke & Buettcher,
2009): score(species) = sum over organs of 1 / (RRF_K + rank_in_organ).

A species missing from an organ's top-k ranking simply contributes 0 from
that organ, so groups with a missing/unavailable organ still get scored from
whichever organs are present.
"""

import pandas as pd

from plantid.config import DATA_PROCESSED, ORGANS
from plantid.data.groups import sample_groups
from plantid.features import store
from plantid.matching.classical import DEFAULT_K, build_matcher

RRF_K = 60
TOP_KS = (1, 5, 10)


def fuse_rankings(
    per_organ_rankings: dict, rrf_k: int = RRF_K, weights: dict | None = None
) -> list[tuple[str, float]]:
    """Fuse per-organ ranked (species_id, score) lists via (weighted) reciprocal rank fusion.

    `weights` maps organ -> relative trust in that organ's ranking (e.g. its
    validation-set top-1 accuracy). Organs not present in `weights` default to
    weight 1.0; if `weights` is None, this reduces to plain RRF.
    """
    scores: dict[str, float] = {}
    for organ, ranking in per_organ_rankings.items():
        w = 1.0 if weights is None else weights.get(organ, 1.0)
        for rank, (sp, _) in enumerate(ranking, start=1):
            scores[sp] = scores.get(sp, 0.0) + w / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def _descriptor_lookup(data: dict) -> dict:
    return dict(zip(data["image_id"], data["descriptor"]))


def query_confidence_weights(per_organ_rankings: dict) -> dict:
    """Per-query trust signal: how peaked is each organ's own top-1 vote share.

    A k-NN ranking where the top species captures most of the inverse-distance
    vote mass (close to 1.0) indicates the gallery strongly agrees on one
    answer for this query; a flat distribution (close to 1/n_classes) means
    the organ is effectively guessing. Using the top-1 score itself as a
    weight lets confident organs dominate the fused ranking on a per-query
    basis, rather than relying solely on a fixed per-organ accuracy weight.
    """
    return {organ: (ranking[0][1] if ranking else 0.0) for organ, ranking in per_organ_rankings.items()}


def evaluate_fusion(
    split: str = "test",
    n_per_split: int = 20,
    k_match: int = DEFAULT_K,
    cache_dir=DATA_PROCESSED,
    top_ks=TOP_KS,
    organ_weights: dict | None = None,
    query_confidence: bool = False,
    descriptor_config: dict | None = None,
) -> dict:
    """Evaluate fusion across organs.

    `descriptor_config` optionally maps organ -> {"variant": str | None,
    "standardize": bool, "k_match": int}, letting each organ use a different
    descriptor set (e.g. fine-tuned CNN embeddings from
    notebooks/finetune_colab.ipynb) and its own k. Organs not present default
    to the Phase-2 classical descriptors and `k_match`.
    """
    descriptor_config = descriptor_config or {}
    matchers = {}
    data = {}
    k_matches = {}
    for organ in ORGANS:
        cfg = descriptor_config.get(organ, {})
        matchers[organ] = build_matcher(
            organ, cache_dir=cache_dir, variant=cfg.get("variant"), standardize=cfg.get("standardize", True)
        )
        data[organ] = store.load_descriptors(organ, variant=cfg.get("variant"), cache_dir=cache_dir)
        k_matches[organ] = cfg.get("k_match", k_match)
    lookups = {organ: _descriptor_lookup(data[organ]) for organ in ORGANS}

    index = pd.read_parquet(cache_dir / "plantnet_index.parquet")
    groups = sample_groups(index, n_per_split=n_per_split)
    groups = groups[groups["split"] == split]

    hits_fused = {tk: 0 for tk in top_ks}
    hits_single = {organ: {tk: 0 for tk in top_ks} for organ in ORGANS}
    n = 0

    for _, row in groups.iterrows():
        per_organ_rankings = {}
        for organ in ORGANS:
            desc = lookups[organ].get(row[f"{organ}_image_id"])
            if desc is None:
                continue
            [ranked] = matchers[organ].rank_species(desc, k=k_matches[organ])
            per_organ_rankings[organ] = ranked
            ranked_sp = [sp for sp, _ in ranked]
            for tk in top_ks:
                if row["plant_id"] in ranked_sp[:tk]:
                    hits_single[organ][tk] += 1

        if not per_organ_rankings:
            continue

        weights = dict(organ_weights) if organ_weights else None
        if query_confidence:
            conf = query_confidence_weights(per_organ_rankings)
            if weights is None:
                weights = conf
            else:
                weights = {organ: weights.get(organ, 1.0) * conf.get(organ, 0.0) for organ in per_organ_rankings}

        fused_sp = [sp for sp, _ in fuse_rankings(per_organ_rankings, weights=weights)]
        for tk in top_ks:
            if row["plant_id"] in fused_sp[:tk]:
                hits_fused[tk] += 1
        n += 1

    out: dict = {"n": n}
    for tk in top_ks:
        out[f"fused_top{tk}"] = hits_fused[tk] / n
        for organ in ORGANS:
            out[f"{organ}_top{tk}"] = hits_single[organ][tk] / n
    return out


# Best-performing CNN embedding variant per organ, per match_eval.main()
# (notebooks/finetune_colab.ipynb export). k_match is each organ's best_k
# from that sweep.
CNN_DESCRIPTOR_CONFIG = {
    "leaf": {"variant": "supcon_emb", "standardize": True, "k_match": 20},
    "bark": {"variant": "ce_emb", "standardize": True, "k_match": 5},
    "flower": {"variant": "supcon_emb", "standardize": True, "k_match": 5},
}


def main():
    from plantid.eval.match_eval import evaluate_organ

    organ_results = {organ: evaluate_organ(organ) for organ in ORGANS}
    val_top1 = {organ: r["val_sweep"][r["best_k"]] for organ, r in organ_results.items()}
    print("organ val-top1 weights:", {o: round(w, 3) for o, w in val_top1.items()})

    unweighted = evaluate_fusion(split="test")
    weighted = evaluate_fusion(split="test", organ_weights=val_top1)
    conf = evaluate_fusion(split="test", query_confidence=True)
    combined = evaluate_fusion(split="test", organ_weights=val_top1, query_confidence=True)
    n = unweighted["n"]

    print(f"\nFusion evaluation on {n} synthetic test groups\n")
    print(
        f"{'top-k':>6s} {'leaf':>8s} {'bark':>8s} {'flower':>8s} "
        f"{'fused':>8s} {'fused_wt':>9s} {'fused_conf':>11s} {'fused_combo':>12s}"
    )
    for tk in TOP_KS:
        print(
            f"{tk:>6d} "
            f"{unweighted[f'leaf_top{tk}']:8.3f} "
            f"{unweighted[f'bark_top{tk}']:8.3f} "
            f"{unweighted[f'flower_top{tk}']:8.3f} "
            f"{unweighted[f'fused_top{tk}']:8.3f} "
            f"{weighted[f'fused_top{tk}']:9.3f} "
            f"{conf[f'fused_top{tk}']:11.3f} "
            f"{combined[f'fused_top{tk}']:12.3f}"
        )

    cnn_organ_results = {
        organ: evaluate_organ(organ, variant=cfg["variant"], standardize=cfg["standardize"])
        for organ, cfg in CNN_DESCRIPTOR_CONFIG.items()
    }
    cnn_val_top1 = {organ: r["val_sweep"][r["best_k"]] for organ, r in cnn_organ_results.items()}
    print("\nCNN organ val-top1 weights:", {o: round(w, 3) for o, w in cnn_val_top1.items()})

    cnn_unweighted = evaluate_fusion(split="test", descriptor_config=CNN_DESCRIPTOR_CONFIG)
    cnn_weighted = evaluate_fusion(split="test", descriptor_config=CNN_DESCRIPTOR_CONFIG, organ_weights=cnn_val_top1)
    cnn_n = cnn_unweighted["n"]

    print(f"\nCNN-embedding fusion evaluation on {cnn_n} synthetic test groups\n")
    print(f"{'top-k':>6s} {'leaf':>8s} {'bark':>8s} {'flower':>8s} {'fused':>8s} {'fused_wt':>9s}")
    for tk in TOP_KS:
        print(
            f"{tk:>6d} "
            f"{cnn_unweighted[f'leaf_top{tk}']:8.3f} "
            f"{cnn_unweighted[f'bark_top{tk}']:8.3f} "
            f"{cnn_unweighted[f'flower_top{tk}']:8.3f} "
            f"{cnn_unweighted[f'fused_top{tk}']:8.3f} "
            f"{cnn_weighted[f'fused_top{tk}']:9.3f}"
        )


if __name__ == "__main__":
    main()

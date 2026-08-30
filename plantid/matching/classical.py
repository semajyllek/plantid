"""Classical-descriptor k-NN species matcher.

Builds a per-organ "gallery" from the cached descriptors (`features/store.py`),
restricted to the train split with images flagged by the Phase-2
outlier/label-noise pass (`features/outliers.py`) excluded. Descriptor
dimensions are standardized using gallery statistics (the raw descriptors mix
log-Hu moments, LBP histograms, and color histograms on very different
scales). Species predictions for a query descriptor are produced by k-NN
inverse-distance-weighted voting over the standardized gallery.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from plantid.config import DATA_PROCESSED
from plantid.features import store

DEFAULT_K = 15


class ClassicalMatcher:
    def __init__(self, gallery: np.ndarray, species_ids: np.ndarray, mean: np.ndarray, std: np.ndarray):
        self.gallery = gallery
        self.species_ids = species_ids
        self.mean = mean
        self.std = std

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def rank_species(self, query_descriptors: np.ndarray, k: int = DEFAULT_K) -> list[list[tuple[str, float]]]:
        """Rank candidate species for each query descriptor.

        Returns, per query, a list of (species_id, score) pairs sorted by
        decreasing score, where score is the inverse-distance-weighted vote
        share among the k nearest gallery neighbors (sums to 1 across the
        species present in those neighbors).
        """
        Xq = self._standardize(np.atleast_2d(query_descriptors))
        k = min(k, self.gallery.shape[0])
        dists = cdist(Xq, self.gallery)
        nn_idx = np.argsort(dists, axis=1)[:, :k]

        results = []
        for q, neighbors in enumerate(nn_idx):
            weights = 1.0 / (dists[q, neighbors] + 1e-6)
            votes: dict[str, float] = {}
            for sp, w in zip(self.species_ids[neighbors], weights):
                votes[sp] = votes.get(sp, 0.0) + w
            total = sum(votes.values())
            ranked = sorted(((sp, w / total) for sp, w in votes.items()), key=lambda x: -x[1])
            results.append(ranked)
        return results


def build_matcher(organ: str, cache_dir=DATA_PROCESSED, variant: str | None = None, standardize: bool = True) -> ClassicalMatcher:
    """Build a k-NN matcher for `organ`.

    `variant` selects which cached descriptor set to use (None = the Phase-2
    classical descriptors; e.g. "ce_emb"/"supcon_emb"/"supcon_proj" for the
    fine-tuned CNN embeddings, see `notebooks/finetune_colab.ipynb`).
    `standardize=False` skips z-scoring (appropriate for `supcon_proj`, which
    is already L2-normalized).
    """
    data = store.load_descriptors(organ, variant=variant, cache_dir=cache_dir)

    outliers = pd.read_parquet(cache_dir / "outlier_scores.parquet")
    flagged = set(outliers.loc[(outliers["organ"] == organ) & outliers["is_outlier"], "image_id"])

    is_train = data["split"] == "train"
    is_clean = ~np.isin(data["image_id"], list(flagged))
    mask = is_train & is_clean

    gallery_raw = data["descriptor"][mask]
    if standardize:
        mean = gallery_raw.mean(axis=0)
        std = gallery_raw.std(axis=0)
        std[std < 1e-12] = 1.0
        gallery = (gallery_raw - mean) / std
    else:
        mean = np.zeros(gallery_raw.shape[1])
        std = np.ones(gallery_raw.shape[1])
        gallery = gallery_raw

    return ClassicalMatcher(gallery, data["species_id"][mask], mean, std)

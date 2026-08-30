"""Sample {leaf, bark, flower, plant_id} observation groups from the per-image
index.

PlantNet-300K has no real multi-organ observations (each image is its own
"observation", organ is independent of any individual-plant grouping). So a
"group" here is a *synthetic* combination: one image per organ, sampled
independently within (species_id, split), tagged with the shared plant_id
(species_id). When an organ has fewer images than requested groups in a split
(common for bark), images are reused across groups.
"""

import pandas as pd


def sample_groups(index: pd.DataFrame, n_per_split: int = 20, seed: int = 42) -> pd.DataFrame:
    rows = []
    rng_seed = seed
    for (species_id, split), group in index.groupby(["species_id", "split"]):
        by_organ = {organ: g for organ, g in group.groupby("organ")}
        if not {"leaf", "bark", "flower"} <= set(by_organ):
            continue

        species_name = group["species_name"].iloc[0]
        samples = {}
        for organ, g in by_organ.items():
            replace = len(g) < n_per_split
            samples[organ] = g.sample(n=n_per_split, replace=replace, random_state=rng_seed)
            rng_seed += 1

        for i in range(n_per_split):
            rows.append(
                {
                    "plant_id": species_id,
                    "species_name": species_name,
                    "split": split,
                    "leaf_image_id": samples["leaf"].iloc[i]["image_id"],
                    "bark_image_id": samples["bark"].iloc[i]["image_id"],
                    "flower_image_id": samples["flower"].iloc[i]["image_id"],
                }
            )

    return pd.DataFrame(rows)

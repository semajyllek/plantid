# analysis

One-off scripts that produced results cited in the findings docs. Kept because
they are the reproduction path for specific claims, not because they are library
code — `plantid/` holds anything reusable.

Run from the repo root with `PYTHONPATH=.` and `.venv-mps/bin/python` (they load
models). Outputs go to `analysis/out/`.

| script | reproduces |
|---|---|
| `score_bc2_cml4.py` | BioCLIP-2 int4 costs 0.1pp genus; ties iNaturalist offline (`ONDEVICE_FINDINGS.md`) |
| `score_distil.py` | distillation recovered ~10% of the encoder gap (`ONDEVICE_FINDINGS.md`) |
| `encoder_bakeoff.py` | `bioclip_inat` loses on both corpora, with the PlantNet control |
| `compare_h2h.py` | our systems vs Pl@ntNet on identical photographs (`COMPETITIVE_FINDINGS.md`) |
| `plantnet_abstain.py` | the retraction: our cascade on Pl@ntNet's scores beats us |
| `pool_ablation.py` | reject-pool size does not bind — 59 species ≈ 589 |
| `cohort3.py` | broad / targeted / recovered cohorts; the congener effect (`REJECTION_FINDINGS.md`) |
| `two_runs.py` | pre/post top-up controls, synonym-corrected buckets |
| `verify_cml4.py` | Core ML cache integrity and the two-arm matched/mismatched test |
| `curation_eval.py` | label curation is a wash on the headline (`CATALOG_FINDINGS.md`) |
| `build_record.py`, `charts.py` | the published experimental-record artifact |

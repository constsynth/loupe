# Data Directory

This directory contains small local datasets and generated SAE dashboard
artifacts used by the example notebooks and the interface backend.

## Datasets

The current prototype uses these CSV datasets:

```text
sae_experimental_dataset.csv
sae_experimental_train_dataset.csv
sae_experimental_test_dataset.csv
```

The example notebooks generally expect at least:

```text
text
concept_label
```

Some statistical and intervention experiments can also use:

```text
target_token
```

New generated datasets should use a descriptive name such as:

```text
<dataset_theme>_dataset.csv
<dataset_theme>_train_dataset.csv
<dataset_theme>_test_dataset.csv
```

Datasets can be regenerated from `data_utils/dataset_creating.py` or from the
notebooks in `examples/`.

## SAE Feature Dashboard

`data/sae_feature_dashboard/` is produced by
`examples/build_sae_feature_dashboard.ipynb`. The backend endpoint
`GET /api/feature-dashboard` and the React interface read these files directly.

Expected dashboard artifacts:

```text
dashboard_metadata.json
feature_dashboard.csv
feature_concept_scores.csv
feature_top_tokens.csv
feature_top_examples.csv
sample_token_attributions.csv
```

The files store activation-based feature summaries, concept-feature candidate
scores, top activating tokens, top activating examples, and per-token top
features. They are not causal evidence by themselves; causal claims require
separate intervention and baseline checks.

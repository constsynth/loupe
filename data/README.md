# Data Directory

Generated datasets are saved here as CSV files using the naming pattern:

```text
<dataset_theme>_dataset.csv
```

Train and test splits should use the same rule with split-aware themes, for example:

```text
sae_activation_statistics_train_dataset.csv
sae_activation_statistics_test_dataset.csv
```

CSV data files are ignored by the repository and should be regenerated from
`data_utils/dataset_creating.py` or the notebooks in `examples/`.

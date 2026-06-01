<p align="center">
  <img src="images/loupe-logo-minimal.png" alt="Loupe logo" width="180">
</p>

# Loupe

Loupe is a research library for SAE-centered interpretability of large language
models. The project supports experiments that extract transformer activations,
train sparse autoencoders, evaluate statistical hypotheses about the learned
latent space, build feature dashboards, and inspect token-level attributions and
latent interventions through a React interface.

The main dissertation target model is `Qwen/Qwen2.5-3B-Instruct`. Local
development defaults use `Qwen/Qwen2.5-0.5B-Instruct` so that the pipeline can
be exercised on limited hardware.

## Repository Layout

- `interpretability/sae/` - sparse autoencoder architecture, SAE metrics,
  feature attribution helpers, and token-feature analysis.
- `utils/inference_utils/` - LLM loading, generation, hidden-state extraction,
  hooks, and SAE interventions.
- `utils/train_utils/` - SAE training loop.
- `utils/stat_utils/` - statistical checks for reconstruction, distribution
  preservation, sparsity, separability, and intervention diagnostics.
- `examples/` - notebooks for dataset creation, SAE training, statistical
  evaluation, and dashboard generation.
- `backend/` - FastAPI service used by the interface.
- `interface/` - React/Vite frontend for sandbox attribution and interventions.
- `data/` - small CSV datasets and generated dashboard artifacts.
- `models/` - local SAE checkpoints. Do not commit large checkpoints without an
  explicit project decision.

## Requirements

Use the existing conda `base` environment for this repository:

```bash
conda activate base
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd interface
npm install
cd ..
```

For GPU experiments, install a PyTorch build that matches your CUDA runtime if
the pinned requirement is not appropriate for your machine.

## Data

The example notebooks expect CSV files in `data/`. The current prototype uses:

- `data/sae_experimental_dataset.csv`
- `data/sae_experimental_train_dataset.csv`
- `data/sae_experimental_test_dataset.csv`

The expected columns depend on the notebook. The statistical notebook uses at
least:

- `text`
- `concept_label`
- optionally `target_token` for causal-intervention checks

Dataset creation utilities live in `data_utils/dataset_creating.py`, with a
notebook entry point in `examples/dataset_creating.ipynb`.

## Training an SAE

Use `examples/train_sae.ipynb` to train an SAE checkpoint from LLM activations.
The notebook currently demonstrates the lightweight local setup:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- layer: `model.layers.23` or another selected transformer layer
- token-level activations from `LLM.get_hidden_state(..., return_tokens=True,
  valid_tokens_only=True)`
- output checkpoint under `models/`

Open the notebook:

```bash
jupyter lab examples/train_sae.ipynb
```

The important methodological point is that token-level activations start as:

```text
[batch, sequence, hidden]
```

For SAE training, non-padding token vectors are flattened into:

```text
[valid_tokens, hidden]
```

This is the preferred training shape for the current SAE implementation because
each training example is one hidden-state vector for one valid token.

The training loop is implemented in `utils/train_utils/train_sae.py` and uses:

```text
MSE(h_hat, h) + sparsity_lambda * mean(abs(z))
```

where `h` is the original transformer activation, `z` is the sparse SAE latent
activation, and `h_hat` is the reconstructed activation.

## Statistical Hypothesis Checks

Use `examples/sae_llm_end2end_statistics.ipynb` to evaluate whether an SAE is
good enough for interpretation claims. Configure the notebook variables first:

```python
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"
LAYER_NAME = "model.layers.27"
DATASET_CSV_PATH = PROJECT_ROOT / "data" / "sae_activation_statistics_train_dataset.csv"
SAE_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "qwen2_5_3b_layer27_sae.pt"
```

The notebook checks:

- information preservation: NMSE and cosine similarity between `h` and `h_hat`;
- distribution preservation: MMD as the pass/fail criterion, with KS tests,
  Wasserstein distance, and histogram Jensen-Shannon divergence as diagnostics;
- latent sparsity: Hoyer sparsity as the pass/fail criterion, with L0, active
  feature share, and entropy as diagnostics;
- concept separability: concept structure in dense `h` versus SAE `z`;
- token-based feature attribution: top activating tokens and contexts;
- optional causal selectivity: SAE-feature interventions compared with random
  feature interventions.

The underlying reusable functions are in `utils/stat_utils/sae_statistics.py`
and `interpretability/sae/feature_analysis.py`.

Activation-based attribution is not causal proof. Use the intervention checks
and baselines before making causal claims about a feature.

## Building the Feature Dashboard

Use `examples/build_sae_feature_dashboard.ipynb` after training an SAE. It reads
a dataset, extracts token-level activations, encodes them with the SAE, and
writes dashboard artifacts to:

```text
data/sae_feature_dashboard/
```

Generated files:

- `dashboard_metadata.json`
- `feature_dashboard.csv`
- `feature_concept_scores.csv`
- `feature_top_tokens.csv`
- `feature_top_examples.csv`
- `sample_token_attributions.csv`

Open the notebook:

```bash
jupyter lab examples/build_sae_feature_dashboard.ipynb
```

For a quick smoke run, keep `MAX_DATASET_ROWS = 16`. For a fuller dashboard,
set it to `None` and make sure the selected model, layer, and SAE checkpoint
match.

The frontend and backend use these saved dashboard files to show concept labels,
top activating tokens, feature summaries, and intervention candidates.

## Running the Backend

Start the FastAPI backend from the repository root:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/api/health
```

Main endpoints:

- `GET /api/health`
- `GET /api/defaults`
- `GET /api/feature-dashboard`
- `POST /api/generate-attributions`
- `POST /api/interventions`

The intervention endpoint uses only additive SAE steering: selected feature
ids are converted to decoder directions and added to the hooked hidden state.

Default local configuration:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- layer: `model.layers.15`
- SAE checkpoint: the first `.pt` file found in `<repo_root>/models`
- dashboard directory: `data/sae_feature_dashboard`

The backend loads the model and SAE lazily on the first inference request. The
SAE checkpoint must match the selected model and layer hidden size.
Generation prompts are formatted through the tokenizer chat template with the
default Qwen system prompt unless `generation.system_prompt` overrides it.

## Running the Interface

Start the React/Vite app:

```bash
cd interface
npm run dev
```

Open:

```text
http://localhost:5173
```

The interface calls `http://localhost:8000` by default. Override the backend URL
with:

```bash
VITE_LOUPE_API_URL=http://localhost:8000 npm run dev
```

Current UI workflows:

- configure model, SAE checkpoint, layer, and activation type;
- inspect token-level SAE attribution in the sandbox;
- load feature and concept labels from `data/sae_feature_dashboard`;
- compare baseline generation with SAE-intervened generation.

If the backend, model, checkpoint, or dashboard artifacts are unavailable, the
frontend may fall back to local mock data. Do not treat mock data as real SAE
evidence.

## Typical Workflow

1. Prepare or regenerate a CSV dataset in `data/`.
2. Train an SAE with `examples/train_sae.ipynb`.
3. Run statistical checks with `examples/sae_llm_end2end_statistics.ipynb`.
4. Build dashboard artifacts with `examples/build_sae_feature_dashboard.ipynb`.
5. Start the backend with `python -m uvicorn backend.main:app --reload --host
   0.0.0.0 --port 8000`.
6. Start the frontend from `interface/` with `npm run dev`.
7. Use the UI for sandbox attribution and intervention experiments.

## Research Notes

- SAE features are activation-based candidates until validated by statistical
  and intervention checks.
- Attention maps, SHAP, LIME, and generative explanations are supporting
  methods in this project, not replacements for SAE reconstruction,
  separability, and causal-selectivity analysis.
- Token-level attribution requires an SAE trained on token activations, not only
  mean-pooled sequence activations.
- Padding tokens must be masked out for statistics and token-level SAE training.
- For dissertation-level claims, record the model, layer, activation type,
  dataset, seed, learning rate, sparsity settings, checkpoint path, and all
  hypothesis-check thresholds.

## Development Checks

Useful lightweight checks:

```bash
python -m py_compile utils/inference_utils/llm.py
python -m py_compile interpretability/sae/sae.py
python -m py_compile utils/stat_utils/sae_statistics.py
```

Frontend production build:

```bash
cd interface
npm run build
```

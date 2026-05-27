# Loupe Backend

FastAPI backend for the React interface in `interface/`.

Endpoints:

- `GET /api/health`
- `GET /api/defaults`
- `GET /api/feature-dashboard`
- `POST /api/generate-attributions`
- `POST /api/interventions`

The backend uses the existing research modules:

- `utils.inference_utils.llm.LLM` for model loading, generation, hooks and SAE interventions;
- `interpretability.sae.sae.SAE` for `h -> z -> h_hat`, reconstruction metrics, sparsity metrics and latent interventions;
- `interpretability.sae.feature_analysis` semantics for token/concept-level feature attribution.

Run from the repository root:

```bash
python3 -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Model and SAE checkpoints are loaded lazily on the first inference request.
The default SAE checkpoint is selected dynamically from the repository
`models/` directory: the backend creates the directory if needed and uses the
first `.pt` file found there. If no checkpoint exists, the default response is
`There are no pretrained SAE for Qwen/Qwen2.5-0.5B-Instruct`, and inference
endpoints will fail until a real checkpoint path is provided.
Use a SAE checkpoint trained for the selected model/layer; checkpoints from a
different Qwen size can fail because hidden dimensions differ.

The feature dashboard endpoint reads saved artifacts from
`data/sae_feature_dashboard`. Generate them with
`examples/build_sae_feature_dashboard.ipynb`; if the directory or required CSV
files do not exist, the endpoint returns `404`.

The attribution endpoint returns activation-based SAE feature candidates, not
causal proof. Causal claims still require the statistical/intervention checks
from the experiment notebooks, including a baseline intervention comparison.

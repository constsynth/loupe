# Loupe Interface

React/Vite prototype for SAE-centered LLM interpretability workflows.

Implemented now:

- initial screen opens the model/SAE configuration workflow;
- navigation exposes model/SAE configuration, sandbox, and intervention views;
- editable model, SAE checkpoint, layer, and activation type configuration screen;
- token-level SAE activation viewer with hover activation values and feature labels;
- saved dashboard panel backed by `data/sae_feature_dashboard`;
- default lightweight model `Qwen/Qwen2.5-0.5B-Instruct` and configurable SAE checkpoint path;
- prompt input wired to the FastAPI backend with mock fallback;
- dashboard-derived concept interventions wired to the FastAPI backend with before/after comparison;
- additive SAE steering through decoder directions for the selected dashboard concept features;
- concept labels and intervention feature ids are loaded from saved dashboard artifacts instead of hardcoded presets;
- minimal interface using only the requested `#FAFBEA` to `#393B0C` palette and `Italian Old Antiqua`.

Not implemented yet:

- full attention-map visualization and attention/FFN-specific backend extraction;
- experiment export from the UI;
- faithfulness/consistency scoring inside the UI.

The UI calls `http://localhost:8000` by default. Override the backend base URL
with `VITE_LOUPE_API_URL`.

The sandbox falls back to local mock token attribution when the backend request
fails. Dashboard and intervention workflows require the backend plus saved
dashboard artifacts; missing dashboard files are shown as a UI error instead of
being treated as real SAE evidence.

Backend contract:

- `GET /api/feature-dashboard`
  - query: `top_features`, `top_concept_scores`, `top_tokens`, `top_examples`;
  - response: dashboard metadata, feature summaries, concept scores, top tokens, top examples.
- `POST /api/generate-attributions`
  - request: `prompt`, `model_name`, `sae_checkpoint_path`, `layer_name`, generation settings;
  - response: generated text, token list, top SAE features, activation values, and SAE metrics.
- `POST /api/interventions`
  - request: `prompt`, selected dashboard concepts, feature ids, additive intervention strength, token positions, and generation settings;
  - response: baseline generation, intervened generation, resolved feature ids, additive strength, and token positions.

Generation settings may include `system_prompt`; if omitted, the backend uses
the default Qwen system prompt and formats the prompt through the tokenizer chat
template before generation.

The activation type selector is currently UI configuration metadata. Backend
requests use the exact `layer_name`; selecting `attention` or `ffn` does not yet
switch to a specialized extraction path.

Run locally:

```bash
cd interface
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

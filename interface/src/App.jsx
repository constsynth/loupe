import { useEffect, useMemo, useState } from "react";

const DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct";
const DEFAULT_SAE = "";
const DEFAULT_LAYER = "model.layers.15";
const API_BASE_URL = import.meta.env.VITE_LOUPE_API_URL ?? "http://localhost:8000";
const DASHBOARD_ADD_STRENGTH = 1.0;

const starterTokens = [
  token("Sparse", 0.82, 42, "feature decomposition"),
  token("autoencoders", 0.94, 118, "latent reconstruction"),
  token("highlight", 0.48, 271, "attention overlap"),
  token("semantic", 0.73, 904, "concept direction"),
  token("features", 0.88, 612, "SAE latent"),
  token("inside", 0.25, 73, "low attribution"),
  token("Qwen", 0.67, 1544, "model family"),
  token("responses.", 0.51, 418, "generation summary"),
];

const views = [
  {
    id: "config",
    title: "Model and SAE configuration",
    kicker: "Model · SAE · Layer",
  },
  {
    id: "sandbox",
    title: "Sandbox",
    kicker: "Prompt · Token attribution · Features",
  },
  {
    id: "intervention",
    title: "Modify model behavior",
    kicker: "Concepts · Before / After",
  },
];

function token(text, activation, featureId, featureLabel) {
  return {
    text,
    activation,
    featureId,
    featureLabel,
  };
}

function pseudoRandom(seed) {
  let value = seed % 2147483647;
  return () => {
    value = (value * 16807) % 2147483647;
    return (value - 1) / 2147483646;
  };
}

function buildTokenAttributions(text) {
  const source = text.trim()
    ? text
    : "Sparse autoencoders reveal latent features in transformer activations.";
  const words = source.match(/[\p{L}0-9._/-]+|[^\s]/gu) ?? [];
  const seed = [...source].reduce((sum, char) => sum + char.charCodeAt(0), 17);
  const next = pseudoRandom(seed);
  const labels = [
    "semantic cluster",
    "instruction style",
    "entity mention",
    "reasoning marker",
    "risk cue",
    "technical term",
  ];

  return words.slice(0, 42).map((word, index) => {
    const activation = Math.max(0.12, Math.min(0.98, next() * 0.86 + 0.12));
    const featureId = Math.floor(next() * 2200) + 1;
    return token(word, activation, featureId, labels[index % labels.length]);
  });
}

function buildFeatureSummary(tokens) {
  const grouped = new Map();
  tokens.forEach((item) => {
    const existing = grouped.get(item.featureId) ?? {
      featureId: item.featureId,
      featureLabel: item.featureLabel,
      peak: 0,
      tokens: [],
    };
    existing.peak = Math.max(existing.peak, item.activation);
    existing.tokens.push(item.text);
    grouped.set(item.featureId, existing);
  });

  return [...grouped.values()]
    .sort((a, b) => b.peak - a.peak)
    .slice(0, 5);
}

function getMockResponse(prompt) {
  const normalized = prompt.trim();
  const generated = normalized
    ? `SAE analysis shows that the request most strongly activates features for terminology, explanation structure, and fact-checking. For a complete result, the backend should return token-level hidden states, latent activations, and reconstruction error.`
    : `Enter a prompt to get token highlighting and attribution to the most active SAE features. The interface is currently showing a demo layer.`;

  const tokens = buildTokenAttributions(`${normalized} ${generated}`);
  return {
    text: generated,
    tokens,
    features: buildFeatureSummary(tokens),
  };
}

function modelNameForApi(modelName) {
  return modelName.includes("·") ? modelName.split("·").pop().trim() : modelName.trim();
}

function conceptsForApi(concepts) {
  return concepts.map((concept) => ({
    id: concept.id,
    name: concept.name,
    feature_ids: concept.featureIds,
    strength: concept.strength,
  }));
}

async function fetchBackendDefaults() {
  const response = await fetch(`${API_BASE_URL}/api/defaults`);

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail ?? `Defaults request failed: ${response.status}`);
  }

  return response.json();
}

function cleanDisplayToken(value) {
  if (value === null || value === undefined) {
    return value;
  }

  const cleaned = String(value)
    .replaceAll("Ġ", " ")
    .replaceAll("▁", " ")
    .replaceAll("Ċ", "\n")
    .replaceAll("ĉ", "\t")
    .trim();
  return cleaned || String(value);
}

function cleanDisplayContext(value) {
  if (value === null || value === undefined) {
    return value;
  }

  const cleaned = String(value)
    .replaceAll("Ġ", " ")
    .replaceAll("▁", " ")
    .replaceAll("Ċ", "\n")
    .replaceAll("ĉ", "\t")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || String(value);
}

function buildDashboardConceptPresets(dashboard) {
  if (!dashboard?.conceptScores?.length) {
    return [];
  }

  const grouped = new Map();
  dashboard.conceptScores.forEach((score) => {
    if (!score.conceptLabel || score.featureId === undefined || score.featureId === null) {
      return;
    }

    const existing = grouped.get(score.conceptLabel) ?? {
      id: score.conceptLabel,
      name: score.conceptLabel,
      featureScores: [],
      strength: DASHBOARD_ADD_STRENGTH,
    };
    existing.featureScores.push({
      featureId: Number(score.featureId),
      score: toNumber(score.score),
    });
    grouped.set(score.conceptLabel, existing);
  });

  return [...grouped.values()]
    .map((concept) => {
      const featureScores = concept.featureScores
        .sort((a, b) => b.score - a.score)
        .filter(
          (item, index, items) =>
            items.findIndex((candidate) => candidate.featureId === item.featureId) === index,
        )
        .slice(0, 5);
      return {
        ...concept,
        featureScores,
        featureIds: featureScores.map((item) => item.featureId),
      };
    })
    .filter((concept) => concept.featureIds.length > 0)
    .sort((a, b) => a.name.localeCompare(b.name));
}

function setFeatureLabel(lookup, featureId, label) {
  if (featureId === undefined || featureId === null || !label || lookup.has(Number(featureId))) {
    return;
  }
  lookup.set(Number(featureId), label);
}

function buildDashboardFeatureLookup(dashboard) {
  const lookup = new Map();
  if (!dashboard) {
    return lookup;
  }

  [...(dashboard.conceptScores ?? [])]
    .sort((a, b) => toNumber(b.score) - toNumber(a.score))
    .forEach((score) => {
      setFeatureLabel(lookup, score.featureId, score.conceptLabel);
    });

  [...(dashboard.features ?? [])]
    .sort((a, b) => toNumber(b.maxActivation) - toNumber(a.maxActivation))
    .forEach((feature) => {
      setFeatureLabel(lookup, feature.featureId, feature.topConcepts?.[0]?.conceptLabel);
    });

  [...(dashboard.topTokens ?? [])]
    .sort((a, b) => toNumber(b.activation) - toNumber(a.activation))
    .forEach((item) => {
      setFeatureLabel(lookup, item.featureId, item.focusConcept ?? item.conceptLabel);
    });

  [...(dashboard.topExamples ?? [])]
    .sort((a, b) => toNumber(b.activation) - toNumber(a.activation))
    .forEach((item) => {
      setFeatureLabel(lookup, item.featureId, item.conceptLabel);
    });

  return lookup;
}

function featureLabelFromDashboard(featureId, apiLabel, dashboardFeatureLookup) {
  return (
    apiLabel ||
    (dashboardFeatureLookup.get(Number(featureId)) ??
    "feature not found in saved dashboard"
    )
  );
}

function normalizeApiAttribution(response, dashboardFeatureLookup = new Map()) {
  return {
    text: response.text,
    tokens: response.tokens.map((item) => ({
      text: cleanDisplayToken(item.text),
      activation: item.activation,
      rawActivation: item.raw_activation,
      featureId: item.feature_id,
      featureLabel: featureLabelFromDashboard(
        item.feature_id,
        item.concept_label ?? item.concept_id,
        dashboardFeatureLookup,
      ),
    })),
    features: response.features.map((item) => ({
      featureId: item.feature_id,
      featureLabel: featureLabelFromDashboard(
        item.feature_id,
        item.concept_label ?? item.concept_id,
        dashboardFeatureLookup,
      ),
      peak: item.activation,
    })),
  };
}

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatNumber(value, digits = 3) {
  return toNumber(value).toFixed(digits);
}

function formatPercent(value) {
  return `${Math.round(toNumber(value) * 100)}%`;
}

function normalizeDashboardResponse(response) {
  return {
    dashboardDir: response.dashboard_dir,
    metadata: response.metadata ?? {},
    methodNote: response.method_note,
    sampleTokenAttributionsCount: response.sample_token_attributions_count ?? 0,
    features: (response.features ?? []).map((item) => ({
      featureId: item.feature_index,
      meanActivation: toNumber(item.mean_activation),
      maxActivation: toNumber(item.max_activation),
      activationDensity: toNumber(item.activation_density),
      topConcepts: (item.top_concepts ?? []).map((concept) => ({
        conceptLabel: concept.concept_label,
        score: toNumber(concept.score),
        meanInside: concept.mean_inside,
        meanOutside: concept.mean_outside,
      })),
    })),
    conceptScores: (response.concept_scores ?? []).map((item) => ({
      conceptLabel: item.concept_label,
      featureId: item.feature_index,
      score: toNumber(item.score),
      scoreMethod: item.score_method,
      meanInside: item.mean_inside,
      meanOutside: item.mean_outside,
      activationRateInside: item.activation_rate_inside,
      activationRateOutside: item.activation_rate_outside,
    })),
    topTokens: (response.top_tokens ?? []).map((item) => ({
      featureId: item.feature_index,
      sampleId: item.sample_id,
      sampleIndex: item.sample_index,
      tokenPosition: item.token_position,
      tokenText: cleanDisplayToken(item.token_text),
      activation: toNumber(item.activation),
      conceptLabel: item.concept_label,
      focusConcept: item.focus_concept,
      leftContext: cleanDisplayContext(item.left_context),
      rightContext: cleanDisplayContext(item.right_context),
      text: item.text,
      featureScore: item.feature_score,
    })),
    topExamples: (response.top_examples ?? []).map((item) => ({
      featureId: item.feature_index,
      sampleId: item.sample_id,
      sampleIndex: item.sample_index,
      activation: toNumber(item.activation),
      conceptLabel: item.concept_label,
      text: item.text,
    })),
  };
}

async function fetchBackendAttribution(prompt, config, dashboardFeatureLookup) {
  const response = await fetch(`${API_BASE_URL}/api/generate-attributions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      model_name: modelNameForApi(config.modelName),
      sae_checkpoint_path: config.saePath,
      layer_name: config.layerName,
      top_k_features: 10,
      top_k_token_features: 3,
      concepts: [],
      generation: {
        max_new_tokens: 128,
        max_length: 512,
        do_sample: false,
      },
    }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail ?? `Backend request failed: ${response.status}`);
  }

  return normalizeApiAttribution(await response.json(), dashboardFeatureLookup);
}

async function fetchFeatureDashboard() {
  const params = new URLSearchParams({
    top_features: "16",
    top_concept_scores: "20",
    top_tokens: "48",
    top_examples: "24",
  });
  const response = await fetch(`${API_BASE_URL}/api/feature-dashboard?${params.toString()}`);

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail ?? `Dashboard request failed: ${response.status}`);
  }

  return normalizeDashboardResponse(await response.json());
}

async function fetchBackendIntervention(
  prompt,
  config,
  selectedConcepts,
  conceptPresets,
  interventionStrength,
) {
  const selected = conceptPresets.filter((concept) => selectedConcepts.includes(concept.id));
  const response = await fetch(`${API_BASE_URL}/api/interventions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      model_name: modelNameForApi(config.modelName),
      sae_checkpoint_path: config.saePath,
      layer_name: config.layerName,
      concepts: conceptsForApi(selected),
      strength: interventionStrength,
      token_positions: null,
      generation: {
        max_new_tokens: 128,
        max_length: 512,
        do_sample: false,
      },
    }),
  });

  if (!response.ok) {
    const errorPayload = await response.json().catch(() => ({}));
    throw new Error(errorPayload.detail ?? `Backend request failed: ${response.status}`);
  }

  return response.json();
}

function useTypingText(text, speed = 18) {
  const [visible, setVisible] = useState("");

  useEffect(() => {
    setVisible("");
    if (!text) {
      return undefined;
    }

    let index = 0;
    const timer = window.setInterval(() => {
      index += 1;
      setVisible(text.slice(0, index));
      if (index >= text.length) {
        window.clearInterval(timer);
      }
    }, speed);

    return () => window.clearInterval(timer);
  }, [text, speed]);

  return visible;
}

function ActivationToken({ item }) {
  const activationClass =
    item.activation > 0.82
      ? "is-peak"
      : item.activation > 0.62
        ? "is-high"
        : item.activation > 0.38
          ? "is-mid"
          : "is-low";

  return (
    <span
      className={`activation-token ${activationClass}`}
      tabIndex="0"
    >
      {item.text}
      <span className="token-tooltip" role="tooltip">
        Feature {item.featureId}
        <small>{item.featureLabel}</small>
        <strong>{item.activation.toFixed(3)}</strong>
        {item.rawActivation !== undefined && item.rawActivation !== null ? (
          <small>raw {formatNumber(item.rawActivation)}</small>
        ) : null}
      </span>
    </span>
  );
}

function TokenViewer({ tokens }) {
  return (
    <div className="token-field" aria-label="Token attributions">
      {tokens.map((item, index) => (
        <ActivationToken item={item} key={`${item.text}-${index}`} />
      ))}
    </div>
  );
}

function FeatureList({ features }) {
  return (
    <div className="feature-list">
      {features.map((feature) => (
        <article className="feature-row" key={feature.featureId}>
          <div>
            <span>Feature {feature.featureId}</span>
            <small>{feature.featureLabel}</small>
          </div>
          <div className="feature-meter" aria-label={`Activation ${feature.peak.toFixed(2)}`}>
            <span style={{ width: `${Math.round(feature.peak * 100)}%` }} />
          </div>
          <strong>{feature.peak.toFixed(2)}</strong>
        </article>
      ))}
    </div>
  );
}

function DashboardStatus({ dashboard, error, isLoading, onReload }) {
  if (error) {
    return (
      <div className="dashboard-status is-error">
        <strong>Dashboard unavailable</strong>
        <span>{error}</span>
        <button className="reload-button" onClick={onReload} type="button">
          Retry
        </button>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="dashboard-status">
        <strong>Loading dashboard</strong>
        <span>Reading saved CSV/JSON artifacts from the backend.</span>
      </div>
    );
  }

  if (!dashboard) {
    return (
      <div className="dashboard-status">
        <strong>Dashboard has not been loaded yet</strong>
        <span>Open the Sandbox or click reload after generating artifacts.</span>
      </div>
    );
  }

  return (
    <div className="dashboard-status is-ready">
      <strong>{dashboard.features.length} features</strong>
      <span>{dashboard.dashboardDir}</span>
    </div>
  );
}

function DashboardMetrics({ dashboard }) {
  const metadata = dashboard.metadata;
  const metrics = [
    ["Model", metadata.model_name_or_path],
    ["Layer", metadata.layer_name],
    ["Samples", metadata.n_samples],
    ["Tokens", metadata.n_valid_tokens],
    ["Latent", metadata.latent_size],
    ["Score", metadata.score_method],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");

  return (
    <div className="metric-strip">
      {metrics.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
      <div>
        <span>Token rows</span>
        <strong>{dashboard.sampleTokenAttributionsCount}</strong>
      </div>
    </div>
  );
}

function DashboardFeatureCards({ features }) {
  if (!features.length) {
    return <p className="dashboard-empty">Feature summary is empty.</p>;
  }

  return (
    <div className="dashboard-feature-grid">
      {features.map((feature) => (
        <article className="dashboard-feature-card" key={feature.featureId}>
          <header>
            <span>Feature {feature.featureId}</span>
            <strong>{formatNumber(feature.maxActivation)}</strong>
          </header>
          <div className="feature-meter" aria-label={`Density ${formatPercent(feature.activationDensity)}`}>
            <span style={{ width: formatPercent(feature.activationDensity) }} />
          </div>
          <small>
            density {formatPercent(feature.activationDensity)} · mean{" "}
            {formatNumber(feature.meanActivation)}
          </small>
          <ul>
            {feature.topConcepts.slice(0, 3).map((concept) => (
              <li key={`${feature.featureId}-${concept.conceptLabel}`}>
                <span>{concept.conceptLabel}</span>
                <strong>{formatNumber(concept.score)}</strong>
              </li>
            ))}
          </ul>
        </article>
      ))}
    </div>
  );
}

function DashboardConceptScores({ scores }) {
  if (!scores.length) {
    return <p className="dashboard-empty">Concept scores are empty.</p>;
  }

  return (
    <div className="dashboard-table-wrap">
      <table className="dashboard-table">
        <thead>
          <tr>
            <th>Concept</th>
            <th>Feature</th>
            <th>Score</th>
            <th>Inside / outside</th>
          </tr>
        </thead>
        <tbody>
          {scores.map((score, index) => (
            <tr key={`${score.conceptLabel}-${score.featureId}-${index}`}>
              <td>{score.conceptLabel}</td>
              <td>{score.featureId}</td>
              <td>{formatNumber(score.score)}</td>
              <td>
                {score.meanInside !== null && score.meanInside !== undefined
                  ? formatNumber(score.meanInside)
                  : "n/a"}{" "}
                /{" "}
                {score.meanOutside !== null && score.meanOutside !== undefined
                  ? formatNumber(score.meanOutside)
                  : "n/a"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DashboardTopTokens({ tokens }) {
  if (!tokens.length) {
    return <p className="dashboard-empty">Top tokens are empty.</p>;
  }

  return (
    <div className="dashboard-token-list">
      {tokens.map((item, index) => (
        <article className="dashboard-token-row" key={`${item.featureId}-${item.sampleId}-${item.tokenPosition}-${index}`}>
          <header>
            <span>{item.tokenText ?? "<token>"}</span>
            <strong>{formatNumber(item.activation)}</strong>
          </header>
          <small>
            Feature {item.featureId} · {item.focusConcept ?? item.conceptLabel ?? "unknown"}
            {item.sampleId ? ` · ${item.sampleId}` : ""}
          </small>
          <p>
            {item.leftContext ? `${item.leftContext} ` : ""}
            <mark>{item.tokenText ?? "<token>"}</mark>
            {item.rightContext ? ` ${item.rightContext}` : ""}
          </p>
        </article>
      ))}
    </div>
  );
}

function DashboardTopExamples({ examples }) {
  if (!examples.length) {
    return <p className="dashboard-empty">Top examples are empty.</p>;
  }

  return (
    <div className="dashboard-example-list">
      {examples.map((item, index) => (
        <article className="dashboard-example" key={`${item.featureId}-${item.sampleId}-${index}`}>
          <header>
            <span>Feature {item.featureId}</span>
            <strong>{formatNumber(item.activation)}</strong>
          </header>
          <small>
            {item.conceptLabel ?? "unknown"}
            {item.sampleId ? ` · ${item.sampleId}` : ""}
          </small>
          <p>{item.text ?? "No text saved for this example."}</p>
        </article>
      ))}
    </div>
  );
}

function FeatureDashboardPanel({ dashboard, isLoading, error, onReload }) {
  return (
    <section className="panel dashboard-panel standalone-panel">
      <div className="dashboard-title-row">
        <div className="section-heading">
          <p>Saved dashboard</p>
          <h2>Feature microscope</h2>
        </div>
        <button className="reload-button" disabled={isLoading} onClick={onReload} type="button">
          {isLoading ? "Loading" : "Reload"}
        </button>
      </div>

      <DashboardStatus
        dashboard={dashboard}
        error={error}
        isLoading={isLoading}
        onReload={onReload}
      />

      {dashboard ? (
        <>
          <DashboardMetrics dashboard={dashboard} />
          <p className="method-note">{dashboard.methodNote}</p>

          <div className="dashboard-grid">
            <section className="dashboard-block">
              <div className="section-heading">
                <p>Features</p>
                <h2>Top SAE features</h2>
              </div>
              <DashboardFeatureCards features={dashboard.features} />
            </section>

            <section className="dashboard-block">
              <div className="section-heading">
                <p>Concept attribution</p>
                <h2>Concept scores</h2>
              </div>
              <DashboardConceptScores scores={dashboard.conceptScores} />
            </section>

            <section className="dashboard-block">
              <div className="section-heading">
                <p>Token level</p>
                <h2>Top activating tokens</h2>
              </div>
              <DashboardTopTokens tokens={dashboard.topTokens} />
            </section>

            <section className="dashboard-block">
              <div className="section-heading">
                <p>Dataset examples</p>
                <h2>Top examples</h2>
              </div>
              <DashboardTopExamples examples={dashboard.topExamples} />
            </section>
          </div>
        </>
      ) : null}
    </section>
  );
}

function InterventionPanel({
  config,
  prompt,
  selectedConcepts,
  setPrompt,
  setSelectedConcepts,
  conceptPresets,
  dashboard,
  dashboardError,
  isDashboardLoading,
  onReloadDashboard,
}) {
  const [isIntervening, setIsIntervening] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [interventionStrength, setInterventionStrength] = useState(1.5);
  const selected = useMemo(
    () => conceptPresets.filter((concept) => selectedConcepts.includes(concept.id)),
    [conceptPresets, selectedConcepts],
  );

  const fallbackBefore =
    "The model answers directly and preserves the original balance of features extracted by the SAE at the selected layer.";
  const fallbackAfter = selected.length
    ? `After the intervention, the modified concepts are: ${selected.map((concept) => concept.name).join(", ")}. The selected SAE decoder directions are added to the hidden state before generation continues.`
    : "Select one or more concepts from the saved SAE dashboard to see a demonstration intervention result.";
  const before = comparison?.baseline_text ?? fallbackBefore;
  const after = comparison?.intervened_text ?? fallbackAfter;

  const typedAfter = useTypingText(after, 14);

  function toggleConcept(id) {
    setSelectedConcepts((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  async function runIntervention() {
    if (!selectedConcepts.length) {
      setComparison(null);
      return;
    }

    setIsIntervening(true);
    try {
      setComparison(await fetchBackendIntervention(
        prompt,
        config,
        selectedConcepts,
        conceptPresets,
        interventionStrength,
      ));
    } catch (error) {
      console.warn(error);
      setComparison(null);
    } finally {
      setIsIntervening(false);
    }
  }

  return (
    <section className="panel intervention-panel">
      <div className="section-heading">
        <p>Intervention</p>
        <h2>Dashboard concepts</h2>
      </div>

      <DashboardStatus
        dashboard={dashboard}
        error={dashboardError}
        isLoading={isDashboardLoading}
        onReload={onReloadDashboard}
      />

      <label>
        <span>Prompt</span>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          rows="5"
          aria-label="Intervention prompt"
        />
      </label>

      <label>
        <span>Intervention strength · {formatNumber(interventionStrength, 2)}</span>
        <input
          type="range"
          min="0"
          max="2"
          step="0.05"
          value={interventionStrength}
          onChange={(event) => setInterventionStrength(Number(event.target.value))}
        />
      </label>

      <div className="concept-grid">
        {conceptPresets.length ? conceptPresets.map((concept) => {
          const active = selectedConcepts.includes(concept.id);
          return (
            <button
              className={`concept-button ${active ? "is-active" : ""}`}
              key={concept.id}
              onClick={() => toggleConcept(concept.id)}
              type="button"
            >
              <span>{concept.name}</span>
              <small>
                add · {concept.featureScores
                  .map((item) => `${item.featureId}:${formatNumber(item.score, 2)}`)
                  .join(", ")}
              </small>
            </button>
          );
        }) : (
          <p className="dashboard-empty">
            Concepts are not loaded. `data/sae_feature_dashboard/feature_concept_scores.csv` is required.
          </p>
        )}
      </div>

      <button
        className="primary-action"
        disabled={isIntervening || selectedConcepts.length === 0}
        onClick={runIntervention}
        type="button"
      >
        {isIntervening ? "Intervening" : "Run intervention"}
      </button>

      <div className="comparison-grid">
        <article className="comparison-pane">
          <span>Before</span>
          <p>{before}</p>
        </article>
        <article className="comparison-pane is-after">
          <span>After</span>
          <p>{typedAfter}<i aria-hidden="true" /></p>
        </article>
      </div>
    </section>
  );
}

function Header({ activeView, setActiveView }) {
  return (
    <header className="topbar">
      <button
        className="brand-button"
        onClick={() => setActiveView("home")}
        type="button"
      >
        <span>Loupe</span>
        <strong>Home</strong>
      </button>
      {activeView !== "home" ? (
        <nav className="view-nav" aria-label="Workspace navigation">
          {views.map((view) => (
            <button
              className={activeView === view.id ? "is-active" : ""}
              key={view.id}
              onClick={() => setActiveView(view.id)}
              type="button"
            >
              {view.title}
            </button>
          ))}
        </nav>
      ) : null}
    </header>
  );
}

function ModelStack({ config }) {
  return (
    <div className="model-stack" aria-label="Default model settings">
      <span>{config.modelName}</span>
      <small>{config.saePath || "SAE checkpoint is not selected"}</small>
    </div>
  );
}

function HomeView({ config, setActiveView }) {
  return (
    <section className="home-view">
      <div className="home-title">
        <h1>Select a workspace mode</h1>
      </div>

      <div className="path-grid">
        {views.map((view, index) => (
          <button
            className="path-card"
            key={view.id}
            onClick={() => setActiveView(view.id)}
            type="button"
            style={{ animationDelay: `${index * 80}ms` }}
          >
            <small>{view.kicker}</small>
            <span>{view.title}</span>
          </button>
        ))}
      </div>

      <ModelStack config={config} />
    </section>
  );
}

function ConfigView({ config, setConfig }) {
  function updateConfig(key, value) {
    setConfig((current) => ({ ...current, [key]: value }));
  }

  return (
    <section className="screen-grid config-grid">
      <section className="panel config-panel">
        <div className="section-heading">
          <p>Configuration</p>
          <h2>Model and SAE</h2>
        </div>
        <label>
          <span>Model</span>
          <input
            value={config.modelName}
            onChange={(event) => updateConfig("modelName", event.target.value)}
          />
        </label>
        <label>
          <span>SAE checkpoint</span>
          <input
            value={config.saePath}
            onChange={(event) => updateConfig("saePath", event.target.value)}
          />
        </label>
      </section>

      <section className="panel config-panel">
        <div className="section-heading">
          <p>Activation source</p>
          <h2>Layer and mode</h2>
        </div>
        <label>
          <span>Layer name</span>
          <input
            value={config.layerName}
            onChange={(event) => updateConfig("layerName", event.target.value)}
          />
        </label>
        <label>
          <span>Activation type</span>
          <select
            value={config.activationType}
            onChange={(event) => updateConfig("activationType", event.target.value)}
          >
            <option value="residual">residual</option>
            <option value="attention">attention</option>
            <option value="ffn">ffn</option>
          </select>
        </label>
      </section>

      <section className="panel config-summary">
        <div className="section-heading">
          <p>Selected</p>
          <h2>Current configuration</h2>
        </div>
        <dl>
          <div>
            <dt>Model</dt>
            <dd>{config.modelName}</dd>
          </div>
          <div>
            <dt>SAE</dt>
            <dd>{config.saePath}</dd>
          </div>
          <div>
            <dt>Layer</dt>
            <dd>{config.layerName}</dd>
          </div>
          <div>
            <dt>Type</dt>
            <dd>{config.activationType}</dd>
          </div>
        </dl>
      </section>
    </section>
  );
}

function SandboxView({
  prompt,
  setPrompt,
  isRunning,
  runAnalysis,
  analysis,
  typedAnswer,
  dashboard,
  dashboardError,
  isDashboardLoading,
  onReloadDashboard,
}) {
  return (
    <>
      <section className="workspace">
        <form className="panel prompt-panel" onSubmit={runAnalysis}>
          <div className="section-heading">
            <p>Sandbox</p>
            <h2>Model request</h2>
          </div>
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows="7"
            aria-label="Prompt"
          />
          <button className="primary-action" disabled={isRunning} type="submit">
            {isRunning ? "Analyzing" : "Run SAE attribution"}
          </button>
        </form>

        <section className="panel response-panel">
          <div className="section-heading">
            <p>Response</p>
            <h2>Token attribution</h2>
          </div>
          <p className="typed-response">
            {typedAnswer}
            <i aria-hidden="true" />
          </p>
          <TokenViewer tokens={analysis.tokens} />
        </section>
      </section>

      <section className="panel features-panel standalone-panel">
        <div className="section-heading">
          <p>SAE</p>
          <h2>Top active features</h2>
        </div>
        <FeatureList features={analysis.features} />
      </section>

      <FeatureDashboardPanel
        dashboard={dashboard}
        error={dashboardError}
        isLoading={isDashboardLoading}
        onReload={onReloadDashboard}
      />
    </>
  );
}

function App() {
  const [activeView, setActiveView] = useState("config");
  const [prompt, setPrompt] = useState(
    "Why do sparse autoencoders help interpret internal LLM features?",
  );
  const [isRunning, setIsRunning] = useState(false);
  const [selectedConcepts, setSelectedConcepts] = useState([]);
  const [config, setConfig] = useState({
    modelName: DEFAULT_MODEL,
    saePath: DEFAULT_SAE,
    layerName: DEFAULT_LAYER,
    activationType: "residual",
  });
  const [analysis, setAnalysis] = useState(() => ({
    text: "SAE analysis shows active tokens and the strongest latent features.",
    tokens: starterTokens,
    features: buildFeatureSummary(starterTokens),
  }));
  const [dashboard, setDashboard] = useState(null);
  const [dashboardError, setDashboardError] = useState("");
  const [isDashboardLoading, setIsDashboardLoading] = useState(false);

  const typedAnswer = useTypingText(analysis.text, 16);
  const dashboardConceptPresets = useMemo(
    () => buildDashboardConceptPresets(dashboard),
    [dashboard],
  );
  const dashboardFeatureLookup = useMemo(
    () => buildDashboardFeatureLookup(dashboard),
    [dashboard],
  );

  useEffect(() => {
    let cancelled = false;

    async function loadDefaults() {
      try {
        const defaults = await fetchBackendDefaults();
        if (cancelled) {
          return;
        }
        setConfig((current) => ({
          ...current,
          modelName: defaults.model_name ?? current.modelName,
          saePath: defaults.sae_checkpoint_path ?? current.saePath,
          layerName: defaults.layer_name ?? current.layerName,
        }));
      } catch (error) {
        console.warn(error);
      }
    }

    loadDefaults();

    return () => {
      cancelled = true;
    };
  }, []);

  async function loadDashboard() {
    setIsDashboardLoading(true);
    setDashboardError("");
    try {
      const loadedDashboard = await fetchFeatureDashboard();
      setDashboard(loadedDashboard);
      return loadedDashboard;
    } catch (error) {
      console.warn(error);
      setDashboard(null);
      setDashboardError(error.message);
      return null;
    } finally {
      setIsDashboardLoading(false);
    }
  }

  useEffect(() => {
    if (
      (activeView === "sandbox" || activeView === "intervention") &&
      dashboard === null &&
      !dashboardError &&
      !isDashboardLoading
    ) {
      loadDashboard();
    }
  }, [activeView]);

  useEffect(() => {
    if (!dashboardConceptPresets.length) {
      setSelectedConcepts([]);
      return;
    }

    setSelectedConcepts((current) => {
      const validIds = new Set(dashboardConceptPresets.map((concept) => concept.id));
      const stillValid = current.filter((id) => validIds.has(id));
      return stillValid.length ? stillValid : [dashboardConceptPresets[0].id];
    });
  }, [dashboardConceptPresets]);

  async function runAnalysis(event) {
    event.preventDefault();
    setIsRunning(true);
    try {
      const activeDashboard = dashboard ?? await loadDashboard();
      const activeFeatureLookup = activeDashboard
        ? buildDashboardFeatureLookup(activeDashboard)
        : dashboardFeatureLookup;
      setAnalysis(await fetchBackendAttribution(prompt, config, activeFeatureLookup));
    } catch (error) {
      console.warn(error);
      setAnalysis(getMockResponse(prompt));
    } finally {
      setIsRunning(false);
    }
  }

  return (
    <main className="app-shell">
      <Header activeView={activeView} setActiveView={setActiveView} />

      {activeView === "home" ? (
        <HomeView config={config} setActiveView={setActiveView} />
      ) : null}
      {activeView === "config" ? (
        <ConfigView config={config} setConfig={setConfig} />
      ) : null}
      {activeView === "sandbox" ? (
        <SandboxView
          analysis={analysis}
          isRunning={isRunning}
          prompt={prompt}
          runAnalysis={runAnalysis}
          setPrompt={setPrompt}
          typedAnswer={typedAnswer}
          dashboard={dashboard}
          dashboardError={dashboardError}
          isDashboardLoading={isDashboardLoading}
          onReloadDashboard={loadDashboard}
        />
      ) : null}
      {activeView === "intervention" ? (
        <InterventionPanel
          config={config}
          prompt={prompt}
          selectedConcepts={selectedConcepts}
          setPrompt={setPrompt}
          setSelectedConcepts={setSelectedConcepts}
          conceptPresets={dashboardConceptPresets}
          dashboard={dashboard}
          dashboardError={dashboardError}
          isDashboardLoading={isDashboardLoading}
          onReloadDashboard={loadDashboard}
        />
      ) : null}
    </main>
  );
}

export default App;

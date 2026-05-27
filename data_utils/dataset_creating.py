import json
import random
import re
import typing as tp
from pathlib import Path

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_community.llms.fake import FakeListLLM
from sklearn.model_selection import train_test_split
from tqdm import tqdm


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "deepseek/deepseek-v4-flash"

DATASET_COLUMNS = [
    "sample_id",
    "dataset_theme",
    "text",
    "expected_answer",
    "concept_label",
    "concept_description",
    "domain",
    "prompt_type",
    "language",
    "target_behavior",
    "target_token",
    "intervention_feature_hypothesis",
    "statistical_checks",
    "metadata_json",
]

STATISTICAL_CHECKS = [
    "reconstruction_nmse",
    "cosine_similarity_original_reconstruction",
    "distribution_similarity_mmd_ks_wasserstein_jsd",
    "latent_sparsity_l0_active_share_hoyer_entropy",
    "concept_separability_jsd_sep",
    "causal_selectivity_q",
]

TARGET_TOKEN_PLACEHOLDERS = {
    "answer",
    "category",
    "correct_fact",
    "label",
    "neutral_answer",
    "numeric_answer",
    "sentiment_label",
    "target_marker",
    "target_token",
    "token",
}

DEFAULT_CONCEPTS = [
    {
        "label": "factual_recall",
        "description": "Prompts that require factual retrieval and stable entity-property associations.",
        "domain": "general_knowledge",
        "prompt_type": "short_answer",
        "target_behavior": "increase_probability_of_correct_fact_token",
        "target_token": "correct_fact",
        "feature_hypothesis": "SAE features should isolate entity, date, and property directions.",
    },
    {
        "label": "mathematical_reasoning",
        "description": "Prompts with arithmetic or symbolic reasoning and a verifiable answer.",
        "domain": "mathematics",
        "prompt_type": "reasoning",
        "target_behavior": "increase_probability_of_correct_numeric_token",
        "target_token": "numeric_answer",
        "feature_hypothesis": "SAE features should separate numerical operations from surface wording.",
    },
    {
        "label": "causal_reasoning",
        "description": "Prompts that require identifying cause-effect relations.",
        "domain": "logic",
        "prompt_type": "explanation",
        "target_behavior": "increase_causal_explanation_consistency",
        "target_token": "because",
        "feature_hypothesis": "SAE features should activate on causal connectors and causal structure.",
    },
    {
        "label": "medical_safety",
        "description": "Prompts about medical advice that require cautious and safety-aware answers.",
        "domain": "medicine",
        "prompt_type": "safety_sensitive",
        "target_behavior": "increase_refusal_or_caution_for_unsafe_advice",
        "target_token": "doctor",
        "feature_hypothesis": "SAE features should capture medical safety and uncertainty contexts.",
    },
    {
        "label": "legal_reasoning",
        "description": "Prompts about legal interpretation that require caveats and jurisdiction awareness.",
        "domain": "law",
        "prompt_type": "safety_sensitive",
        "target_behavior": "increase_legal_caveat_probability",
        "target_token": "jurisdiction",
        "feature_hypothesis": "SAE features should distinguish legal concepts from generic advice.",
    },
    {
        "label": "bias_sensitive",
        "description": "Prompts designed to expose stereotype-sensitive language model behavior.",
        "domain": "ethics",
        "prompt_type": "bias_probe",
        "target_behavior": "reduce_stereotype_completion_probability",
        "target_token": "neutral_answer",
        "feature_hypothesis": "SAE features should help identify social-bias sensitive directions.",
    },
    {
        "label": "sentiment",
        "description": "Prompts with positive, negative, or neutral sentiment cues.",
        "domain": "sentiment_analysis",
        "prompt_type": "classification",
        "target_behavior": "increase_correct_sentiment_label_probability",
        "target_token": "sentiment_label",
        "feature_hypothesis": "SAE features should separate sentiment-bearing tokens from topic tokens.",
    },
    {
        "label": "programming_debug",
        "description": "Prompts that ask for code diagnosis or small code corrections.",
        "domain": "programming",
        "prompt_type": "debugging",
        "target_behavior": "increase_probability_of_correct_bug_fix",
        "target_token": "fix",
        "feature_hypothesis": "SAE features should isolate syntax, error, and repair patterns.",
    },
]


GENERATION_PROMPT = PromptTemplate.from_template(
    """
You generate a synthetic dataset for SAE interpretability experiments in a transformer language model.

The generated dataset has the following characteristics:
- Dataset theme: {dataset_theme}
- Language: {language}
- Concept label: {concept_label}
- Concept description: {concept_description}
- Domain: {domain}
- Prompt type: {prompt_type}
- Target behavior for intervention tests: {target_behavior}
- Feature hypothesis: {feature_hypothesis}
- Number of samples: {n_samples}

Return only JSON Lines. Each line must be a valid JSON object with exactly these keys:
text, expected_answer, target_token, metadata_json.

Field requirements:
- text: the exact prompt later passed to the LLM for hidden-state extraction.
- expected_answer: the concise correct or desired answer for text.
- target_token: a concrete surface token that can be tracked in the generated answer
  or prompt. It must be an actual token such as "6", "because", "doctor", "fix",
  or "positive", not a category, placeholder, marker, or label such as
  "correct_fact", "numeric_answer", "target_marker", or "sentiment_label".
- metadata_json: a compact JSON object encoded as a string. It may include
  difficulty, ambiguity_level, expected_activation_focus, and intervention_note.

Do not return fields that are already provided by the concept configuration:
concept_label, concept_description, domain, prompt_type, language, target_behavior,
or intervention_feature_hypothesis.
Do not include markdown or comments.
"""
)


def sanitize_dataset_theme(dataset_theme: str) -> str:
    """Return a filesystem-safe dataset theme used in `<dataset_theme>_dataset.csv`."""
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", dataset_theme.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "synthetic"


def dataset_path(dataset_theme: str, output_dir: str | Path = "data") -> Path:
    """Return the canonical CSV path for a dataset theme."""
    return Path(output_dir) / f"{sanitize_dataset_theme(dataset_theme)}_dataset.csv"


def create_synthetic_dataset(
    dataset_theme: str,
    samples_per_concept: int = 8,
    output_dir: str | Path = "data",
    concepts: list[dict[str, str]] | None = None,
    llm: Runnable | None = None,
    openrouter_api_key: str | None = None,
    openrouter_model: str = OPENROUTER_MODEL_NAME,
    openrouter_api_base: str = OPENROUTER_API_BASE,
    temperature: float = 0.7,
    language: str = "ru",
    seed: int = 42,
    show_progress: bool = True,
) -> Path:
    """
    Generate a synthetic CSV dataset for SAE experiments.

    The dataset is saved as `<dataset_theme>_dataset.csv` and contains prompts plus
    labels needed for the statistical checks described in the dissertation notes:
    reconstruction quality, distribution similarity, sparsity, concept separability,
    and causal intervention selectivity.

    If `openrouter_api_key` is provided, generation uses OpenRouter through
    `langchain_openai.ChatOpenAI` with the default model
    `deepseek/deepseek-v4-flash`. If both `llm` and `openrouter_api_key` are not
    provided, a deterministic `FakeListLLM` from `langchain_community` is used
    so examples can run without external API keys. For custom generation, pass
    any LangChain-compatible LLM or Runnable through `llm`. Set
    `show_progress=False` to disable the tqdm progress bar.
    """
    if samples_per_concept < 1:
        raise ValueError("samples_per_concept must be at least 1")

    concepts = concepts or DEFAULT_CONCEPTS
    output_path = dataset_path(dataset_theme, output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if llm is None and openrouter_api_key:
        llm = _make_openrouter_llm(
            api_key=openrouter_api_key,
            model=openrouter_model,
            api_base=openrouter_api_base,
            temperature=temperature,
        )

    if llm is None:
        responses = [
            _make_offline_jsonl_response(
                dataset_theme=dataset_theme,
                concept=concept,
                samples_per_concept=samples_per_concept,
                language=language,
                seed=seed + idx,
            )
            for idx, concept in enumerate(concepts)
        ]
        llm = FakeListLLM(responses=responses)

    chain = GENERATION_PROMPT | llm | StrOutputParser()
    rows: list[dict[str, str]] = []

    concept_iterator = tqdm(
        concepts,
        desc="Generating synthetic dataset",
        unit="concept",
        disable=not show_progress,
    )

    for concept in concept_iterator:
        concept_iterator.set_postfix(concept=concept["label"])
        raw_output = chain.invoke(
            _make_generation_inputs(
                dataset_theme=dataset_theme,
                language=language,
                concept=concept,
                samples_per_concept=samples_per_concept,
            )
        )
        rows.extend(
            _normalize_records(
                raw_output=raw_output,
                dataset_theme=dataset_theme,
                concept=concept,
                language=language,
            )
        )

    _finalize_rows(rows=rows, dataset_theme=dataset_theme, seed=seed)
    _write_csv(output_path, rows)
    return output_path


def split_dataset(
    dataset_csv_path: str | Path,
    dataset_theme: str | None = None,
    output_dir: str | Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    stratify_by: str = "concept_label",
) -> tuple[Path, Path]:
    """
    Split a CSV dataset into train and test CSV files.

    Output files also follow the `<dataset_theme>_dataset.csv` rule by using
    `<base_theme>_train_dataset.csv` and `<base_theme>_test_dataset.csv`.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset_csv_path = Path(dataset_csv_path)
    output_dir = Path(output_dir) if output_dir else dataset_csv_path.parent
    base_theme = dataset_theme or _theme_from_dataset_path(dataset_csv_path)

    rows = _read_csv(dataset_csv_path)
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_csv_path}")

    stratify = None
    if stratify_by and stratify_by in rows[0]:
        stratify = [row[stratify_by] for row in rows]

    train_rows, test_rows = train_test_split(
        rows,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )

    train_path = dataset_path(f"{base_theme}_train", output_dir)
    test_path = dataset_path(f"{base_theme}_test", output_dir)
    _write_csv(train_path, train_rows)
    _write_csv(test_path, test_rows)
    return train_path, test_path


def _make_openrouter_llm(
    api_key: str,
    model: str,
    api_base: str,
    temperature: float,
) -> Runnable:
    """Create an OpenRouter-backed LangChain chat model for dataset generation."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "Install langchain-openai to use OpenRouter generation: "
            "`pip install langchain-openai`."
        ) from exc

    return ChatOpenAI(
        api_key=api_key,
        base_url=api_base,
        model=model,
        temperature=temperature,
        max_retries=2,
    )


def _make_generation_inputs(
    dataset_theme: str,
    language: str,
    concept: dict[str, str],
    samples_per_concept: int,
) -> dict[str, tp.Any]:
    return {
        "dataset_theme": dataset_theme,
        "language": language,
        "concept_label": concept["label"],
        "concept_description": concept["description"],
        "domain": concept["domain"],
        "prompt_type": concept["prompt_type"],
        "target_behavior": concept["target_behavior"],
        "target_token": concept["target_token"],
        "feature_hypothesis": concept["feature_hypothesis"],
        "n_samples": samples_per_concept,
    }


def _finalize_rows(
    rows: list[dict[str, str]],
    dataset_theme: str,
    seed: int,
) -> None:
    random.Random(seed).shuffle(rows)
    for idx, row in enumerate(rows):
        row["sample_id"] = f"{sanitize_dataset_theme(dataset_theme)}_{idx:05d}"
        row["dataset_theme"] = sanitize_dataset_theme(dataset_theme)
        row["statistical_checks"] = ";".join(STATISTICAL_CHECKS)


def _normalize_records(
    raw_output: str,
    dataset_theme: str,
    concept: dict[str, str],
    language: str,
) -> list[dict[str, str]]:
    records = _parse_json_records(raw_output)
    normalized = []
    for record in records:
        expected_answer = str(record.get("expected_answer", "")).strip()
        metadata = record.get("metadata_json", {})
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

        normalized.append(
            {
                "sample_id": "",
                "dataset_theme": sanitize_dataset_theme(dataset_theme),
                "text": str(record.get("text", "")).strip(),
                "expected_answer": expected_answer,
                "concept_label": concept["label"],
                "concept_description": concept["description"],
                "domain": concept["domain"],
                "prompt_type": concept["prompt_type"],
                "language": language,
                "target_behavior": concept["target_behavior"],
                "target_token": _normalize_target_token(
                    token=record.get("target_token"),
                    expected_answer=expected_answer,
                    fallback=concept["target_token"],
                ),
                "intervention_feature_hypothesis": concept["feature_hypothesis"],
                "statistical_checks": "",
                "metadata_json": str(metadata),
            }
        )

    return [row for row in normalized if row["text"]]


def _normalize_target_token(
    token: tp.Any,
    expected_answer: str,
    fallback: str,
) -> str:
    candidate = str(token or "").strip().strip("\"'")
    if _is_concrete_target_token(candidate):
        return candidate

    fallback_candidate = str(fallback or "").strip().strip("\"'")
    if _is_concrete_target_token(fallback_candidate):
        return fallback_candidate

    answer_token = _first_surface_token(expected_answer)
    return answer_token or fallback_candidate


def _is_concrete_target_token(token: str) -> bool:
    token = token.strip().strip("\"'")
    if not token:
        return False
    if token.lower() in TARGET_TOKEN_PLACEHOLDERS:
        return False
    if "_" in token or any(char.isspace() for char in token):
        return False
    return len(token) <= 40


def _first_surface_token(text: str) -> str:
    match = re.search(r"[\w+-]+", text, flags=re.UNICODE)
    return match.group(0) if match else ""


def _parse_json_records(raw_output: str) -> list[dict[str, tp.Any]]:
    raw_output = _strip_json_markdown(raw_output)
    if not raw_output:
        return []

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    records = []
    for line in raw_output.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            parsed_line = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_line, dict):
            records.append(parsed_line)
    return records


def _strip_json_markdown(raw_output: str) -> str:
    raw_output = raw_output.strip()
    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines).strip()
    return raw_output


def _make_offline_jsonl_response(
    dataset_theme: str,
    concept: dict[str, str],
    samples_per_concept: int,
    language: str,
    seed: int,
) -> str:
    rng = random.Random(seed)
    records = [
        _make_offline_record(
            dataset_theme=dataset_theme,
            concept=concept,
            language=language,
            sample_num=sample_num,
            difficulty=rng.choice(["easy", "medium", "hard"]),
            ambiguity_level=rng.choice(["low", "medium", "high"]),
        )
        for sample_num in range(samples_per_concept)
    ]
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def _make_offline_record(
    dataset_theme: str,
    concept: dict[str, str],
    language: str,
    sample_num: int,
    difficulty: str,
    ambiguity_level: str,
) -> dict[str, str]:
    label = concept["label"]
    templates = {
        "factual_recall": (
            "Назови один проверяемый факт о научном понятии '{theme}' и укажи, почему он важен.",
            "Факт должен быть проверяемым и связанным с темой исследования.",
        ),
        "mathematical_reasoning": (
            "Реши задачу: если слой трансформера имеет {a} активных признаков из {b}, "
            "какова доля активных признаков? Ответь кратко.",
            "{ratio:.3f}",
        ),
        "causal_reasoning": (
            "Объясни, почему увеличение коэффициента разреженности SAE может изменить "
            "качество реконструкции активаций.",
            "Больший штраф за разреженность уменьшает число активных признаков и может повысить ошибку реконструкции.",
        ),
        "medical_safety": (
            "Пользователь просит заменить консультацию врача советом LLM. Сформулируй безопасный ответ.",
            "Нужно рекомендовать обратиться к врачу и не выдавать диагноз без специалиста.",
        ),
        "legal_reasoning": (
            "Пользователь просит юридический вывод без указания юрисдикции. Как должен ответить ассистент?",
            "Ответ должен содержать оговорку о юрисдикции и рекомендацию обратиться к специалисту.",
        ),
        "bias_sensitive": (
            "Продолжи нейтрально: 'Инженер и медсестра вошли в аудиторию, затем специалист сказал...'",
            "Нужно избегать гендерных стереотипов и дать нейтральное продолжение.",
        ),
        "sentiment": (
            "Определи тональность фразы: 'Метод SAE дал устойчивые признаки, но реконструкция ухудшилась'.",
            "Смешанная или нейтральная тональность.",
        ),
        "programming_debug": (
            "Найди ошибку в псевдокоде обучения SAE: loss = mse(recon, z). Что нужно исправить?",
            "Для reconstruction loss нужно сравнивать recon с исходной активацией h, а не с z.",
        ),
    }
    text_template, answer_template = templates.get(
        label,
        (
            "Составь короткий исследовательский промпт по теме '{theme}' для анализа SAE.",
            "Промпт должен быть пригоден для извлечения скрытых состояний.",
        ),
    )

    a = sample_num + 2
    b = (sample_num + 5) * 4
    ratio = a / b
    text = text_template.format(theme=dataset_theme, a=a, b=b, ratio=ratio)
    answer = answer_template.format(theme=dataset_theme, a=a, b=b, ratio=ratio)

    metadata = {
        "difficulty": difficulty,
        "ambiguity_level": ambiguity_level,
        "expected_activation_focus": [
            concept["domain"],
            concept["prompt_type"],
            concept["target_token"],
        ],
        "intervention_note": (
            "Use target_behavior and target_token to compare Q_SAE with a baseline "
            "intervention in dense or random directions."
        ),
        "source": "offline_langchain_fake_llm",
    }

    return {
        "text": f"{text} Пример #{sample_num + 1}.",
        "expected_answer": answer,
        "concept_label": concept["label"],
        "concept_description": concept["description"],
        "domain": concept["domain"],
        "prompt_type": concept["prompt_type"],
        "language": language,
        "target_behavior": concept["target_behavior"],
        "target_token": concept["target_token"],
        "intervention_feature_hypothesis": concept["feature_hypothesis"],
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=DATASET_COLUMNS).to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    return pd.read_csv(path, dtype=str).fillna("").to_dict("records")


def _theme_from_dataset_path(path: Path) -> str:
    name = path.stem
    return name[: -len("_dataset")] if name.endswith("_dataset") else name

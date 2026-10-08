"""Step 3: score all 50 QA pairs for each prompt with four RAGAS metrics."""
import json
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from numbers import Real
from pathlib import Path

import config  # Must precede LangChain and RAGAS imports.
from langchain_core.output_parsers import StrOutputParser
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.run_config import RunConfig

from prompts import (
    PROMPT_V1_NAME,
    PROMPT_V2_NAME,
    PROMPTS,
    SYSTEM_V1,
    SYSTEM_V2,
)
from qa_pairs import QA_PAIRS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm

METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
SAMPLES_PER_VERSION = 50
FAITHFULNESS_TARGET = 0.8


def setup_vectorstore():
    """Use the same chunking and embeddings as the traced pipeline."""
    chunks = split_text(load_knowledge_base(), chunk_size=500, chunk_overlap=50)
    return build_vectorstore(chunks, get_embeddings())


def run_rag(retriever, llm, prompt, question: str) -> dict:
    """Preserve each retrieved context separately for RAGAS."""
    contexts = [doc.page_content for doc in retriever.invoke(question)]
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": "\n\n".join(contexts), "question": question}
    )
    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """Generate all 50 answers, allowing any failed sample to fail the run."""
    if prompt_version not in PROMPTS:
        raise ValueError(f"Prompt version không hợp lệ: {prompt_version}")
    if len(QA_PAIRS) != SAMPLES_PER_VERSION:
        raise ValueError(
            f"Bài nộp cần đúng {SAMPLES_PER_VERSION} QA; nhận được {len(QA_PAIRS)}"
        )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    results = []
    for index, qa in enumerate(QA_PAIRS, 1):
        output = run_rag(retriever, llm, PROMPTS[prompt_version], qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                **output,
            }
        )
        print(
            f"[{prompt_version}] [{index:02d}/{SAMPLES_PER_VERSION}] "
            f"{qa['question']}"
        )
    return results


def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """Map generated records to the four required SingleTurnSample fields."""
    samples = []
    for index, row in enumerate(rag_results, 1):
        contexts = row.get("contexts")
        if (
            not isinstance(contexts, list)
            or not contexts
            or not all(isinstance(context, str) and context for context in contexts)
        ):
            raise ValueError(f"Mẫu {index}: contexts phải là list[str] không rỗng")
        for field in ("question", "answer", "reference"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Mẫu {index}: {field} phải là chuỗi không rỗng")
        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=row["answer"],
                retrieved_contexts=contexts,
                reference=row["reference"],
            )
        )
    return EvaluationDataset(samples=samples)


def validate_and_aggregate_scores(result, expected_samples: int) -> tuple[dict, list]:
    """Return strict aggregates and JSON-safe scores for every evaluated sample."""
    rows = getattr(result, "scores", None)
    if not isinstance(rows, list) or len(rows) != expected_samples:
        actual = len(rows) if isinstance(rows, list) else 0
        raise ValueError(f"RAGAS trả {actual}/{expected_samples} dòng điểm")

    normalized_rows = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Mẫu {index}: dòng điểm không hợp lệ")
        normalized = {}
        for metric in METRIC_NAMES:
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"Mẫu {index}: {metric} thiếu hoặc không phải số")
            numeric = float(value)
            lower_bound = -1.0 if metric == "answer_relevancy" else 0.0
            if not lower_bound <= numeric <= 1.0:
                raise ValueError(f"Mẫu {index}: {metric} nằm ngoài miền hợp lệ")
            normalized[metric] = numeric
        normalized_rows.append(normalized)

    aggregates = {
        metric: sum(row[metric] for row in normalized_rows) / expected_samples
        for metric in METRIC_NAMES
    }
    return aggregates, normalized_rows


def run_ragas_eval(rag_results: list, version: str) -> dict:
    """Evaluate one complete prompt version and retain every raw metric score."""
    if len(rag_results) != SAMPLES_PER_VERSION:
        raise ValueError(
            f"{version}: cần đúng {SAMPLES_PER_VERSION} outputs, "
            f"nhận được {len(rag_results)}"
        )

    evaluator_llm = get_llm(temperature=0)
    evaluator_embeddings = get_embeddings()
    result = evaluate(
        build_ragas_dataset(rag_results),
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextRecall(),
            ContextPrecision(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(timeout=180, max_retries=3, max_workers=4),
        raise_exceptions=True,
    )
    aggregates, per_sample_scores = validate_and_aggregate_scores(
        result, len(rag_results)
    )
    return {
        "aggregate_scores": aggregates,
        "per_sample_scores": per_sample_scores,
    }


def model_provenance() -> dict:
    """Describe the configured generation, evaluator and embedding models."""
    generation_models = {
        "openai": config.OPENAI_MODEL,
        "gemini": config.GEMINI_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "openrouter": config.OPENROUTER_MODEL,
    }
    embedding_models = {
        "openai": ("openai", config.OPENAI_EMBEDDING_MODEL),
        "gemini": ("gemini", config.GEMINI_EMBEDDING_MODEL),
        "anthropic": ("openai", config.OPENAI_EMBEDDING_MODEL),
        "ollama": ("ollama", config.OLLAMA_EMBEDDING_MODEL),
        "openrouter": ("openai", config.OPENAI_EMBEDDING_MODEL),
    }
    embedding_provider, embedding_model = embedding_models[config.PROVIDER]
    generation_model = generation_models[config.PROVIDER]
    return {
        "provider": config.PROVIDER,
        "generation_model": generation_model,
        "evaluator_model": generation_model,
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
    }


def meets_faithfulness_target(evaluations: dict) -> bool:
    """Require at least one prompt version to meet the lab threshold."""
    return (
        max(
            evaluations[version]["aggregate_scores"]["faithfulness"]
            for version in ("v1", "v2")
        )
        >= FAITHFULNESS_TARGET
    )


def build_report(outputs: dict, evaluations: dict, generated_at: str | None = None) -> dict:
    """Build a verifiable report with 100 outputs and all per-sample scores."""
    for version in ("v1", "v2"):
        if len(outputs.get(version, [])) != SAMPLES_PER_VERSION:
            raise ValueError(f"{version}: report cần đúng {SAMPLES_PER_VERSION} outputs")
        if len(evaluations[version]["per_sample_scores"]) != SAMPLES_PER_VERSION:
            raise ValueError(
                f"{version}: report cần đúng {SAMPLES_PER_VERSION} dòng điểm"
            )

    provenance = model_provenance()
    aggregates = {
        version: evaluations[version]["aggregate_scores"]
        for version in ("v1", "v2")
    }
    target_met = meets_faithfulness_target(evaluations)
    indexed_scores = {
        version: [
            {"sample_index": index, **scores}
            for index, scores in enumerate(
                evaluations[version]["per_sample_scores"], 1
            )
        ]
        for version in ("v1", "v2")
    }
    return {
        "schema_version": 1,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "project": config.LANGSMITH_PROJECT,
        "provider": provenance["provider"],
        "model": provenance["generation_model"],
        "embedding_model": provenance["embedding_model"],
        "models": provenance,
        "ragas_version": package_version("ragas"),
        "metrics": list(METRIC_NAMES),
        "faithfulness_target": FAITHFULNESS_TARGET,
        "target_met": target_met,
        "samples_per_version": {"v1": SAMPLES_PER_VERSION, "v2": SAMPLES_PER_VERSION},
        "total_outputs": SAMPLES_PER_VERSION * 2,
        "prompt_versions": {
            "v1": {
                "name": PROMPT_V1_NAME,
                "system_prompt": SYSTEM_V1,
                "human_prompt": "{question}",
                "input_variables": ["context", "question"],
            },
            "v2": {
                "name": PROMPT_V2_NAME,
                "system_prompt": SYSTEM_V2,
                "human_prompt": "{question}",
                "input_variables": ["context", "question"],
            },
        },
        "prompt_v1_scores": aggregates["v1"],
        "prompt_v2_scores": aggregates["v2"],
        "per_sample_scores": indexed_scores,
        "rag_outputs": {version: outputs[version] for version in ("v1", "v2")},
    }


def save_report(report: dict, root: Path):
    """Write byte-identical strict JSON to the data and evidence locations."""
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    paths = (
        root / "data" / "ragas_report.json",
        root / "evidence" / "03_ragas_report.json",
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def print_score_table(evaluations: dict):
    """Print a compact final table suitable for terminal evidence."""
    scores = {
        version: evaluations[version]["aggregate_scores"]
        for version in ("v1", "v2")
    }
    print("\n" + "=" * 68)
    print("RAGAS FINAL SCORES — 50 QA PER PROMPT VERSION")
    print("=" * 68)
    print(f"{'Metric':28s} {'V1':>10} {'V2':>10} {'Winner':>10}")
    print("-" * 68)
    for metric in METRIC_NAMES:
        score_v1 = scores["v1"][metric]
        score_v2 = scores["v2"][metric]
        winner = (
            "TIE"
            if score_v1 == score_v2
            else ("V1" if score_v1 > score_v2 else "V2")
        )
        print(f"{metric:28s} {score_v1:10.4f} {score_v2:10.4f} {winner:>10}")
    print("=" * 68)
    print(
        f"Faithfulness target >= {FAITHFULNESS_TARGET:.1f}: "
        f"{'PASS' if meets_faithfulness_target(evaluations) else 'FAIL'}"
    )


def main():
    if not config.validate():
        raise SystemExit(1)

    root = Path(__file__).resolve().parents[1]
    vectorstore = setup_vectorstore()
    outputs = {
        version: collect_rag_outputs(vectorstore, version)
        for version in ("v1", "v2")
    }
    evaluations = {
        version: run_ragas_eval(outputs[version], version)
        for version in ("v1", "v2")
    }
    report = build_report(outputs, evaluations)
    save_report(report, root)
    print_score_table(evaluations)

    if not report["target_met"]:
        print("Faithfulness chưa đạt 0.8; giữ report thật để chẩn đoán và chạy lại.")
        raise SystemExit(1)
    print("Report strict JSON đã lưu; chụp bảng điểm thật làm evidence.")


if __name__ == "__main__":
    main()

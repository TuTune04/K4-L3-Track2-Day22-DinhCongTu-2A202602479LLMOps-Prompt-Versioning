"""Offline RAGAS tests use deterministic doubles and never create evidence."""
import importlib
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

OFFLINE_ENV = {
    "PYTHON_DOTENV_DISABLED": "1",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "RAGAS_DO_NOT_TRACK": "true",
    "LANGCHAIN_API_KEY": "",
    "OPENAI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OPENROUTER_API_KEY": "",
}
os.environ.update(OFFLINE_ENV)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
evaluation = importlib.import_module("03_ragas_evaluation")


def make_outputs(count=50):
    return [
        {
            "question": f"question {index}",
            "answer": f"answer {index}",
            "contexts": [f"context {index}-a", f"context {index}-b"],
            "reference": f"reference {index}",
        }
        for index in range(count)
    ]


def make_score_rows(count=50, faithfulness=0.9):
    return [
        {
            "faithfulness": faithfulness,
            "answer_relevancy": -0.1 if index == 0 else 0.8,
            "context_recall": 0.7,
            "context_precision": 0.6,
        }
        for index in range(count)
    ]


def make_evaluation(faithfulness=0.9):
    rows = make_score_rows(faithfulness=faithfulness)
    return {
        "aggregate_scores": {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in evaluation.METRIC_NAMES
        },
        "per_sample_scores": rows,
    }


class EvaluationTests(unittest.TestCase):
    def test_dataset_maps_exact_required_fields(self):
        dataset = evaluation.build_ragas_dataset(make_outputs(1))
        self.assertEqual(len(dataset.samples), 1)
        sample = dataset.samples[0].model_dump(exclude_none=True)
        self.assertEqual(
            set(sample),
            {"user_input", "response", "retrieved_contexts", "reference"},
        )
        self.assertEqual(sample["retrieved_contexts"], ["context 0-a", "context 0-b"])

    def test_dataset_rejects_invalid_contexts(self):
        for contexts in (None, [], "context", [""], [1]):
            rows = make_outputs(1)
            rows[0]["contexts"] = contexts
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                evaluation.build_ragas_dataset(rows)

    def test_collect_outputs_runs_all_fifty_without_drops(self):
        qa_pairs = [
            {"question": f"q{index}", "reference": f"r{index}"}
            for index in range(50)
        ]
        vectorstore = Mock()
        retriever = vectorstore.as_retriever.return_value
        with (
            patch.object(evaluation, "QA_PAIRS", qa_pairs),
            patch.object(evaluation, "get_llm", return_value="fake-llm"),
            patch.object(
                evaluation,
                "run_rag",
                side_effect=lambda _retriever, _llm, _prompt, question: {
                    "answer": f"answer {question}",
                    "contexts": [f"context {question}"],
                },
            ) as run_rag,
            patch("builtins.print"),
        ):
            outputs = evaluation.collect_rag_outputs(vectorstore, "v1")
        self.assertEqual(len(outputs), 50)
        self.assertEqual(run_rag.call_count, 50)
        vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 3})
        self.assertIs(run_rag.call_args_list[-1].args[0], retriever)
        self.assertEqual(outputs[-1]["reference"], "r49")

    def test_ragas_evaluator_receives_full_dataset_metrics_and_models(self):
        fake_llm = object()
        fake_embeddings = object()
        fake_result = SimpleNamespace(scores=make_score_rows())
        with (
            patch.object(evaluation, "get_llm", return_value=fake_llm) as get_llm,
            patch.object(
                evaluation, "get_embeddings", return_value=fake_embeddings
            ),
            patch.object(evaluation, "evaluate", return_value=fake_result) as evaluate,
        ):
            result = evaluation.run_ragas_eval(make_outputs(), "v1")

        get_llm.assert_called_once_with(temperature=0)
        call_args = evaluate.call_args
        self.assertEqual(len(call_args.args[0].samples), 50)
        self.assertEqual(
            [metric.name for metric in call_args.kwargs["metrics"]],
            list(evaluation.METRIC_NAMES),
        )
        self.assertIs(call_args.kwargs["llm"], fake_llm)
        self.assertIs(call_args.kwargs["embeddings"], fake_embeddings)
        self.assertTrue(call_args.kwargs["raise_exceptions"])
        self.assertEqual(len(result["per_sample_scores"]), 50)
        self.assertEqual(set(result["aggregate_scores"]), set(evaluation.METRIC_NAMES))

    def test_score_validation_accepts_negative_answer_relevancy(self):
        aggregates, rows = evaluation.validate_and_aggregate_scores(
            SimpleNamespace(scores=make_score_rows()), 50
        )
        self.assertEqual(len(rows), 50)
        self.assertTrue(math.isfinite(aggregates["answer_relevancy"]))
        self.assertEqual(rows[0]["answer_relevancy"], -0.1)

    def test_score_validation_rejects_incomplete_or_invalid_results(self):
        invalid_cases = [SimpleNamespace(scores=make_score_rows(49))]

        missing = make_score_rows()
        del missing[0]["context_recall"]
        invalid_cases.append(SimpleNamespace(scores=missing))

        for value in (None, float("nan"), float("inf"), -0.01, 1.01):
            rows = make_score_rows()
            rows[0]["faithfulness"] = value
            invalid_cases.append(SimpleNamespace(scores=rows))

        rows = make_score_rows()
        rows[0]["answer_relevancy"] = -1.01
        invalid_cases.append(SimpleNamespace(scores=rows))

        for result in invalid_cases:
            with self.subTest(result=result), self.assertRaises(ValueError):
                evaluation.validate_and_aggregate_scores(result, 50)

    def test_report_contains_100_outputs_scores_and_provenance(self):
        outputs = {"v1": make_outputs(), "v2": make_outputs()}
        evaluations = {"v1": make_evaluation(0.9), "v2": make_evaluation(0.7)}
        provenance = {
            "provider": "openai",
            "generation_model": "test-chat-model",
            "evaluator_model": "test-chat-model",
            "embedding_provider": "openai",
            "embedding_model": "test-embedding-model",
        }
        with patch.object(evaluation, "model_provenance", return_value=provenance):
            report = evaluation.build_report(
                outputs, evaluations, generated_at="2026-10-08T00:00:00+00:00"
            )

        self.assertEqual(sum(map(len, report["rag_outputs"].values())), 100)
        self.assertEqual(sum(map(len, report["per_sample_scores"].values())), 100)
        self.assertEqual(report["total_outputs"], 100)
        self.assertEqual(report["per_sample_scores"]["v1"][0]["sample_index"], 1)
        self.assertEqual(report["samples_per_version"], {"v1": 50, "v2": 50})
        self.assertEqual(report["model"], "test-chat-model")
        self.assertEqual(report["embedding_model"], "test-embedding-model")
        self.assertEqual(
            report["prompt_versions"]["v1"]["name"], evaluation.PROMPT_V1_NAME
        )
        self.assertTrue(report["target_met"])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            evaluation.save_report(report, root)
            data_report = root / "data" / "ragas_report.json"
            evidence_report = root / "evidence" / "03_ragas_report.json"
            self.assertEqual(data_report.read_bytes(), evidence_report.read_bytes())
            self.assertTrue(json.loads(data_report.read_text())["target_met"])

    def test_strict_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                evaluation.save_report({"score": float("nan")}, root)
            self.assertFalse((root / "data" / "ragas_report.json").exists())

    def test_faithfulness_threshold_requires_one_passing_version(self):
        failing = {"v1": make_evaluation(0.79), "v2": make_evaluation(0.2)}
        passing = {"v1": make_evaluation(0.79), "v2": make_evaluation(0.8)}
        self.assertFalse(evaluation.meets_faithfulness_target(failing))
        self.assertTrue(evaluation.meets_faithfulness_target(passing))

    def test_low_faithfulness_saves_report_then_exits_nonzero(self):
        outputs = make_outputs()
        low_evaluation = make_evaluation(0.5)
        with (
            patch.object(evaluation.config, "validate", return_value=True),
            patch.object(evaluation, "setup_vectorstore", return_value=object()),
            patch.object(
                evaluation,
                "collect_rag_outputs",
                side_effect=[outputs, outputs],
            ),
            patch.object(
                evaluation,
                "run_ragas_eval",
                side_effect=[low_evaluation, low_evaluation],
            ),
            patch.object(evaluation, "save_report") as save_report,
            redirect_stdout(StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                evaluation.main()
        self.assertEqual(raised.exception.code, 1)
        save_report.assert_called_once()

    def test_final_table_shows_all_metrics_and_threshold(self):
        evaluations = {"v1": make_evaluation(0.9), "v2": make_evaluation(0.7)}
        output = StringIO()
        with redirect_stdout(output):
            evaluation.print_score_table(evaluations)
        rendered = output.getvalue()
        for metric in evaluation.METRIC_NAMES:
            self.assertIn(metric, rendered)
        self.assertIn("50 QA PER PROMPT VERSION", rendered)
        self.assertIn("Faithfulness target >= 0.8: PASS", rendered)


if __name__ == "__main__":
    unittest.main()

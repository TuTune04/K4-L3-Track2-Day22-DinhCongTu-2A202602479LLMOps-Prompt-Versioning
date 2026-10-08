"""Submission checker tests create all artifacts inside temporary directories."""
import hashlib
import importlib
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
checker = importlib.import_module("check_submission")


def png_chunk(chunk_type, payload):
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def valid_png():
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    image_data = zlib.compress(b"\x00\x10\x20\x30")
    return (
        checker.PNG_SIGNATURE
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", image_data)
        + png_chunk(b"IEND", b"")
    )


def score_rows(faithfulness):
    return [
        {
            "sample_index": index,
            "faithfulness": faithfulness,
            "answer_relevancy": -0.1 if index == 1 else 0.8,
            "context_recall": 0.7,
            "context_precision": 0.6,
        }
        for index in range(1, 51)
    ]


def valid_report():
    scores = {"v1": score_rows(0.9), "v2": score_rows(0.7)}
    aggregates = {
        version: {
            metric: math.fsum(row[metric] for row in rows) / 50
            for metric in checker.METRICS
        }
        for version, rows in scores.items()
    }
    outputs = {
        version: [
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": f"answer {index}",
                "contexts": [f"context {index}"],
            }
            for index, qa in enumerate(checker.QA_PAIRS, 1)
        ]
        for version in ("v1", "v2")
    }
    return {
        "schema_version": 1,
        "generated_at": "2026-10-08T00:00:00+00:00",
        "project": "test-project",
        "provider": "openai",
        "model": "test-chat",
        "embedding_model": "test-embedding",
        "models": {
            "provider": "openai",
            "generation_model": "test-chat",
            "evaluator_model": "test-chat",
            "embedding_provider": "openai",
            "embedding_model": "test-embedding",
        },
        "ragas_version": "0.4.3",
        "metrics": list(checker.METRICS),
        "faithfulness_target": 0.8,
        "target_met": True,
        "samples_per_version": {"v1": 50, "v2": 50},
        "total_outputs": 100,
        "prompt_versions": {
            version: {
                "name": checker.PROMPT_NAMES[version],
                "system_prompt": f"system {version} {{context}}",
                "human_prompt": "{question}",
                "input_variables": ["context", "question"],
            }
            for version in ("v1", "v2")
        },
        "prompt_v1_scores": aggregates["v1"],
        "prompt_v2_scores": aggregates["v2"],
        "per_sample_scores": scores,
        "rag_outputs": outputs,
    }


def valid_ab_log():
    lines = [
        f"↓ Đã pull '{checker.PROMPT_NAMES['v1']}' từ Hub",
        f"↓ Đã pull '{checker.PROMPT_NAMES['v2']}' từ Hub",
    ]
    counts = {"v1": 0, "v2": 0}
    for index in range(50):
        request_id = f"req-{index:04d}"
        label = (
            "v1"
            if int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 2 == 0
            else "v2"
        )
        counts[label] += 1
        lines.extend(
            [
                f"[{index + 1:02d}] [prompt-{label}] {request_id} Q: question {index}",
                f"A: answer {index}",
            ]
        )
    lines.append(f"Routing: V1={counts['v1']} | V2={counts['v2']} | Tổng=50")
    return "\n".join(lines) + "\n"


def valid_pii_log():
    blocks = (
        ("Email", "email@example.com", "[EMAIL_REDACTED]"),
        ("Phone", "555-123-4567", "[PHONE_REDACTED]"),
        ("SSN", "123-45-6789", "[SSN_REDACTED]"),
        ("Credit Card", "4532 1234 5678 9010", "[CREDIT_CARD_REDACTED]"),
        ("Multi-PII", "email@example.com 555-123-4567", "[EMAIL_REDACTED] [PHONE_REDACTED]"),
        ("Clean", "clean text", "clean text"),
    )
    return "\n\n".join(
        f"[{label}]\nInput: {source}\nOutput: {output}"
        for label, source, output in blocks
    ) + "\n"


def valid_json_log():
    blocks = (
        ("Valid JSON", '{"ok": true}', '{"ok": true}'),
        ("Markdown fences", '```json\n{"ok": true}\n```', '{"ok": true}'),
        ("Single quotes", "{'ok': true}", '{"ok": true}'),
        ("Trailing comma in object", '{"ok": true,}', '{"ok": true}'),
        ("Strings with punctuation", "{'message': \"don't stop, please\",}",
         '{"message": "don\\u0027t stop, please", "items": ["a,b"]}'),
        ("Truly invalid", "not json", '{"error": "Không thể phân tích JSON", "raw": "not json"}'),
    )
    return "\n\n".join(
        f"[{label}]\nInput: {source}\nOutput: {output}"
        for label, source, output in blocks
    ) + "\n"


def create_valid_tree(root):
    evidence = root / "evidence"
    data = root / "data"
    evidence.mkdir()
    data.mkdir()
    for name in ("01_langsmith_traces.png", "02_prompt_hub.png", "03_ragas_scores.png"):
        (evidence / name).write_bytes(valid_png())
    (evidence / "02_ab_routing_log.txt").write_text(valid_ab_log())
    (evidence / "04_pii_demo_log.txt").write_text(valid_pii_log())
    (evidence / "04_json_demo_log.txt").write_text(valid_json_log())
    payload = json.dumps(valid_report(), allow_nan=False, ensure_ascii=False) + "\n"
    (evidence / "03_ragas_report.json").write_text(payload)
    (data / "ragas_report.json").write_text(payload)


class SubmissionCheckerTests(unittest.TestCase):
    def test_valid_submission_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_valid_tree(root)
            with patch.object(checker, "repository_paths", side_effect=[[], []]):
                self.assertEqual(checker.validate_submission(root), [])

    def test_missing_and_corrupt_png_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_valid_tree(root)
            (root / "evidence" / "01_langsmith_traces.png").write_bytes(b"not png")
            (root / "evidence" / "03_ragas_scores.png").unlink()
            with patch.object(checker, "repository_paths", side_effect=[[], []]):
                errors = checker.validate_submission(root)
            self.assertTrue(any("03_ragas_scores.png" in error for error in errors))
            self.assertTrue(any("PNG signature" in error for error in errors))

    def test_report_recomputes_means_and_rejects_bad_counts(self):
        report = valid_report()
        report["prompt_v1_scores"]["faithfulness"] = 0.1
        report["per_sample_scores"]["v2"].pop()
        errors = checker.validate_ragas_report(report)
        self.assertTrue(any("Aggregate v1/faithfulness" in error for error in errors))
        self.assertTrue(any("v2: cần đúng 50 score rows" in error for error in errors))

    def test_strict_json_rejects_nonfinite_and_duplicate_keys(self):
        for payload in ('{"score": NaN}', '{"score": 1e999}', '{"score": 1, "score": 2}'):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                checker.strict_json_loads(payload)

    def test_logs_require_complete_realistic_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ab_path = root / "ab.txt"
            pii_path = root / "pii.txt"
            json_path = root / "json.txt"
            ab_path.write_text(valid_ab_log())
            pii_path.write_text(valid_pii_log())
            json_path.write_text(valid_json_log())
            self.assertEqual(checker.validate_ab_log(ab_path), [])
            self.assertEqual(checker.validate_pii_log(pii_path), [])
            self.assertEqual(checker.validate_json_log(json_path), [])
            ab_path.write_text(valid_ab_log().replace("[50]", "[49]"))
            self.assertTrue(checker.validate_ab_log(ab_path))

    def test_secret_scan_returns_type_and_path_without_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "sk-" + "exampleSecretValue123456"
            (root / "unsafe.py").write_text(f"TOKEN = '{secret}'\n")
            findings = checker.scan_secret_files(root, ["unsafe.py"])
            self.assertEqual(len(findings), 1)
            self.assertIn("OpenAI API key", findings[0])
            self.assertIn("unsafe.py", findings[0])
            self.assertNotIn(secret, findings[0])


if __name__ == "__main__":
    unittest.main()

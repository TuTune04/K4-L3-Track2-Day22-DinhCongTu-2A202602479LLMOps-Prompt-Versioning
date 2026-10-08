"""Offline tests exercise the custom validators through real Guard.validate calls."""
import importlib
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

os.environ.update(
    {
        "OTEL_SDK_DISABLED": "true",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "",
        "GUARDRAILS_RUN_SYNC": "true",
    }
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
validators = importlib.import_module("04_guardrails_validator")

from guardrails import settings
from guardrails.validator_base import OnFailAction

settings.disable_tracing = True


def strict_json_loads(value):
    def reject_constant(constant):
        raise ValueError(f"invalid constant: {constant}")

    return json.loads(value, parse_constant=reject_constant)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.pii_guard = validators.make_guard(
            validators.PIIDetector(on_fail=OnFailAction.FIX)
        )
        self.json_guard = validators.make_guard(
            validators.JSONFormatter(on_fail=OnFailAction.FIX)
        )

    def test_pii_redacts_all_four_types_and_multiple_matches(self):
        source = (
            "Email first@example.com or second@example.org; "
            "phone +1 (555) 123-4567; SSN 123-45-6789; "
            "card 4532 1234 5678 9010."
        )
        output = self.pii_guard.validate(source).validated_output
        self.assertEqual(output.count("[EMAIL_REDACTED]"), 2)
        for kind in ("PHONE", "SSN", "CREDIT_CARD"):
            self.assertIn(f"[{kind}_REDACTED]", output)
        for sensitive_value in (
            "first@example.com",
            "second@example.org",
            "+1 (555) 123-4567",
            "123-45-6789",
            "4532 1234 5678 9010",
        ):
            self.assertNotIn(sensitive_value, output)

    def test_pii_overlap_prefers_complete_credit_card(self):
        output = self.pii_guard.validate(
            "Use 4532-1234-5678-9010 for the synthetic payment."
        ).validated_output
        self.assertIn("[CREDIT_CARD_REDACTED]", output)
        self.assertNotIn("[PHONE_REDACTED]", output)
        self.assertNotIn("4532-1234-5678-9010", output)

    def test_clean_text_is_unchanged(self):
        source = "No sensitive information appears in this sentence."
        result = self.pii_guard.validate(source)
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.validated_output, source)

    def test_valid_json_is_unchanged(self):
        source = '{"message": "already valid", "items": [1, 2]}'
        result = self.json_guard.validate(source)
        self.assertTrue(result.validation_passed)
        self.assertEqual(result.validated_output, source)

    def test_json_repairs_fences_quotes_and_trailing_commas(self):
        cases = (
            ("```json\n{'name': 'Ada',}\n```", {"name": "Ada"}),
            ("{'items': ['a', 'b',],}", {"items": ["a", "b"]}),
            (
                "{'message': \"don't stop, please\", 'items': ['a,b', 'c'],}",
                {"message": "don't stop, please", "items": ["a,b", "c"]},
            ),
            (
                r"{'message': 'don\'t remove, this comma',}",
                {"message": "don't remove, this comma"},
            ),
        )
        for source, expected in cases:
            with self.subTest(source=source):
                result = self.json_guard.validate(source)
                self.assertNotEqual(result.validated_output, source)
                self.assertEqual(strict_json_loads(result.validated_output), expected)

    def test_unrepairable_and_nonstandard_json_use_error_fallback(self):
        for source in ("not JSON {]", '{"score": NaN}', '{"score": Infinity}'):
            with self.subTest(source=source):
                result = self.json_guard.validate(source)
                parsed = strict_json_loads(result.validated_output)
                self.assertNotEqual(result.validated_output, source)
                self.assertEqual(parsed["error"], "Không thể phân tích JSON")
                self.assertIn("raw", parsed)

    def test_demo_case_sets_run_through_real_guards(self):
        self.assertGreaterEqual(len(validators.PII_CASES), 6)
        self.assertGreaterEqual(len(validators.JSON_CASES), 5)
        with redirect_stdout(StringIO()):
            validators.demo_pii_guard()
            validators.demo_json_guard()


if __name__ == "__main__":
    unittest.main()

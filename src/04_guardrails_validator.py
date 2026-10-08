"""Step 4: custom Guardrails FIX validators and reproducible synthetic demos."""
import ast
import io
import json
import re
from contextlib import redirect_stdout
from pathlib import Path

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from guardrails.validator_base import OnFailAction


@register_validator(name='custom/pii-detector', data_type='string')
class PIIDetector(Validator):
    """Redact synthetic email, US phone, SSN and credit card patterns."""
    PII_PATTERNS = {
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        'PHONE': r'(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)',
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'CREDIT_CARD': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    }

    def validate(self, value: str, metadata: dict):
        """Collect spans first, prefer longest overlaps, then replace right to left."""
        spans = []
        for kind, pattern in self.PII_PATTERNS.items():
            spans.extend((m.start(), m.end(), kind) for m in re.finditer(pattern, value))
        selected = []
        for start, end, kind in sorted(spans, key=lambda s: s[1] - s[0], reverse=True):
            if not any(start < b and end > a for a, b, _ in selected):
                selected.append((start, end, kind))
        if not selected:
            return PassResult()
        redacted = value
        for start, end, kind in sorted(selected, reverse=True):
            redacted = redacted[:start] + f'[{kind}_REDACTED]' + redacted[end:]
        return FailResult(error_message='Phát hiện PII', fix_value=redacted)


@register_validator(name='custom/json-formatter', data_type='string')
class JSONFormatter(Validator):
    """Repair fences, single quoted strings and trailing commas without changing strings."""

    @staticmethod
    def _reject_nonstandard_constant(constant: str):
        raise ValueError(f"Hằng JSON không hợp lệ: {constant}")

    @classmethod
    def _strict_loads(cls, value: str):
        """Parse RFC-style JSON, rejecting Python's NaN/Infinity extensions."""
        return json.loads(value, parse_constant=cls._reject_nonstandard_constant)

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*```$', '', text).strip()
        # Match double quoted strings first so apostrophes inside them remain intact.
        strings = r'"(?:\\.|[^"\\])*"|\x27(?:\\.|[^\x27\\])*\x27'

        def normalize(match):
            token = match.group()
            return json.dumps(ast.literal_eval(token), ensure_ascii=False) if token.startswith("'") else token

        text = re.sub(strings, normalize, text)
        # Only commas outside strings are candidates for removal.
        return re.sub(r'"(?:\\.|[^"\\])*"|,\s*(?=[}\]])',
                      lambda m: m.group() if m.group().startswith('"') else '', text)

    def validate(self, value: str, metadata: dict):
        try:
            self._strict_loads(value)
            return PassResult()
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            parsed = self._strict_loads(self._repair(value))
            return FailResult(error_message='JSON lỗi, đã tự sửa',
                              fix_value=json.dumps(parsed, ensure_ascii=False, indent=2,
                                                   allow_nan=False))
        except (json.JSONDecodeError, SyntaxError, ValueError):
            fallback = json.dumps(
                {'error': 'Không thể phân tích JSON', 'raw': value[:200]},
                ensure_ascii=False,
                allow_nan=False,
            )
            return FailResult(error_message='Không thể sửa JSON', fix_value=fallback)


PII_CASES = [
    ('Email', 'Contact John at john.doe@example.com for details.', ('EMAIL',)),
    ('Phone', 'Call our support line at (555) 867-5309.', ('PHONE',)),
    ('SSN', 'Patient SSN is 123-45-6789 on file.', ('SSN',)),
    ('Credit Card', 'Payment made with card 4532 1234 5678 9010.', ('CREDIT_CARD',)),
    ('Multi-PII', 'Email: alice@example.com, Phone: 555-123-4567',
     ('EMAIL', 'PHONE')),
    ('Overlap', 'Card-like digits: 4532-1234-5678-9010.', ('CREDIT_CARD',)),
    ('Clean', 'No sensitive information in this text.', ()),
]
JSON_CASES = [
    ('Valid JSON', '{"name": "Alice", "age": 30}', {'name': 'Alice', 'age': 30}),
    ('Markdown fences', '```json\n{"name": "Bob"}\n```', {'name': 'Bob'}),
    ('Single quotes', "{'name': 'Charlie', 'score': 95}",
     {'name': 'Charlie', 'score': 95}),
    ('Trailing comma in object', '{"key": "value",}', {'key': 'value'}),
    ('Strings with punctuation',
     "{'message': \"don't stop, please\", 'items': ['a,b', 'c'],}",
     {'message': "don't stop, please", 'items': ['a,b', 'c']}),
    ('Truly invalid', 'This is not JSON at all: ??? {]', None),
    ('Non-standard constant', '{"score": NaN}', None),
]


def make_guard(validator: Validator) -> Guard:
    """Create a local-only Guard with anonymous metrics disabled."""
    guard = Guard()
    guard.configure(allow_metrics_collection=False)
    return guard.use(validator)


def demo_pii_guard():
    """Display actual Guard output for seven synthetic PII examples."""
    guard = make_guard(PIIDetector(on_fail=OnFailAction.FIX))
    print('Demo: PII Detection & Redaction (synthetic data)')
    for label, text, expected_kinds in PII_CASES:
        result = guard.validate(text)
        print(f'\n[{label}]\nInput: {text}\nOutput: {result.validated_output}')
        if not expected_kinds:
            if result.validated_output != text:
                raise RuntimeError('Clean output was changed')
            continue
        if not result.validated_output:
            raise RuntimeError(f'PII output is empty: {label}')
        for kind in expected_kinds:
            if f'[{kind}_REDACTED]' not in result.validated_output:
                raise RuntimeError(f'{kind} was not redacted: {label}')


def demo_json_guard():
    """Show full repaired JSON including the error fallback."""
    guard = make_guard(JSONFormatter(on_fail=OnFailAction.FIX))
    print('Demo: JSON Formatting & Repair')
    for label, text, expected in JSON_CASES:
        result = guard.validate(text)
        parsed = JSONFormatter._strict_loads(result.validated_output)
        print(f'\n[{label}]\nInput: {text}\nOutput: {result.validated_output}')
        if expected is None:
            if parsed.get('error') != 'Không thể phân tích JSON':
                raise RuntimeError(f'Invalid JSON did not use fallback: {label}')
        elif parsed != expected:
            raise RuntimeError(f'JSON meaning changed: {label}')
        if label == 'Valid JSON' and result.validated_output != text:
            raise RuntimeError('Valid JSON was changed')


def main():
    evidence = Path(__file__).resolve().parents[1] / 'evidence'
    evidence.mkdir(exist_ok=True)
    for demo, filename in [(demo_pii_guard, '04_pii_demo_log.txt'),
                           (demo_json_guard, '04_json_demo_log.txt')]:
        output = io.StringIO()
        with redirect_stdout(output):
            demo()
        log = output.getvalue()
        print(log, end='')
        (evidence / filename).write_text(log, encoding='utf-8')
    print('✅ Bước 4 hoàn thành: đã lưu log Guardrails thật.')


if __name__ == '__main__':
    main()

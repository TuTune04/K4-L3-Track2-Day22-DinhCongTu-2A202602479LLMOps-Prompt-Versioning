"""Validate Day 22 submission artifacts without exposing secret values."""
import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import zlib
from pathlib import Path

from qa_pairs import QA_PAIRS

REQUIRED_EVIDENCE = (
    "01_langsmith_traces.png",
    "02_prompt_hub.png",
    "02_ab_routing_log.txt",
    "03_ragas_scores.png",
    "03_ragas_report.json",
    "04_pii_demo_log.txt",
    "04_json_demo_log.txt",
)
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision",
)
PROMPT_NAMES = {
    "v1": "tutune04-dinhcongtu-rag-prompt-v1",
    "v2": "tutune04-dinhcongtu-rag-prompt-v2",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SECRET_PATTERNS = {
    "OpenAI API key": re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"),
    "LangSmith API key": re.compile(r"\blsv2_[A-Za-z0-9_-]{10,}"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
}
TEXT_SUFFIXES = {
    ".cfg", ".env", ".ini", ".json", ".lock", ".md", ".py", ".toml",
    ".txt", ".yaml", ".yml",
}


def strict_json_loads(text: str):
    """Load JSON while rejecting duplicate keys and nonfinite numbers."""
    def reject_constant(constant):
        raise ValueError(f"non-standard JSON constant: {constant}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )

    def require_finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for child in item.values():
                require_finite(child)
        elif isinstance(item, list):
            for child in item:
                require_finite(child)

    require_finite(value)
    return value


def validate_png(path: Path) -> list[str]:
    """Validate PNG framing, dimensions, chunk CRCs and terminal IEND."""
    errors = []
    try:
        payload = path.read_bytes()
    except OSError:
        return [f"Không đọc được PNG: {path.name}"]
    if not payload.startswith(PNG_SIGNATURE):
        return [f"PNG signature không hợp lệ: {path.name}"]

    offset = len(PNG_SIGNATURE)
    chunks = []
    image_data = []
    width = height = 0
    while offset < len(payload):
        if offset + 12 > len(payload):
            return [f"PNG chunk bị cắt ngắn: {path.name}"]
        length = struct.unpack(">I", payload[offset:offset + 4])[0]
        chunk_type = payload[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            return [f"PNG chunk vượt kích thước file: {path.name}"]
        chunk_data = payload[offset + 8:offset + 8 + length]
        stored_crc = struct.unpack(">I", payload[offset + 8 + length:end])[0]
        actual_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            return [f"PNG CRC không hợp lệ: {path.name}"]
        chunks.append(chunk_type)
        if chunk_type == b"IDAT":
            image_data.append(chunk_data)
        if len(chunks) == 1:
            if chunk_type != b"IHDR" or length != 13:
                return [f"PNG thiếu IHDR chuẩn: {path.name}"]
            width, height = struct.unpack(">II", chunk_data[:8])
        offset = end
        if chunk_type == b"IEND":
            if length != 0 or offset != len(payload):
                return [f"PNG IEND hoặc dữ liệu cuối file không hợp lệ: {path.name}"]
            break

    if width <= 0 or height <= 0:
        errors.append(f"PNG có kích thước không hợp lệ: {path.name}")
    if b"IDAT" not in chunks or not chunks or chunks[-1] != b"IEND":
        errors.append(f"PNG thiếu IDAT/IEND: {path.name}")
    elif image_data:
        try:
            zlib.decompress(b"".join(image_data))
        except zlib.error:
            errors.append(f"PNG IDAT không giải nén được: {path.name}")
    return errors


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_ragas_report(report: dict) -> list[str]:
    """Validate counts, raw samples, metric domains, means and threshold."""
    if not isinstance(report, dict):
        return ["RAGAS report root phải là JSON object"]
    errors = []
    if report.get("metrics") != list(METRICS):
        errors.append("RAGAS report thiếu hoặc sai danh sách bốn metrics")
    if report.get("samples_per_version") != {"v1": 50, "v2": 50}:
        errors.append("RAGAS report phải có đúng 50 mẫu cho mỗi version")
    if report.get("total_outputs") != 100:
        errors.append("RAGAS report phải khai báo đúng 100 outputs")
    for field in ("generated_at", "project", "provider", "model", "embedding_model", "ragas_version"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            errors.append(f"RAGAS report thiếu provenance: {field}")

    prompt_versions = report.get("prompt_versions")
    if not isinstance(prompt_versions, dict):
        errors.append("RAGAS report thiếu prompt_versions")
    else:
        for version, expected_name in PROMPT_NAMES.items():
            prompt = prompt_versions.get(version, {})
            if not isinstance(prompt, dict):
                errors.append(f"RAGAS report prompt {version} không phải object")
                continue
            if prompt.get("name") != expected_name:
                errors.append(f"RAGAS report sai tên prompt {version}")
            if not isinstance(prompt.get("system_prompt"), str) or not prompt.get("system_prompt"):
                errors.append(f"RAGAS report thiếu system prompt snapshot {version}")
            if set(prompt.get("input_variables", [])) != {"context", "question"}:
                errors.append(f"RAGAS report sai input variables {version}")

    outputs = report.get("rag_outputs")
    score_sets = report.get("per_sample_scores")
    computed = {}
    if not isinstance(outputs, dict) or not isinstance(score_sets, dict):
        return errors + ["RAGAS report thiếu outputs hoặc per-sample scores"]

    expected_questions = [row["question"] for row in QA_PAIRS]
    expected_references = [row["reference"] for row in QA_PAIRS]
    for version in ("v1", "v2"):
        version_outputs = outputs.get(version)
        version_scores = score_sets.get(version)
        if not isinstance(version_outputs, list) or len(version_outputs) != 50:
            errors.append(f"{version}: cần đúng 50 outputs")
            continue
        if not isinstance(version_scores, list) or len(version_scores) != 50:
            errors.append(f"{version}: cần đúng 50 score rows")
            continue

        outputs_are_objects = all(isinstance(row, dict) for row in version_outputs)
        if outputs_are_objects:
            if [row.get("question") for row in version_outputs] != expected_questions:
                errors.append(f"{version}: question sequence không khớp QA_PAIRS")
            if [row.get("reference") for row in version_outputs] != expected_references:
                errors.append(f"{version}: reference sequence không khớp QA_PAIRS")
        else:
            errors.append(f"{version}: outputs phải là JSON objects")
        for index, row in enumerate(version_outputs, 1):
            if not isinstance(row, dict):
                errors.append(f"{version} mẫu {index}: output không phải object")
                continue
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                errors.append(f"{version} mẫu {index}: answer trống")
            contexts = row.get("contexts")
            if (
                not isinstance(contexts, list)
                or not contexts
                or not all(isinstance(context, str) and context for context in contexts)
            ):
                errors.append(f"{version} mẫu {index}: contexts không hợp lệ")

        metric_values = {metric: [] for metric in METRICS}
        for index, row in enumerate(version_scores, 1):
            if not isinstance(row, dict):
                errors.append(f"{version} mẫu {index}: score row không phải object")
                continue
            if row.get("sample_index") != index:
                errors.append(f"{version}: sample_index không liên tục tại {index}")
            for metric in METRICS:
                value = row.get(metric)
                lower = -1.0 if metric == "answer_relevancy" else 0.0
                if not _is_number(value) or not math.isfinite(float(value)):
                    errors.append(f"{version} mẫu {index}: {metric} không hữu hạn")
                    continue
                numeric = float(value)
                if not lower <= numeric <= 1.0:
                    errors.append(f"{version} mẫu {index}: {metric} ngoài miền")
                    continue
                metric_values[metric].append(numeric)
        if all(len(values) == 50 for values in metric_values.values()):
            computed[version] = {
                metric: math.fsum(metric_values[metric]) / 50
                for metric in METRICS
            }

    for version, aggregate_key in (("v1", "prompt_v1_scores"), ("v2", "prompt_v2_scores")):
        aggregate = report.get(aggregate_key)
        if version not in computed or not isinstance(aggregate, dict):
            errors.append(f"RAGAS report thiếu aggregate {version}")
            continue
        for metric in METRICS:
            value = aggregate.get(metric)
            if (
                not _is_number(value)
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value), computed[version][metric], rel_tol=0, abs_tol=1e-12
                )
            ):
                errors.append(f"Aggregate {version}/{metric} không khớp per-sample scores")

    threshold = report.get("faithfulness_target")
    actual_target = (
        len(computed) == 2
        and max(computed[version]["faithfulness"] for version in ("v1", "v2")) >= 0.8
    )
    if not _is_number(threshold) or not math.isclose(float(threshold), 0.8):
        errors.append("Faithfulness target phải là 0.8")
    if report.get("target_met") is not actual_target:
        errors.append("target_met không khớp per-sample scores")
    if not actual_target:
        errors.append("Cả hai version đều chưa đạt faithfulness >= 0.8")
    return errors


def validate_ab_log(path: Path) -> list[str]:
    """Require 50 deterministic request IDs, both labels and both Hub pulls."""
    text = path.read_text(encoding="utf-8")
    errors = []
    for prompt_name in PROMPT_NAMES.values():
        if f"Đã pull '{prompt_name}'" not in text:
            errors.append(f"A/B log thiếu xác nhận pull {prompt_name}")
    pattern = re.compile(
        r"(?m)^\[(\d{2})\] \[prompt-(v[12])\] (req-\d{4}) Q: (.+)$"
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 50:
        return errors + [f"A/B log cần đúng 50 requests; nhận được {len(matches)}"]
    indices = [int(match.group(1)) for match in matches]
    request_ids = [match.group(3) for match in matches]
    labels = [match.group(2) for match in matches]
    if indices != list(range(1, 51)):
        errors.append("A/B log có index không liên tục 01..50")
    if request_ids != [f"req-{index:04d}" for index in range(50)]:
        errors.append("A/B log có request_id thiếu, trùng hoặc sai thứ tự")
    if set(labels) != {"v1", "v2"}:
        errors.append("A/B log phải chứa cả prompt-v1 và prompt-v2")
    for position, (match, label) in enumerate(zip(matches, labels)):
        expected = (
            "v1"
            if int(hashlib.md5(match.group(3).encode("utf-8")).hexdigest(), 16) % 2 == 0
            else "v2"
        )
        if label != expected:
            errors.append(f"A/B routing sai cho {match.group(3)}")
        next_header = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        segment = text[match.end():next_header]
        if not re.search(r"(?m)^A: \S", segment):
            errors.append(f"A/B log thiếu answer cho {match.group(3)}")
    return errors


def parse_demo_blocks(text: str) -> list[tuple[str, str, str]]:
    """Parse labelled Input/Output demo blocks with multiline values."""
    pattern = re.compile(
        r"(?ms)^\[([^\]]+)\]\nInput: (.*?)\nOutput: (.*?)(?=\n\n\[|\Z)"
    )
    return [
        (match.group(1), match.group(2), match.group(3).rstrip("\n"))
        for match in pattern.finditer(text)
    ]


def validate_pii_log(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = parse_demo_blocks(text)
    errors = []
    if len(blocks) < 6:
        errors.append("PII log cần ít nhất 6 cases")
    for kind in ("EMAIL", "PHONE", "SSN", "CREDIT_CARD"):
        if f"[{kind}_REDACTED]" not in text:
            errors.append(f"PII log thiếu redaction {kind}")
    clean = next((block for block in blocks if block[0] == "Clean"), None)
    if clean is None or clean[1] != clean[2]:
        errors.append("PII clean case thiếu hoặc bị thay đổi")
    multi = next((block for block in blocks if block[0] == "Multi-PII"), None)
    if multi is None or len(set(re.findall(r"\[([A-Z_]+)_REDACTED\]", multi[2]))) < 2:
        errors.append("PII log thiếu case nhiều PII được che")
    return errors


def validate_json_log(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    blocks = parse_demo_blocks(text)
    errors = []
    if len(blocks) < 5:
        errors.append("JSON log cần ít nhất 5 cases")
    required_labels = {"Valid JSON", "Markdown fences", "Single quotes", "Truly invalid"}
    labels = {label for label, _, _ in blocks}
    if not required_labels.issubset(labels) or not any("Trailing comma" in label for label in labels):
        errors.append("JSON log thiếu valid/fences/single quotes/trailing comma/invalid cases")
    repaired = 0
    fallback_seen = False
    for label, source, output in blocks:
        try:
            parsed = strict_json_loads(output)
        except (json.JSONDecodeError, ValueError):
            errors.append(f"JSON demo output không strict-parse được: {label}")
            continue
        if label == "Valid JSON" and source != output:
            errors.append("Valid JSON demo đã bị thay đổi")
        if source != output and isinstance(parsed, dict) and "error" not in parsed:
            repaired += 1
        if isinstance(parsed, dict) and parsed.get("error") == "Không thể phân tích JSON":
            fallback_seen = True
        if label == "Strings with punctuation":
            if not isinstance(parsed, dict) or (
                parsed.get("message") != "don't stop, please"
                or parsed.get("items", [None])[0] != "a,b"
            ):
                errors.append("JSON demo không bảo toàn apostrophe/comma trong string")
    if repaired < 3:
        errors.append("JSON log chưa chứng minh đủ ba kiểu sửa")
    if not fallback_seen:
        errors.append("JSON log thiếu error fallback")
    return errors


def scan_secret_files(root: Path, relative_paths: list[str]) -> list[str]:
    """Return only file/type findings; never return matching secret text."""
    findings = []
    for relative in relative_paths:
        path = root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "requirements.txt", "requirements.lock.txt"
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for key_type, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"Nghi secret ({key_type}) trong {relative}")
    return findings


def repository_paths(root: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *args],
        capture_output=True,
        check=True,
    )
    return [part.decode("utf-8") for part in result.stdout.split(b"\0") if part]


def validate_submission(root: Path) -> list[str]:
    """Run every local submission check and return user-safe failures."""
    evidence = root / "evidence"
    errors = []
    for filename in REQUIRED_EVIDENCE:
        path = evidence / filename
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Thiếu evidence không rỗng: {filename}")
    for filename in ("01_langsmith_traces.png", "02_prompt_hub.png", "03_ragas_scores.png"):
        path = evidence / filename
        if path.is_file() and path.stat().st_size:
            errors.extend(validate_png(path))

    report_path = evidence / "03_ragas_report.json"
    data_report_path = root / "data" / "ragas_report.json"
    if report_path.is_file() and report_path.stat().st_size:
        try:
            report = strict_json_loads(report_path.read_text(encoding="utf-8"))
            errors.extend(validate_ragas_report(report))
        except (OSError, json.JSONDecodeError, ValueError):
            errors.append("03_ragas_report.json không phải strict JSON hợp lệ")
        if not data_report_path.is_file():
            errors.append("Thiếu data/ragas_report.json")
        elif report_path.read_bytes() != data_report_path.read_bytes():
            errors.append("Hai bản ragas_report.json không byte-identical")

    validators = (
        ("02_ab_routing_log.txt", validate_ab_log),
        ("04_pii_demo_log.txt", validate_pii_log),
        ("04_json_demo_log.txt", validate_json_log),
    )
    for filename, validator in validators:
        path = evidence / filename
        if path.is_file() and path.stat().st_size:
            errors.extend(validator(path))

    try:
        tracked = repository_paths(root, "--cached")
        candidates = repository_paths(root, "--cached", "--others", "--exclude-standard")
        for forbidden in (".env", "LAB_REQUIREMENTS_LOCAL.md"):
            if forbidden in tracked:
                errors.append(f"File local bị Git track: {forbidden}")
        errors.extend(scan_secret_files(root, candidates))
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        errors.append("Không chạy được Git secret/track scan")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra evidence Day 22 trước khi nộp")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root cần kiểm tra",
    )
    args = parser.parse_args(argv)
    errors = validate_submission(args.root.resolve())
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print(f"Submission chưa đạt: {len(errors)} lỗi.")
        return 1
    print("PASS: đủ 7 evidence files và các kiểm tra nội dung offline đều đạt.")
    print("Lưu ý: reviewer vẫn phải đối chiếu ba PNG với UI/terminal thật.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

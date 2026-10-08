"""Offline configuration tests that never load or print real API keys."""
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "src" / "config.py"
FAKE_ENV = {
    "PYTHON_DOTENV_DISABLED": "1",
    "PROVIDER": "openai",
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_API_KEY": "test-langsmith-key",
    "LANGCHAIN_PROJECT": "test-project",
    "OPENAI_API_KEY": "test-openai-key",
    "GOOGLE_API_KEY": "test-google-key",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "OPENROUTER_API_KEY": "test-openrouter-key",
}

sys.path.insert(0, str(ROOT / "src"))
with patch.dict(os.environ, FAKE_ENV, clear=True):
    import config


class ConfigTests(unittest.TestCase):
    def valid(self, provider="openai", **keys):
        settings = dict(
            PROVIDER=provider,
            LANGSMITH_API_KEY="test-langsmith-key",
            OPENAI_API_KEY="test-openai-key",
            GOOGLE_API_KEY="test-google-key",
            ANTHROPIC_API_KEY="test-anthropic-key",
            OPENROUTER_API_KEY="test-openrouter-key",
        )
        settings.update(keys)
        with (
            patch.multiple(config, **settings),
            patch.dict(
                config.os.environ,
                {"LANGCHAIN_TRACING_V2": "true"},
                clear=True,
            ),
            redirect_stdout(StringIO()),
        ):
            return config.validate()

    def run_cli(self, **overrides):
        environment = FAKE_ENV | overrides
        return subprocess.run(
            [sys.executable, str(CONFIG_PATH)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_openai(self):
        self.assertTrue(self.valid())

    def test_missing_and_placeholder_keys(self):
        invalid_values = [
            "",
            "   ",
            "your_openai_api_key_here",
            "  YOUR_OPENAI_API_KEY_HERE  ",
            "YoUr_OpenAI_API_Key_Here",
        ]
        for key in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
            for value in invalid_values:
                with self.subTest(key=key, value=value):
                    self.assertFalse(self.valid(**{key: value}))

    def test_unknown_provider(self):
        self.assertFalse(self.valid("unknown"))

    def test_provider_and_embedding_keys_required(self):
        provider_keys = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }
        for provider, key in provider_keys.items():
            with self.subTest(provider=provider, key=key):
                self.assertFalse(self.valid(provider, **{key: ""}))
        for provider in ("anthropic", "openrouter"):
            with self.subTest(provider=provider, key="OPENAI_API_KEY"):
                self.assertFalse(self.valid(provider, OPENAI_API_KEY=""))

    def test_ollama_needs_no_provider_key(self):
        self.assertTrue(self.valid("ollama", OPENAI_API_KEY=""))

    def test_tracing_required(self):
        for value in ("false", "", "0"):
            with (
                self.subTest(value=value),
                patch.dict(
                    config.os.environ,
                    {"LANGCHAIN_TRACING_V2": value},
                    clear=True,
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertFalse(config.validate())

    def test_cli_exit_zero_with_complete_fake_config(self):
        for overrides in ({}, {"PROVIDER": "  OpEnAi  ", "LANGCHAIN_TRACING_V2": " TRUE "}):
            with self.subTest(overrides=overrides):
                result = self.run_cli(**overrides)
                output = result.stdout + result.stderr
                self.assertEqual(result.returncode, 0, output)
                self.assertNotIn(FAKE_ENV["OPENAI_API_KEY"], output)
                self.assertNotIn(FAKE_ENV["LANGCHAIN_API_KEY"], output)

    def test_cli_exit_nonzero_for_invalid_config(self):
        cases = (
            {"PROVIDER": "unknown"},
            {"OPENAI_API_KEY": ""},
            {"OPENAI_API_KEY": "  YOUR_OPENAI_API_KEY_HERE  "},
            {"LANGCHAIN_API_KEY": "YoUr_LangSmith_API_Key_Here"},
            {"LANGCHAIN_TRACING_V2": "false"},
            {"PROVIDER": "anthropic", "OPENAI_API_KEY": ""},
            {"PROVIDER": "openrouter", "OPENAI_API_KEY": ""},
        )
        for environment in cases:
            with self.subTest(environment=environment):
                result = self.run_cli(**environment)
                self.assertNotEqual(
                    result.returncode,
                    0,
                    result.stdout + result.stderr,
                )

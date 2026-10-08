"""Configuration tests never print or use real API keys."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import config


class ConfigTests(unittest.TestCase):
    def valid(self, provider='openai', **keys):
        settings = dict(PROVIDER=provider, LANGSMITH_API_KEY='test-key',
                        OPENAI_API_KEY='test-key', GOOGLE_API_KEY='test-key',
                        ANTHROPIC_API_KEY='test-key', OPENROUTER_API_KEY='test-key')
        settings.update(keys)
        with patch.multiple(config, **settings), patch.dict(config.os.environ, {'LANGCHAIN_TRACING_V2': 'true'}):
            return config.validate()

    def test_valid_openai(self):
        self.assertTrue(self.valid())

    def test_missing_and_placeholder_keys(self):
        for value in ['', 'your_openai_api_key_here', '   ']:
            self.assertFalse(self.valid(OPENAI_API_KEY=value))

    def test_unknown_provider(self):
        self.assertFalse(self.valid('unknown'))

    def test_embedding_keys_required(self):
        for provider in ['anthropic', 'openrouter']:
            self.assertFalse(self.valid(provider, OPENAI_API_KEY=''))

    def test_tracing_required(self):
        with patch.dict(config.os.environ, {'LANGCHAIN_TRACING_V2': 'false'}):
            self.assertFalse(config.validate())

"""Runner tests verify exit propagation without importing real lab steps."""
import importlib
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runner = importlib.import_module("run_all")


class RunnerTests(unittest.TestCase):
    def test_single_step_success_returns_zero(self):
        module = SimpleNamespace(main=lambda: None)
        with patch.object(runner.importlib, "import_module", return_value=module) as load:
            with redirect_stdout(StringIO()):
                exit_code = runner.main(["--step", "2"])
        self.assertEqual(exit_code, 0)
        load.assert_called_once_with("02_prompt_hub_ab_routing")

    def test_single_step_failure_returns_nonzero_without_secret_message(self):
        secret_like_value = "sk-" + "not-a-real-secret-value"

        def fail():
            raise RuntimeError(f"request failed with {secret_like_value}")

        output = StringIO()
        with patch.object(
            runner.importlib, "import_module", return_value=SimpleNamespace(main=fail)
        ), redirect_stdout(output):
            exit_code = runner.main(["--step", "1"])
        self.assertEqual(exit_code, 1)
        self.assertIn("RuntimeError", output.getvalue())
        self.assertNotIn(secret_like_value, output.getvalue())

    def test_default_run_stops_at_first_failed_step(self):
        calls = []

        def module_for(name):
            calls.append(name)
            if name == "02_prompt_hub_ab_routing":
                return SimpleNamespace(main=lambda: (_ for _ in ()).throw(SystemExit(3)))
            return SimpleNamespace(main=lambda: None)

        with patch.object(runner.importlib, "import_module", side_effect=module_for):
            with redirect_stdout(StringIO()):
                exit_code = runner.main([])
        self.assertEqual(exit_code, 1)
        self.assertEqual(
            calls,
            ["01_langsmith_rag_pipeline", "02_prompt_hub_ab_routing"],
        )

    def test_explicit_nonzero_return_is_failure(self):
        module = SimpleNamespace(main=lambda: 2)
        with patch.object(runner.importlib, "import_module", return_value=module):
            with redirect_stdout(StringIO()):
                self.assertEqual(runner.main(["--step", "4"]), 1)


if __name__ == "__main__":
    unittest.main()

"""Offline tests verify retrieval and prompt routing; they do not create evidence."""
import os
import importlib
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import ANY, Mock, call, patch

OFFLINE_ENV = {
    "PYTHON_DOTENV_DISABLED": "1",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "LANGCHAIN_API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "OPENAI_API_KEY": "",
    "GOOGLE_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "OPENROUTER_API_KEY": "",
}
os.environ.update(OFFLINE_ENV)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langsmith.utils import LangSmithConflictError
from utils.data_loader import build_vectorstore
import prompts

step1 = importlib.import_module("01_langsmith_rag_pipeline")
step2 = importlib.import_module("02_prompt_hub_ab_routing")


class RAGTests(unittest.TestCase):
    def test_lcel_retrieves_context_and_question(self):
        store = build_vectorstore(
            ["alpha knowledge", "beta knowledge", "gamma knowledge"],
            DeterministicFakeEmbedding(size=16),
        )
        captured = []

        def respond(prompt):
            captured.append(prompt.to_messages())
            return AIMessage(content="answer from context")

        with patch.object(step1, "get_llm", return_value=RunnableLambda(respond)):
            chain, retriever = step1.build_rag_chain(store)
            self.assertEqual(
                step1.ask(chain, "what is alpha?"), "answer from context"
            )
        self.assertEqual(len(retriever.invoke("alpha")), 3)
        self.assertIn("alpha knowledge", captured[0][0].content)
        self.assertEqual(captured[0][1].content, "what is alpha?")

    def test_step1_uses_required_chunking(self):
        with (
            patch.object(step1, "get_embeddings", return_value=object()),
            patch.object(step1, "load_knowledge_base", return_value="knowledge"),
            patch.object(step1, "split_text", return_value=["chunk"]) as split,
            patch.object(step1, "build_vectorstore", return_value="store") as build,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(step1.setup_vectorstore(), "store")
        split.assert_called_once_with("knowledge", chunk_size=500, chunk_overlap=50)
        build.assert_called_once_with(["chunk"], ANY)

    def test_ab_routes_are_stable_and_both_receive_requests(self):
        first = [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)]
        self.assertEqual(
            first,
            [step2.get_prompt_version(f"req-{i:04d}") for i in range(50)],
        )
        self.assertEqual(set(first), {prompts.PROMPT_V1_NAME, prompts.PROMPT_V2_NAME})
        expected = {
            "req-0000": prompts.PROMPT_V2_NAME,
            "req-0001": prompts.PROMPT_V2_NAME,
            "req-0002": prompts.PROMPT_V1_NAME,
            "req-0049": prompts.PROMPT_V2_NAME,
            "same-user": prompts.PROMPT_V1_NAME,
        }
        self.assertEqual(
            {request_id: step2.get_prompt_version(request_id) for request_id in expected},
            expected,
        )

    def test_question_dataset_has_fifty_entries(self):
        self.assertEqual(len(step1.SAMPLE_QUESTIONS), 50)
        self.assertEqual(step1.SAMPLE_QUESTIONS, step2.SAMPLE_QUESTIONS)

    def test_hub_failure_is_not_fallback_success(self):
        client = Mock()
        client.pull_prompt.side_effect = RuntimeError("Hub unavailable")
        with self.assertRaises(RuntimeError):
            step2.pull_prompts_from_hub(client)

    def test_hub_rejects_templates_missing_required_variables(self):
        invalid_prompts = (
            ChatPromptTemplate.from_template("Question: {question}"),
            ChatPromptTemplate.from_template("Context: {context}"),
            ChatPromptTemplate.from_template(
                "Context: {context}; question: {question}; extra: {extra}"
            ),
        )
        for invalid_prompt in invalid_prompts:
            client = Mock()
            client.pull_prompt.return_value = invalid_prompt
            with self.subTest(variables=invalid_prompt.input_variables):
                with self.assertRaises(ValueError):
                    step2.pull_prompts_from_hub(client)

    def test_hub_push_and_pull(self):
        client = Mock()
        client.pull_prompt.side_effect = [prompts.PROMPT_V1, prompts.PROMPT_V2]
        with redirect_stdout(StringIO()):
            step2.push_prompts_to_hub(client)
            loaded = step2.pull_prompts_from_hub(client)
        self.assertEqual(client.push_prompt.call_count, 2)
        self.assertEqual(set(loaded), {prompts.PROMPT_V1_NAME, prompts.PROMPT_V2_NAME})
        client.push_prompt.assert_has_calls(
            [
                call(
                    prompts.PROMPT_V1_NAME,
                    object=prompts.PROMPT_V1,
                    description=ANY,
                ),
                call(
                    prompts.PROMPT_V2_NAME,
                    object=prompts.PROMPT_V2,
                    description=ANY,
                ),
            ]
        )

    def test_hub_push_only_ignores_unchanged_conflicts(self):
        unchanged = Mock()
        unchanged.push_prompt.side_effect = LangSmithConflictError(
            "Nothing to commit: prompt is unchanged"
        )
        with redirect_stdout(StringIO()):
            step2.push_prompts_to_hub(unchanged)
        self.assertEqual(unchanged.push_prompt.call_count, 2)

        changed_conflict = Mock()
        changed_conflict.push_prompt.side_effect = LangSmithConflictError(
            "Prompt commit conflict"
        )
        with self.assertRaises(LangSmithConflictError), redirect_stdout(StringIO()):
            step2.push_prompts_to_hub(changed_conflict)
        self.assertEqual(changed_conflict.push_prompt.call_count, 1)

    def test_prompts_and_ab_outputs_include_context(self):
        store = build_vectorstore(
            ["fact one", "fact two", "fact three"],
            DeterministicFakeEmbedding(size=16),
        )
        llm = FakeListChatModel(responses=["grounded answer"])
        for version, prompt in prompts.PROMPTS.items():
            self.assertEqual(set(prompt.input_variables), {"context", "question"})
            result = step2.ask_ab(
                store.as_retriever(search_kwargs={"k": 3}),
                llm,
                prompt,
                "question",
                version,
                request_id="req-test",
                batch_id="batch-test",
            )
            self.assertEqual(result["version"], version)
            self.assertEqual(result["request_id"], "req-test")
            self.assertEqual(result["batch_id"], "batch-test")
            self.assertEqual(len(result["contexts"]), 3)
            self.assertEqual(result["answer"], "grounded answer")

    def test_prompts_have_distinct_grounded_instructions(self):
        self.assertNotEqual(prompts.SYSTEM_V1, prompts.SYSTEM_V2)
        self.assertIn("2–4", prompts.SYSTEM_V1)
        self.assertIn("3–5", prompts.SYSTEM_V2)
        for system_prompt in (prompts.SYSTEM_V1, prompts.SYSTEM_V2):
            self.assertIn("{context}", system_prompt)
            self.assertIn("không", system_prompt.casefold())

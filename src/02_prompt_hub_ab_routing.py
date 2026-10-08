"""Step 2: publish and retrieve two Hub prompts, route 50 queries by MD5."""
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import config
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError
from prompts import PROMPT_V1_NAME, PROMPT_V2_NAME, PROMPT_V1, PROMPT_V2
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


def push_prompts_to_hub(client: Client):
    """Publish both prompts; identical commits are safe to reuse."""
    for name, prompt in [(PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)]:
        try:
            url = client.push_prompt(name, object=prompt, description=f'Day22 TuTune04: {name}')
            print(f'✅ Đã push {name} → {url}')
        except LangSmithConflictError as exc:
            if 'nothing to commit' not in str(exc).lower():
                raise
            print(f'ℹ️ Prompt {name} không đổi; dùng phiên bản hiện có trên Hub.')


def pull_prompts_from_hub(client: Client) -> dict:
    """Require actual Hub templates; failure must not masquerade as success."""
    prompts = {}
    for name in [PROMPT_V1_NAME, PROMPT_V2_NAME]:
        prompt = client.pull_prompt(name)
        input_variables = set(prompt.input_variables)
        if input_variables != {"context", "question"}:
            raise ValueError(
                f"Hub prompt {name} cần đúng hai biến context/question; "
                f"nhận được {sorted(input_variables)}"
            )
        prompts[name] = prompt
        print(f"↓ Đã pull '{name}' từ Hub")
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Same request ID always maps to the same prompt name across processes."""
    hash_int = int(hashlib.md5(request_id.encode('utf-8')).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


@traceable(name='ab-rag-query', tags=['ab-test', 'step2'])
def ask_ab(
    retriever,
    llm,
    prompt,
    question: str,
    version: str,
    request_id: str = "",
    batch_id: str = "",
) -> dict:
    """Return the answer and retrieved context with its routing label."""
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]
    answer = (prompt | llm | StrOutputParser()).invoke({
        'context': '\n\n'.join(contexts), 'question': question})
    return {
        "request_id": request_id,
        "batch_id": batch_id,
        "question": question,
        "answer": answer,
        "version": version,
        "contexts": contexts,
    }


def setup_vectorstore():
    """Build the same index as step 1."""
    return build_vectorstore(split_text(load_knowledge_base()), get_embeddings())


def main():
    if not config.validate():
        raise SystemExit(1)
    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)
    retriever = setup_vectorstore().as_retriever(search_kwargs={"k": 3}).with_config(
        {"run_name": "retrieve-context", "tags": ["retrieval", "step2"]}
    )
    llm = get_llm()
    batch_id = datetime.now(timezone.utc).strftime("step2-%Y%m%dT%H%M%SZ")
    lines = [f"Project: {config.LANGSMITH_PROJECT}", f"Batch: {batch_id}"]
    lines.extend(f"↓ Đã pull '{name}' từ Hub" for name in prompts)
    counts = {'v1': 0, 'v2': 0}
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f'req-{i:04d}'
        key = get_prompt_version(request_id)
        version = 'v1' if key == PROMPT_V1_NAME else 'v2'
        result = ask_ab(
            retriever,
            llm,
            prompts[key],
            question,
            version,
            request_id=request_id,
            batch_id=batch_id,
            langsmith_extra={
                "metadata": {
                    "batch_id": batch_id,
                    "request_id": request_id,
                    "prompt_version": version,
                    "prompt_name": key,
                }
            },
        )
        counts[version] += 1
        line = f"[{i+1:02d}] [prompt-{version}] {request_id} Q: {question}\nA: {result['answer']}"
        print(line)
        lines.append(line)
    wait_for_all_tracers()
    summary = f"Routing: V1={counts['v1']} | V2={counts['v2']} | Tổng={sum(counts.values())}"
    print(summary)
    lines.append(summary)
    evidence = Path(__file__).resolve().parents[1] / 'evidence'
    evidence.mkdir(exist_ok=True)
    (evidence / '02_ab_routing_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

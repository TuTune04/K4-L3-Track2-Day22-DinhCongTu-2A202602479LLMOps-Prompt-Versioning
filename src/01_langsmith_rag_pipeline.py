"""Step 1: FAISS retrieval, LCEL generation and 50 LangSmith query traces."""
from datetime import datetime, timezone

import config  # Must precede LangChain imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from langchain_core.tracers.langchain import wait_for_all_tracers
from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


def setup_vectorstore():
    """Embed overlapping knowledge base chunks and index them in FAISS."""
    embeddings = get_embeddings()
    chunks = split_text(load_knowledge_base(), chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")
    return build_vectorstore(chunks, embeddings)


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời. '
     'Nếu thiếu thông tin, nói rõ không biết.\n\nContext:\n{context}'),
    ('human', '{question}'),
])


def build_rag_chain(vectorstore):
    """Return the LCEL chain and its top three document retriever."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}).with_config(
        {"run_name": "retrieve-context", "tags": ["retrieval", "step1"]}
    )
    chain = (
        {
            "context": retriever
            | (lambda docs: "\n\n".join(d.page_content for d in docs)),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT | get_llm() | StrOutputParser()
    )
    return chain, retriever


@traceable(name='rag-query', tags=['rag', 'step1'])
def ask(chain, question: str) -> str:
    """Trace one question, including LCEL retrieval and generation children."""
    return chain.invoke(question)


def main():
    if not config.validate():
        raise SystemExit(1)
    chain, _ = build_rag_chain(setup_vectorstore())
    batch_id = datetime.now(timezone.utc).strftime("step1-%Y%m%dT%H%M%SZ")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        request_id = f"rag-{i - 1:04d}"
        answer = ask(
            chain,
            question,
            langsmith_extra={
                "metadata": {"batch_id": batch_id, "request_id": request_id}
            },
        )
        print(f'[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}\nA: {answer}\n')
    wait_for_all_tracers()
    print(f"✅ Đã xử lý {len(SAMPLE_QUESTIONS)} câu; xác nhận traces trên LangSmith "
          f"project '{config.LANGSMITH_PROJECT}'.")


if __name__ == '__main__':
    main()

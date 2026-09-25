from numpy import sort
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, LLM_MODEL, TOP_K
from vectorstore import similarity_search

_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the "
    "provided context, which comes from the user's own uploaded documents. "
    "If the context doesn't contain the answer, say so clearly instead of "
    "guessing. Cite the source document name(s) when relevant."
)


def _build_prompt(question: str, chunks: list [Document]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        doc_name = chunk.metadata.get("doc_name" "unknown")
        context_blocks.append(f"[source {i}: {doc_name}\n{chunk.page_content}]")

    context = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context: \n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def answer_question(question: str, user_id: str, k: int = TOP_K) -> dict:
    chunks = similarity_search(question, user_id=user_id, k=k)
    prompt = _build_prompt(question, chunks)
    response = _llm.invoke(prompt)
    sources = sorted({c.metadata.get("doc_name", "unknown") for c in chunks})

    return {
        "answer": response.content,
        "sources": sources,
        "chunksj_used": len(chunks),
    }
from botocore.httpsession import where
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import GOOGLE_API_KEY, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME

_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=str(CHROMA_DIR),
)

def add_documents(chunks: list[Document], user_id: str, doc_name: str) -> int:
    for chunk in chunks:
        chunk.metadata["user_id"] = user_id
        chunk.metadata["doc_name"] = doc_name

    _vectorstore.add_documents(chunks)
    return len(chunks)

def similarity_search(query: str, user_id: str, k: int):
    return _vectorstore.similarity_search(
        query=query,
        k=k,
        filter={"user_id": user_id}
    )

def list_user_documents (user_id:str) -> list[dict]:
    raw = _vectorstore.get(where={"user_id": user_id})
    counts: dict[str, int] = {}

    for meta in raw.get("metadatas", []):
        name = meta.get("doc_name", "unknown")
        counts[name] = counts.get(name, 0) + 1

    return [{"doc_name": name, "chunks": count} for name, count in counts.items()]

def delete_user_document(user_id: str, doc_name: str) -> None:
    _vectorstore.delete(where={"$and": [{"user_id": user_id}, {"doc_name": doc_name}]})
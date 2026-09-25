from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pydantic import BaseModel

from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS
from loaders import load_and_chunk
from vectorstore import add_documents, list_user_documents, delete_user_document
from rag_chain import answer_question


app = FastAPI(title="Multi user rag document assistant")

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def server_home():
    return {"status": "server is running"}


@app.post("/users/{user_id}/documents")
async def upload_document(user_id: str, file: UploadFile = File(...)):
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}",
        )

    save_path = UPLOAD_DIR/f"{user_id}_{file.filename}"
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    chunks = load_and_chunk(save_path)
    num_chunks =add_documents(chunks=chunks, user_id=user_id, doc_name=file.filename)

    return {
        "message": f"Successfully processed '{file.filename}'",
        "chunks_created": num_chunks,
    }

@app.get("/users/{user_id}/documents")
def get_documents(user_id: str):
    return {"documents": list_user_documents(user_id)}

@app.delete("/users/{user_id}/documents/{doc_name}")
def delete_document(user_id: str, doc_name: str):
    delete_user_document(user_id=user_id, doc_name=doc_name)
    return {"message": f"Delected '{doc_name}' for user '{user_id}'"}

@app.post("/users/{user_Id}/chat")
def chat(user_id:str, request: ChatRequest):
    return answer_question(question=request.question, user_id=user_id)

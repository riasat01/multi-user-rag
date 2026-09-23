from fastapi import FastAPI

app = FastAPI(title="Multi user rag document assistant")

@app.get("/")
async def server_home():
    return {"status": "server is running"}

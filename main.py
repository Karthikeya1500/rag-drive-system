from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "RAG Drive System Running 🚀"}


@app.post("/sync-drive")
def sync_drive():
    return {"message": "Google Drive sync started"}
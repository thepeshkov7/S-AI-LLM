"""Local inference API for S AI 0.1 LLM. Not production-ready."""
import os
from pathlib import Path
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from model import TinyLLM

app = FastAPI(title="S AI 0.1 LLM", version="0.1.0")
model = None


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_bytes: int = Field(default=80, ge=1, le=256)


@app.on_event("startup")
def load_model():
    global model
    path = Path(os.getenv("SAI_CHECKPOINT", "s_ai_0_1.pt"))
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyLLM(context=checkpoint["context"])
    model.load_state_dict(checkpoint["model"])
    model.eval()


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "model": "S AI 0.1 LLM"}


@app.post("/generate")
def generate(req: GenerateRequest, authorization: str | None = Header(default=None)):
    token = os.getenv("SAI_API_TOKEN")
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained or checkpoint missing")
    ids = list(req.prompt.encode("utf-8"))
    with torch.inference_mode():
        for _ in range(req.max_new_bytes):
            x = torch.tensor([ids[-model.context:]], dtype=torch.long)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1] / 0.9, dim=-1)
            ids.append(int(torch.multinomial(probs, 1).item()))
    return {"model": "S AI 0.1 LLM", "text": bytes(ids).decode("utf-8", errors="replace")}

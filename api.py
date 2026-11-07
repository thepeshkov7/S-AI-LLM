"""Private inference service for the S AI research checkpoint."""
from contextlib import asynccontextmanager
import hmac
import logging
import os
from pathlib import Path
from threading import Lock, Semaphore
from time import monotonic

import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from generate import TextGenerator


MODEL_NAME = "S AI 0.1 LLM"
LOG = logging.getLogger("uvicorn.error")


class InferenceState:
    def __init__(self, token, requests_per_minute, max_in_flight):
        if requests_per_minute < 1 or max_in_flight < 1:
            raise ValueError("Inference limits must be positive")
        self.token = f"Bearer {token}".encode() if token else None
        self.generator = None
        self.limit = requests_per_minute
        self.lock = Lock()
        self.slots = Semaphore(max_in_flight)
        self.window_start = monotonic()
        self.window_count = 0
        self.counts = {status: 0 for status in ("200", "429", "500", "503")}
        self.seconds = 0.0

    def acquire(self):
        with self.lock:
            elapsed = monotonic() - self.window_start
            if elapsed >= 60:
                self.window_start = monotonic()
                self.window_count = 0
                elapsed = 0
            if self.window_count >= self.limit:
                self.counts["429"] += 1
                raise HTTPException(429, "Rate limit exceeded", headers={"Retry-After": str(60 - int(elapsed))})
            if not self.slots.acquire(blocking=False):
                self.counts["429"] += 1
                raise HTTPException(429, "Inference busy", headers={"Retry-After": "1"})
            self.window_count += 1

    def record(self, status, seconds=0.0):
        with self.lock:
            self.counts[status] += 1
            self.seconds += seconds

    def metrics(self):
        with self.lock:
            lines = [
                "# TYPE sai_generate_requests_total counter",
                *(f'sai_generate_requests_total{{status="{status}"}} {count}' for status, count in self.counts.items()),
                "# TYPE sai_generate_seconds_total counter",
                f"sai_generate_seconds_total {self.seconds:.6f}",
                "# TYPE sai_checkpoint_loaded gauge",
                f"sai_checkpoint_loaded {int(self.generator is not None)}",
            ]
        return "\n".join(lines) + "\n"


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(int(os.environ.get("SAI_THREADS", "2")))
    state = InferenceState(
        os.getenv("SAI_API_TOKEN"),
        int(os.environ.get("SAI_REQUESTS_PER_MINUTE", "30")),
        int(os.environ.get("SAI_MAX_IN_FLIGHT", "1")),
    )
    app.state.inference = state
    if state.token is None:
        LOG.warning("SAI_API_TOKEN is missing; service will be unready")
    checkpoint = Path(os.environ.get("SAI_CHECKPOINT", "s_ai_0_1.pt"))
    if not checkpoint.is_file():
        LOG.warning("S AI checkpoint file is missing")
    else:
        try:
            state.generator = TextGenerator.load(checkpoint, os.environ.get("SAI_DEVICE", "cpu"))
            LOG.info("Loaded S AI checkpoint")
        except Exception:
            LOG.exception("Unable to load S AI checkpoint")
    yield


app = FastAPI(
    title=MODEL_NAME,
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exception: RequestValidationError):
    return JSONResponse({"detail": "Invalid generation request"}, status_code=422)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int | None = Field(default=None, ge=1, le=256)
    max_new_bytes: int | None = Field(default=None, ge=1, le=256)
    temperature: float = Field(default=0.85, ge=0.1, le=2.0, allow_inf_nan=False)
    top_k: int = Field(default=40, ge=1, le=256)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)

    @model_validator(mode="after")
    def check_parameters(self):
        if not self.prompt.strip():
            raise ValueError("Prompt must contain text")
        if self.max_new_tokens is not None and self.max_new_bytes is not None:
            raise ValueError("Use only one output limit")
        return self


def authorize(request: Request, authorization: str = Header(default="")):
    state = request.app.state.inference
    if state.token is None or not hmac.compare_digest(authorization.encode(), state.token):
        raise HTTPException(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    return state


@app.get("/health")
def health(request: Request):
    state = request.app.state.inference
    ready = state.generator is not None and state.token is not None
    return JSONResponse(
        {"model": MODEL_NAME, "model_loaded": state.generator is not None, "ready": ready},
        status_code=200 if ready else 503,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(state: InferenceState = Depends(authorize)):
    return PlainTextResponse(
        state.metrics(), media_type="text/plain; version=0.0.4", headers={"Cache-Control": "no-store"},
    )


@app.post("/generate")
def generate(req: GenerateRequest, state: InferenceState = Depends(authorize)):
    if state.generator is None:
        state.record("503")
        raise HTTPException(503, "Model checkpoint unavailable")
    state.acquire()
    started = monotonic()
    try:
        completion, generated_tokens = state.generator.complete(
            req.prompt, req.max_new_tokens or req.max_new_bytes or 80,
            req.temperature, req.top_k, req.seed, req.max_new_bytes,
        )
    except Exception:
        state.record("500", monotonic() - started)
        LOG.exception("S AI generation failed")
        raise HTTPException(500, "Generation failed") from None
    finally:
        state.slots.release()
    state.record("200", monotonic() - started)
    return JSONResponse(
        {
            "model": MODEL_NAME,
            "text": req.prompt + completion,
            "completion": completion,
            "generated_tokens": generated_tokens,
        },
        headers={"Cache-Control": "no-store"},
    )

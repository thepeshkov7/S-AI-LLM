# S AI 0.1 LLM — integration plan

The current code is a tiny byte-level Transformer prototype. It is **not** suitable for replacing a production assistant yet.

1. Train with `python train.py`; the checkpoint is `s_ai_0_1.pt`.
2. Install `pip install -r requirements-api.txt`.
3. Set a strong `SAI_API_TOKEN` in the server environment (never in Git or Android code).
4. Start locally: `uvicorn api:app --host 127.0.0.1 --port 8000`.
5. Check `GET /health`, then test `POST /generate` with Bearer authentication.
6. For production, place the inference service behind HTTPS, authentication, rate limits and monitoring. Keep the token on a trusted backend.
7. Add a configurable model-provider switch to the existing S AI backend, test on a staging environment, and only then remove OpenAI credentials and integrations.

No OpenAI credentials have been removed. No production deployment has been changed.

# S AI 0.1 LLM inference service

Deploy this service separately from the S AI app. The app backend is the only client of the inference service. It checks the user's existing Supabase session, applies per-user quotas, and keeps the inference token in its own server environment. The browser and Android app use the existing S AI backend with their existing session credentials.

## Local check

From the repository root, activate the virtual environment and install `requirements-api.txt`. Set `SAI_CHECKPOINT` to the trained checkpoint and generate a token for the local shell.

```bash
export SAI_CHECKPOINT="$PWD/runs/s_ai_0_1.pt"
export SAI_API_TOKEN="$(openssl rand -hex 32)"
uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

In an authenticated client or another shell configured with the same local token:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/generate \
  -H "Authorization: Bearer $SAI_API_TOKEN" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"Once in a quiet village","max_new_tokens":40,"seed":42}'
curl -fsS http://127.0.0.1:8000/metrics -H "Authorization: Bearer $SAI_API_TOKEN"
```

`GET /health` returns 503 if the checkpoint or token is unavailable. `POST /generate` requires a Bearer token and returns the model name, `completion`, `generated_tokens` and the original `text` field containing the prompt plus continuation. Limits are 1,000 prompt characters and 256 new tokens. Legacy callers may send `max_new_bytes` instead of `max_new_tokens`. Incompatible or invalid requests receive 422, unauthenticated requests receive 401, busy or rate-limited requests receive 429 with `Retry-After`, and unavailable checkpoints receive 503. A generation failure returns 500 without internal error details.

## Dedicated Linux host

Install the code and virtual environment under `/opt/sai-llm` and store the verified checkpoint at `/var/lib/sai-llm/s_ai_0_1.pt`. Allow the dedicated, unprivileged `sai-llm` system account read access. Provision the following environment with the host's secret manager or a root-owned `/etc/sai-llm.env` with permissions `0600`.

```dotenv
SAI_API_TOKEN=<random-256-bit-secret>
SAI_CHECKPOINT=/var/lib/sai-llm/s_ai_0_1.pt
SAI_DEVICE=cpu
SAI_THREADS=2
SAI_REQUESTS_PER_MINUTE=30
SAI_MAX_IN_FLIGHT=1
```

Use one worker because the request limit and metrics are process-local. The rate limit applies across requests authenticated with the service token. For multiple workers or hosts, use a shared limiter in the gateway and monitor each replica.

Example `sai-llm.service` for systemd:

```ini
[Unit]
Description=S AI 0.1 LLM inference
After=network.target

[Service]
User=sai-llm
Group=sai-llm
WorkingDirectory=/opt/sai-llm
EnvironmentFile=/etc/sai-llm.env
ExecStart=/opt/sai-llm/.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1 --limit-concurrency 32 --no-access-log
Restart=on-failure
NoNewPrivileges=true
PrivateTmp=true
ProtectHome=true
ProtectSystem=strict

[Install]
WantedBy=multi-user.target
```

Bind the upstream to loopback and route a dedicated hostname through a TLS proxy. For example, after configuring DNS and firewall rules, Caddy can obtain a certificate automatically:

```caddyfile
inference.example.com {
    request_body {
        max_size 16KB
    }
    reverse_proxy 127.0.0.1:8000 {
        transport http {
            response_header_timeout 30s
        }
    }
}
```

On staging, verify HTTPS and authenticated responses, alert on readiness failures, monitor request counts by outcome and mean generation latency from the Prometheus counters. Keep the service token out of URLs, browser bundles, APKs, logs and Git. Add `SAI_INFERENCE_URL` and `SAI_INFERENCE_TOKEN` to the app backend when integrating.

## S AI rollout

1. Start with the source matching the published web and Android releases. Back up the deploy, source and signed APK, and verify the app can be built from the backup.
2. Put the model selector behind a staging feature flag. Keep the existing provider as the default. For an explicit S AI selection, send supported text inputs to this service, adapt `completion` to the app's existing response format and use the existing history path. On inference failure, show an error for the selected model.
3. Verify authentication, quotas, chat history, web and Android on the same staging backend. Evaluate instruction-following and supported features before deciding on a production migration. The inference endpoint currently accepts plain-text prompts and produces text continuations.
4. Audit the provider call sites separately. Move dependent services or decide how to keep their functionality before removing shared credentials. Change the production default and remove project secrets only after verifying the migration and obtaining the user's go-ahead.

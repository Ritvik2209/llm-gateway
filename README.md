# LLM Gateway

Python 3.11+ project scaffold for an LLM API gateway.

## Health Check

The FastAPI application exposes a single health endpoint:

```http
GET /health
```

Expected response:

```json
{"status": "ok"}
```

# LLM Gateway

A self-hosted API gateway that sits between internal teams and multiple LLM providers, adding routing, failover, per-team rate limiting, budget enforcement, and observability.

This is a portfolio project, built to demonstrate the production-style patterns that sit around an LLM API rather than the model call itself: multi-provider fallback, circuit breaking, cost control, and metrics. Every number in this README comes from an actual measurement or a constant in the source, and the [Known Limitations](#known-limitations) section documents the gaps that testing exposed rather than hiding them. The load-test results were produced against the running stack, and the resilience behavior is covered by an integration suite that drives the full HTTP path. It is intended to be read as an honest engineering artifact, including the parts that are unfinished.

## Features

- **Multi-provider routing** across three implemented providers: **Ollama** (local models), **Groq** (hosted inference), and a built-in **Mock** provider for deterministic testing.
- **Automatic failover** — when a provider fails, the request transparently retries down the team's priority chain, and a per-provider circuit breaker stops hammering a provider that is already failing.
- **Redis-backed rate limiting** using a sliding 60-second window implemented as an atomic Lua script.
- **Redis-backed budget enforcement** with a monthly per-team spend cap, an 80% warning header, and hard rejection at the cap.
- **Per-team configuration** — API key, allowed models, provider priority, an optional injected system prompt, request rate, and monthly budget.
- **Prometheus + Grafana observability**, with the datasource and a five-panel dashboard provisioned as code so the stack comes up already wired.
- **One-command setup** via Docker Compose (gateway, Redis, Prometheus, Grafana).
- **Retry with exponential backoff** on provider calls, and background health probes tracking per-provider status and latency.

## Architecture

A request flows through the gateway in a fixed order, failing fast at the first gate it does not pass:

```
Client
  │
  ▼
Authentication          Bearer API key → team config          401 on unknown key
  │
  ▼
Model authorization     model ∈ team.allowed_models           403 if not allowed
  │
  ▼
Rate limiting           Redis sliding window, 60s             429 + Retry-After
  │
  ▼
Budget check            month-to-date spend vs cap            402 at cap
  │                     ≥80% → X-Budget-Warning header
  ▼
Provider selection      priority chain + circuit breaker      503 if none available
  │                     retry w/ backoff, fall through on failure
  ▼
Response + accounting   record spend, tokens, latency, metrics
```

Provider selection walks the team's `provider_priority` list in order, skipping any provider whose circuit is open, and falling through to the next on failure:

```
team-alpha: provider_priority = [ groq, ollama, mock ]

   ┌────────┐  fails   ┌────────┐  fails   ┌────────┐
   │  groq  │ ───────▶ │ ollama │ ───────▶ │  mock  │
   └────────┘          └────────┘          └────────┘
        │                   │                   │
        └───── success ─────┴─── success ───────┘
                            │
                            ▼
              response.provider = whichever served it
```

Each provider has an independent circuit breaker (`app/circuit_breaker.py`):

```
                4 consecutive failures
   ┌──────────┐ ────────────────────▶ ┌──────────┐
   │  CLOSED  │                       │   OPEN   │
   │ (normal) │ ◀──────────────────── │ (skipped)│
   └──────────┘      success          └──────────┘
         ▲                                  │
         │                                  │ 30s cooldown elapsed
         │           success                ▼
         └───────────────────────── ┌───────────────┐
                                    │   HALF_OPEN   │
                                    │ (one attempt) │
                                    └───────────────┘
```

Thresholds are the defaults in `app/circuit_breaker.py`: `failure_threshold=4`, `cooldown_seconds=30`. Provider calls are retried up to 4 times with exponential backoff (`app/retry.py`: `max_retries=3`, `base_delay=0.5`), and one exhausted provider records exactly one circuit-breaker failure.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Unified chat completion |
| `GET` | `/health` | Liveness check |
| `GET` | `/admin/health` | Per-provider status, circuit state, latencies, error rate |
| `GET` | `/metrics` | Prometheus exposition |
| `POST` | `/admin/mock/toggle-failure` | Force the mock provider to fail, for testing failover |

## Quick Start

**Prerequisites:** Docker and Docker Compose. Optionally [Ollama](https://ollama.com) and a [Groq](https://console.groq.com) API key.

```bash
git clone https://github.com/Ritvik2209/llm-gateway.git
cd llm-gateway
```

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

`.env` is gitignored and is never committed. The Groq key is only needed for teams whose priority chain reaches `groq`; the mock provider works without any credentials.

**A note on Ollama:** it is not containerized and is not part of the Compose stack. You must run it separately on the host (`ollama serve`, with a model pulled, e.g. `ollama pull llama3.2`). `OllamaProvider` defaults to `http://localhost:11434` (`app/providers/ollama_provider.py`), but `docker-compose.yml` overrides it with `OLLAMA_BASE_URL=http://host.docker.internal:11434` so the container can reach the host. If Ollama is not running, requests routed to it fail over to the next provider in the chain.

Start the stack:

```bash
docker-compose up -d --build
```

This brings up four containers: the gateway on `:8000`, Redis on `:6379`, Prometheus on `:9090`, and Grafana on `:3000`.

Send a request. This example uses the `team-loadtest` demo key, which routes only to the mock provider and therefore works immediately with no external dependencies:

```bash
curl -s http://127.0.0.1:8000/v1/chat \
  -H "Authorization: Bearer demo-team-loadtest-local-only" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"messages": [{"role": "user", "content": "hello"}], "model": "mock-model"}'
```

```json
{
  "id": "a56a08bc-8020-42bd-94a0-7e5aca6ed992",
  "model": "mock-model",
  "content": "mock response",
  "input_tokens": 10,
  "output_tokens": 5,
  "provider": "mock",
  "finish_reason": "stop"
}
```

Returns HTTP 200. `id` is a fresh UUID per request, so yours will differ; every other field is fixed
for the mock provider.

To exercise a real provider, use the `team-alpha` key (`demo-team-alpha-local-only`), whose priority chain is `groq → ollama → mock` and which is configured with a system prompt that forces French replies. The demo keys in `config/teams.yaml` are local-only placeholders, not secrets; replace them before running this anywhere real.

## Testing

```bash
pip install -r requirements.txt
pytest
```

**40 tests**, all passing, requiring no network access and no credentials. The suite covers auth, schemas, budget math, the rate-limit window, circuit-breaker transitions, provider fallback selection, health monitoring, streaming, metrics, and system-prompt enrichment.

Six of those are end-to-end integration tests (`tests/test_integration.py`) that drive the full FastAPI request path through `TestClient` — covering the complete request lifecycle, transparent provider fallback with metric assertions, circuit-breaker `closed → open → half_open → closed` transitions, budget warning and cap enforcement, and rate limiting.

## Load Test Results

Measured with `hey` against the running Compose stack. Full methodology, per-run output, and Prometheus cross-checks are in **[LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md)**.

**The gateway saturates at roughly 450 req/s.** Against the mock provider with its simulated delay reduced to ~0, throughput was effectively flat while concurrency increased tenfold:

| Concurrency | Requests/sec | P50 | Errors |
|---|---|---|---|
| 10 | 447.15 | 0.0212 s | 0% |
| 50 | 453.09 | 0.1029 s | 0% |
| 100 | 455.90 | 0.1984 s | 0% |

Under 2% spread across a 10x concurrency increase is a hard ceiling, not a scaling curve — beyond roughly concurrency 10, added load converts into queueing delay rather than throughput. Measured average latency matched Little's Law (`concurrency / throughput`) to within ~1% at every level.

**Gateway overhead is 1-2% of real request latency.** Against real providers at concurrency 5, the gateway's own work is a rounding error next to inference time:

| | Mock baseline | Groq | Ollama (local CPU) |
|---|---|---|---|
| P50 latency | 0.0212 s | 1.2104 s | 2.8800 s |
| Gateway share of P50 | — | ~1.8% | ~0.8% |

The practical conclusion is that gateway overhead is not worth optimizing until provider latency is addressed; real-provider throughput of ~1.5 req/s sits roughly 300x below the gateway's own ceiling.

## Observability

Prometheus scrapes the gateway every 5 seconds (`monitoring/prometheus.yml`). Grafana is provisioned as code — datasource and dashboard are mounted at startup, so no manual setup is required. Grafana is at `http://localhost:3000`, Prometheus at `http://localhost:9090`.

The **LLM Gateway Overview** dashboard (`monitoring/grafana/dashboards/llm-gateway-overview.json`) has five panels:

1. **Request Rate by Provider**
2. **Error Rate by Type**
3. **P95 Latency by Provider**
4. **Circuit Breaker State**
5. **Token Throughput**

<!-- TODO: screenshot not yet captured. Add a real Grafana screenshot at docs/grafana-dashboard.png;
     this image reference is currently a placeholder and will render broken until that file exists. -->
![Grafana Dashboard](docs/grafana-dashboard.png)

Exported metrics: `gateway_requests_total`, `gateway_request_duration_seconds`, `gateway_errors_total`, `gateway_fallback_triggered_total`, `gateway_circuit_breaker_state`, `gateway_tokens_total`.

## Known Limitations

These were found by testing the running system, and are documented rather than papered over. Full detail and reproduction for each is in **[LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md#known-limitations)**.

- **`allowed_providers` is never enforced in routing.** Provider selection filters `provider_priority` against the globally registered providers only (`app/main.py:179-181`), so a team can be served by a provider absent from its allowlist. The field reads as a security control but does not act as one.
- **Budget enforcement is pre-charge, not atomic.** The cap is checked against spend accrued *before* the current request, so the request that crosses the cap still succeeds and only the next one gets a 402 — a team can overspend by up to one request. Concurrent requests can also each observe an under-cap balance and both proceed.
- **Health monitoring is decoupled from routing.** `get_provider_candidates` accepts a `HealthMonitor` but never reads it; only circuit-breaker state affects selection. A provider marked `down` by health checks is still attempted until its circuit opens on real failures.
- **Falling back to the mock provider returns fabricated content as HTTP 200.** If every real provider in a chain fails and `mock` is listed, the client receives `"mock response"` with a success status and no indication the answer is synthetic. Observed live during a real-provider test.
- **The mock provider is priced at $0.00**, so mock traffic never accumulates spend and budget caps cannot be exercised against it without overriding the pricing table in tests.
- **Health-monitor probes consume real provider quota.** Background probes issue real chat completions, so provider spend and rate-limit consumption are a function of uptime, not just request volume.
- **`gateway_request_duration_seconds` measures the provider call only**, excluding auth, rate limiting, budget checks, and queueing. It is not end-to-end latency and should not be compared against client-side measurements.

Also worth noting: `app/providers/anthropic_provider.py` and `app/providers/openai_provider.py` exist but are empty placeholders. Only Ollama, Groq, and Mock are implemented.

## Tech Stack

- **Python 3.12** — FastAPI, Pydantic, httpx, uvicorn
- **Redis** — rate-limit windows and budget accounting
- **Prometheus** — metrics collection
- **Grafana** — dashboards, provisioned as code
- **Docker / Docker Compose** — local orchestration
- **pytest** — unit and integration tests (pytest-asyncio, fakeredis)

## Development Notes

This was built iteratively, with the resilience and cost-control behavior driven out by running the stack under real load and against real providers rather than by unit tests alone. That process surfaced several genuine bugs — including a crash on every successful mock request, a rate limiter that made its own load test impossible, and a silent fallback that returned fabricated content as a success — each of which is documented with reproduction detail in [LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md).

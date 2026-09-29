# LLM Gateway

[![CI](https://github.com/Ritvik2209/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Ritvik2209/llm-gateway/actions/workflows/ci.yml)

A self-hosted API gateway that sits between internal teams and multiple LLM providers, adding routing, failover, per-team rate limiting, budget enforcement, and observability.

This is a portfolio project, built to demonstrate the production-style patterns that sit around an LLM API rather than the model call itself: multi-provider fallback, circuit breaking, cost control, and metrics. Every number in this README comes from an actual measurement or a constant in the source, and the [Known Limitations](#known-limitations) section documents the gaps that testing exposed rather than hiding them. The load-test results were produced against the running stack, and the resilience behavior is covered by an integration suite that drives the full HTTP path. It is intended to be read as an honest engineering artifact, including the parts that are unfinished.

## Features

- **Multi-provider routing** across three implemented providers: **Ollama** (local models), **Groq** (hosted inference), and a built-in **Mock** provider for deterministic testing.
- **Logical model tiers with real failover** — `config/models.yaml` declares which models each provider serves, plus tiers that resolve one logical name to a *different physical model per provider* (`chat-general` is `openai/gpt-oss-20b` on Groq and `llama3.2` on Ollama). That is what lets a fallback chain cross providers serving disjoint model sets. Callers may still name a physical model directly. A provider that cannot serve the request is skipped rather than attempted and failed, and a per-provider circuit breaker stops hammering one that is already failing.
- **Redis-backed rate limiting on two dimensions** — requests per minute and tokens per minute — over a sliding 60-second window. Both limits are evaluated in a single Lua script, so a request rejected on tokens does not consume a request slot. Rejections name which limit was hit.
- **Redis-backed budget enforcement** with a monthly per-team spend cap. Each request's worst-case cost is reserved atomically *before* the provider call and reconciled against actual usage afterwards, so the cap is a hard limit even under concurrency. An 80% crossing sets a warning header.
- **Per-team configuration** — API key, allowed models, provider priority, an optional injected system prompt, request rate, token rate, and monthly budget.
- **Prometheus + Grafana observability**, with the datasource and four dashboards (29 panels) provisioned as code so the stack comes up already wired, plus eight alert rules routed through Alertmanager.
- **Admin API** — view every team's limits, read live usage against each limit, adjust limits at runtime, and read an audit trail of who changed what. Runtime changes are written back into `config/teams.yaml`, so the file stays the single source of truth and the running policy cannot silently diverge from the declared one.
- **Config hot reload** — team and model config is re-read when the files change, validated before it is applied, and swapped in whole. A malformed edit is rejected and the running config keeps serving, so a YAML typo cannot cause an outage. Also exposed as an explicit admin endpoint.
- **One-command setup** via Docker Compose (gateway, Redis, Prometheus, Grafana).
- **Classified retry and fallback** — provider failures are typed rather than stringly-wrapped, so a rejected credential fails immediately instead of consuming 3.5s of backoff, a rejected request does not count against the provider that correctly refused it, and a rate limit the provider says will outlast the backoff budget fails over at once instead of retrying into a wall.
- **Exponential backoff** on retryable provider calls, and **quota-aware health monitoring**: provider health is learned passively from real request outcomes, and synthetic probes are spent only where there is no cheaper signal — a provider never seen, or one whose circuit is open and therefore receiving no traffic. Health is keyed on the provider-model pair, so a provider quota-exhausted on one model is not marked down for the models it is still serving.

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
Rate limiting           requests + tokens, sliding 60s        429 + Retry-After
  │
  ▼
Budget reservation      reserve worst-case cost atomically    402 if it will not fit
  │                     ≥80% → X-Budget-Warning header
  ▼
Provider selection      allowlist → tier/catalog → circuit    503 if none available
  │                     retry w/ backoff, fall through on failure
  ▼
Response + accounting   reconcile reservation vs actual usage
```

Provider selection filters the team's `provider_priority` list in three stages, each answering a different question — *may* this team use the provider (allowlist), *can* it serve this request (model catalog or tier), *should* we attempt it now (circuit breaker) — and falls through to the next candidate on failure. Each candidate carries its own physical model, so one tier request can be served by providers that host different models:

```
request: model = "chat-general"       team-alpha priority = [ groq, ollama ]

   ┌──────────────────────────┐  fails   ┌──────────────────────┐
   │ groq                     │ ───────▶ │ ollama               │
   │ openai/gpt-oss-20b       │          │ llama3.2             │
   └──────────────────────────┘          └──────────────────────┘
              │                                     │
              └────────────── success ──────────────┘
                              │
                              ▼
        response.model    = the physical model that ran
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

Thresholds are the defaults in `app/circuit_breaker.py`: `failure_threshold=4`, `cooldown_seconds=30`. Retryable provider calls are retried up to 4 times with exponential backoff (`app/retry.py`: `max_retries=3`, `base_delay=0.5`), and one exhausted provider records exactly one circuit-breaker failure. Non-retryable failures (`app/providers/errors.py`) skip retry and fallback entirely, and a failure the provider is not at fault for — a request it correctly refused — is excluded from its health and circuit state.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat` | Unified chat completion |
| `GET` | `/health` | Liveness check |
| `GET` | `/admin/health` | Provider status and circuit state, with per-provider-model health nested beneath each — **admin key required** |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/admin/teams` | Every team's limits, and which fields are runtime-editable — **admin key required** |
| `GET` | `/admin/teams/{id}/usage` | Live position against request, token and budget limits — **admin key required** |
| `PATCH` | `/admin/teams/{id}` | Adjust `requests_per_minute`, `tokens_per_minute`, `monthly_budget_usd` — **admin key required** |
| `GET` | `/admin/audit` | Recent administrative changes with before/after values — **admin key required** |
| `GET` | `/admin/config` | Configuration currently in effect, and when it loaded — **admin key required** |
| `POST` | `/admin/config/reload` | Apply config from disk without restarting; 409 if it does not parse — **admin key required** |
| `POST` | `/admin/mock/toggle-failure` | Force the mock provider to fail, for testing failover — **admin key required** |

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

## Demo

With the stack running, `scripts/demo.py` exercises every behaviour end to end against the
live gateway — tiers and failover, capability routing, both rate-limit dimensions, budget
enforcement, retry classification, circuit-breaker transitions, the admin API, validated
hot reload, and the cost and health metrics:

```bash
docker-compose up -d --build
python scripts/demo.py          # 34 checks
```

Each section states what it is proving, shows the real request and response, and asserts
the outcome, so a regression fails loudly rather than printing something plausible. It is
idempotent — it waits out a circuit-breaker cooldown left by a previous run rather than
mismeasuring it — and it restores `config/teams.yaml` byte-for-byte before exiting.

## Testing

```bash
pip install -r requirements.txt
pytest
```

**171 tests**, all passing, requiring no network access and no credentials. The suite covers auth, schemas, budget math, the rate-limit window, circuit-breaker transitions, provider fallback selection, per-provider-model health, streaming resilience, the provider error taxonomy, model tiers, config reload, the admin API, metrics, and system-prompt enrichment.

Fifteen of those are end-to-end integration tests (`tests/test_integration.py`) that drive the full FastAPI request path through `TestClient` — covering the complete request lifecycle, transparent provider fallback with metric assertions, circuit-breaker `closed → open → half_open → closed` transitions, budget reservation, release on failure, and cap enforcement, and rate limiting.

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

Four dashboards are provisioned from `monitoring/grafana/dashboards/`, three of them
purpose-built for a distinct question:

| Dashboard | Answers |
|---|---|
| **Operations** | Is the gateway healthy and are providers behaving? Provider health, provider error rate, rejections by cause, fallback events, circuit-breaker state, config-reload outcomes |
| **Business** | What is each team spending? Budget utilisation, month-to-date spend, spend rate by team and by model, token throughput |
| **Performance** | How much latency does the gateway itself add? Gateway overhead, end-to-end and provider percentiles, per-provider latency, throughput |
| **Overview** | An at-a-glance summary, kept as the quickest single view |

Three details are worth noting, because each is a way a dashboard commonly misleads:

- **State reads as a word.** Provider health and circuit state are stat panels with value
  mappings, so they show `degraded` or `open` rather than only a colour. Colour is a second
  channel, never the only one.
- **Zero is distinguishable from "no data".** A Prometheus counter that has never been
  incremented has no series at all, so a naive panel renders "No data" where it should
  render a reassuring `0`. The rejected-reload and provider-error-rate panels zero-fill
  explicitly.
- **Rejections are not provider failures.** Rate-limit, budget and unknown-model rejections
  carry `provider="none"`, so the provider error rate excludes them from both the numerator
  and the denominator. Otherwise the gateway's own policy decisions would be charged to a
  provider that was never called.

<!-- The screenshot below is of the Overview dashboard and predates the three purpose-built
     boards above. Grafana's image-renderer plugin is not installed, so retaking it is a
     manual step: open the Operations dashboard and replace docs/grafana-dashboard.png. -->
![Grafana Dashboard](docs/grafana-dashboard.png)

Fourteen metrics are exported:

| Metric | What it carries |
|---|---|
| `gateway_requests_total` | Requests by team, model, provider, status |
| `gateway_request_duration_seconds` | Time inside provider calls only |
| `gateway_request_total_duration_seconds` | End-to-end handler time |
| `gateway_overhead_seconds` | The gateway's own work, provider time excluded |
| `gateway_errors_total` | Failures labelled by class from the error taxonomy |
| `gateway_fallback_triggered_total` | Failovers, counted only when availability forced them |
| `gateway_circuit_breaker_state` | 0 closed, 1 half-open, 2 open |
| `gateway_provider_health` | Per provider-model: -1 unknown, 0 healthy, 1 degraded, 2 down |
| `gateway_tokens_total` | Input and output tokens by team and provider |
| `gateway_cost_usd_total` | Spend by team, provider, model |
| `gateway_team_spend_usd` | Month-to-date spend, mirrored from Redis |
| `gateway_team_budget_usd` | Configured cap, so utilisation is a division |
| `gateway_config_reloads_total` | Applied and rejected, both pre-created at zero |
| `gateway_config_loaded_timestamp_seconds` | When the running config was loaded |

All three latency histograms share explicit millisecond-resolution buckets. The
`prometheus_client` defaults begin at 5ms, which put a sub-5ms workload into one
undifferentiated bucket and made every percentile report the bucket edge rather than a
measurement.

Cost is a counter, so spend over any window is a query rather than another metric — `sum(increase(gateway_cost_usd_total[1d])) by (team_id)` gives cost per team per day. Budget utilisation comes from the two gauges, which mirror the authoritative Redis state and are seeded at startup so a restarted gateway does not appear to have reset every budget: `100 * gateway_team_spend_usd / gateway_team_budget_usd`.

## Alerting

Eight rules in `monitoring/alerts.yml`, evaluated by Prometheus and routed through
Alertmanager on `:9093`:

| Alert | Fires when |
|---|---|
| `GatewayProviderErrorRate` | A provider fails more than 10% of requests for 5m |
| `GatewayProviderModelDown` | One provider-model pair has been down for 2m |
| `GatewayProviderDown` | *Every* model of a provider has been down for 2m |
| `GatewayCircuitBreakerOpen` | A circuit has been open for 1m |
| `GatewayTeamApproachingBudget` | A team passes 80% of its monthly cap |
| `GatewayTeamBudgetExhausted` | A team hits 100% and is being refused with 402 |
| `GatewayLatencySLABreached` | Gateway overhead P99 exceeds 50ms for 10m |
| `GatewayConfigReloadRejected` | Config on disk was refused, so it and the running policy have diverged |

Four of these needed more than a threshold. The provider error rate excludes the
synthetic `provider="none"` label, or it would page about a provider that was never
called for a rejection the gateway made itself. The budget rules divide by a guarded
denominator, because a team with no cap would evaluate to `+Inf` and fire forever. The
latency rule watches *overhead* rather than end-to-end latency, so it measures the
gateway's own cost rather than slow inference. And the two provider-down rules are
separated because health is per provider-model: paging "provider is down" for one
failing model would overstate an impairment as an outage, so that case is a warning
naming the model and the critical alert requires `min by (provider_name)` to reach
`down`, meaning no model is still working.

Two inhibitions keep one incident to one page: a provider known to be down suppresses
its own error-rate alert and its per-model alerts, and an exhausted budget suppresses
the 80% warning. The health-derived alerts set a `provider` label explicitly, because
the health metric labels it `provider_name` while the error-rate metric labels it
`provider` — an `equal` match across two different label names silently never fires.

Annotations say what is happening, whether automatic failover is already absorbing it,
and where to look next — not a restatement of the expression. Inhibition rules stop one
incident becoming several pages.

**Slack delivery is configured but unverified.** The receiver is complete apart from the
webhook URL; switching to it is a one-line route change in
`monitoring/alertmanager/alertmanager.yml` plus a real URL, with no code change. Until
then the default receiver posts to a local sink so the whole path can be checked without a
credential:

```bash
python scripts/alert_sink.py     # prints the exact payload Alertmanager would deliver
```

## Known Limitations

These were found by testing the running system, and are documented rather than papered over. Full detail and reproduction for each is in **[LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md#known-limitations)**.

- **Budget reservations are conservative.** Output cost is reserved at the caller's `max_tokens` ceiling, which is what makes the cap a hard limit, but it means a team close to its cap can be refused a request whose actual cost would have fitted. Spend is also held as a float; integer micro-dollars would be the correct representation for money.
- **Health monitoring is decoupled from routing.** `get_provider_candidates` accepts a `HealthMonitor` but never reads it; only circuit-breaker state affects selection. A provider marked `down` by health checks is still attempted until its circuit opens on real failures.
- **The mock provider would return fabricated content as HTTP 200 if a team allowed it.** It is excluded from every real team's chain and blocked by allowlist enforcement, so this is closed by configuration rather than by construction — a team that explicitly allows `mock` can still receive `"mock response"` with a success status.
- **The mock provider is priced at $0.00**, so mock traffic never accumulates spend and budget caps cannot be exercised against it without overriding the pricing table in tests.
- **Only limits are runtime-editable.** Onboarding a team, rotating a key or changing an allowlist still means editing YAML by hand: the admin API deliberately refuses anything but the three numeric limits, since an endpoint that can grant `is_admin` or widen an allowlist is a privilege-escalation path. API keys remain plaintext with no rotation or expiry.
- **Reverting a limit does not restore the file byte-for-byte.** Values written by the admin API round-trip through a Python float, so restoring `5.00` writes `5.0`. Semantically identical, but a revert is not a textual no-op. Holding money as integer micro-dollars would fix this and the float-epsilon issue together.
- **The audit trail is not durable.** Entries go to stdout and to a capped Redis list, but Redis has no volume, so the queryable copy is lost if the container is removed.
- **Only the non-streaming provider paths are classified.** `chat_stream` still raises untyped errors, and a mid-stream failure feeds neither the circuit breaker nor the health monitor, so streaming failures are invisible to both.
- **Health status is still decoupled from routing.** Provider health is now accurate and cheap to collect, but `get_provider_candidates` still does not read it — only circuit-breaker state gates a provider. A probe that detects recovery does not close the circuit; that still requires a real request after the cooldown.
- **`gateway_request_duration_seconds` measures the provider call only**, excluding auth, rate limiting, budget checks, and queueing. It is not end-to-end latency and should not be compared against client-side measurements.

## Tech Stack

- **Python 3.12** — FastAPI, Pydantic, httpx, uvicorn
- **Redis** — rate-limit windows and budget accounting
- **Prometheus** — metrics collection
- **Grafana** — dashboards, provisioned as code
- **Docker / Docker Compose** — local orchestration
- **pytest** — unit and integration tests (pytest-asyncio, fakeredis)

## Development Notes

This was built iteratively, with the resilience and cost-control behavior driven out by running the stack under real load and against real providers rather than by unit tests alone. That process surfaced several genuine bugs — including a crash on every successful mock request, a rate limiter that made its own load test impossible, and a silent fallback that returned fabricated content as a success — each of which is documented with reproduction detail in [LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md).

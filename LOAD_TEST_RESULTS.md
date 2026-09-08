# LLM Gateway — Load Test Results

**Date:** 2026-09-08
**Tool:** `hey` (500/2000/5000 requests at concurrency 10/50/100)
**Endpoint:** `POST http://127.0.0.1:8000/v1/chat`
**Team:** `team-loadtest` (provider `mock`, model `mock-model`, 100000 req/min limit)
**Stack:** Docker Compose — gateway, Redis, Prometheus, Grafana

## Gateway overhead (primary results)

Run with `MockProvider.latency_seconds = 0.001`, so the provider contributes a negligible 1 ms and
essentially all measured time is the gateway's own work.

| Concurrency | Total requests | Duration | Requests/sec | P50 | P95 | P99 | Error rate |
|---|---|---|---|---|---|---|---|
| 10  | 500  | 1.1182 s  | 447.15 | 0.0212 s | 0.0332 s | 0.0356 s | 0.00% (500/500 → 200) |
| 50  | 2000 | 4.4141 s  | 453.09 | 0.1029 s | 0.1703 s | 0.2297 s | 0.00% (2000/2000 → 200) |
| 100 | 5000 | 10.9674 s | 455.90 | 0.1984 s | 0.3020 s | 0.7098 s | 0.00% (5000/5000 → 200) |

Supporting figures from the same `hey` output files:

| Concurrency | Average | Fastest | Slowest | Source file |
|---|---|---|---|---|
| 10  | 0.0223 s | 0.0110 s | 0.0394 s | `loadtest_nolatency_c10.txt` |
| 50  | 0.1096 s | 0.0143 s | 0.2641 s | `loadtest_nolatency_c50.txt` |
| 100 | 0.2168 s | 0.1441 s | 0.8582 s | `loadtest_nolatency_c100.txt` |

### The gateway saturates at roughly 450 req/s

Throughput is flat across all three concurrency levels — 447.15, 453.09, 455.90 req/s — a spread of
under 2% while offered concurrency increases tenfold. This is a hard ceiling, not a scaling curve.
Every unit of concurrency added beyond roughly 10 converts directly into queueing delay rather than
throughput: average latency rises almost exactly in proportion, from 0.0223 s to 0.1096 s to
0.2168 s.

Little's Law confirms the server is fully saturated. For a saturated queue, average latency should
equal concurrency divided by throughput:

| Concurrency | Predicted (c / throughput) | Measured average | Difference |
|---|---|---|---|
| 10  | 10 / 447.15 = 0.0224 s  | 0.0223 s | 0.4% |
| 50  | 50 / 453.09 = 0.1104 s  | 0.1096 s | 0.7% |
| 100 | 100 / 455.90 = 0.2194 s | 0.2168 s | 1.2% |

The fit is within about 1% at every level, which is what a queue in front of a fixed-capacity server
looks like. The practical read is that the gateway's useful operating point is near concurrency 10;
past that, clients pay latency for no additional work completed. The tail degrades faster than the
median under saturation — at concurrency 100 the P99 reaches 0.7098 s and the slowest request
0.8582 s, roughly 3.5x and 4.3x the median.

## With simulated provider latency (comparison)

The original runs, with `MockProvider.latency_seconds = 0.1`, adding a deliberate 100 ms
`asyncio.sleep` to every request to imitate a slow upstream provider. These numbers are **not** a
measurement of gateway overhead; the 100 ms floor dominates them.

| Concurrency | Total requests | Duration | Requests/sec | P50 | P95 | P99 | Error rate |
|---|---|---|---|---|---|---|---|
| 10  | 500  | 5.7957 s  | 86.27  | 0.1121 s | 0.1336 s | 0.1390 s | 0.00% (500/500 → 200) |
| 50  | 2000 | 6.5055 s  | 307.43 | 0.1594 s | 0.1933 s | 0.2217 s | 0.00% (2000/2000 → 200) |
| 100 | 5000 | 10.0485 s | 497.59 | 0.1990 s | 0.2358 s | 0.2952 s | 0.00% (5000/5000 → 200) |

| Concurrency | Average | Fastest | Slowest | Source file |
|---|---|---|---|---|
| 10  | 0.1145 s | 0.1031 s | 0.1451 s | `loadtest_results_c10.txt` |
| 50  | 0.1590 s | 0.1037 s | 0.2360 s | `loadtest_results_c50.txt` |
| 100 | 0.1976 s | 0.1068 s | 0.3435 s | `loadtest_results_c100.txt` |

Comparing the two sets is instructive. At concurrency 10 and 50 the simulated delay was the binding
constraint, so removing it multiplied throughput by 5.2x and 1.5x respectively. At concurrency 100
it did not help at all — throughput went slightly *down*, from 497.59 to 455.90 req/s. That is the
clearest evidence that the earlier concurrency-100 run had already hit the same roughly 450 req/s
wall documented above. The 100 ms sleep releases the event loop, so it partly overlapped with
gateway work rather than adding to it; once removed, the gateway's own capacity is the only limit
left. In other words, the apparent "scaling" from 86 to 307 to 498 req/s in the original run was
mostly the concurrency-bound ceiling of `concurrency / 0.1` lifting, not the gateway getting faster.

## Prometheus cross-check

| Query | Result |
|---|---|
| `gateway_requests_total` | 7501, single series: `team_id="team-loadtest", provider="mock", status="success"` |
| `sum(gateway_request_duration_seconds_count)` | 7501 |
| `histogram_quantile(0.95, sum(rate(gateway_request_duration_seconds_bucket[5m])) by (le))` | 0.0486 s |

The 7501 figure is the 7500 load-test requests plus the single manual verification request, and no
`status="error"` series exists at all — independently confirming the 0% error rate `hey` reported.
Counters reset when the gateway container was rebuilt, so this count covers the primary run only.

One caveat on the latency metric: `gateway_request_duration_seconds` is declared as *"Provider
request latency in seconds"* and, at `app/main.py:220-239`, is started immediately before the
provider call and observed immediately after it. It therefore measures only the provider call, and
excludes authentication, rate limiting, budget checking, response serialization, and any time a
request spends queued before its handler runs. It is not an end-to-end figure and should not be
compared directly against `hey`'s client-side latency — the server-side P95 of 0.0486 s against
`hey`'s 0.3020 s at concurrency 100 is precisely the queueing delay that the saturation analysis
above describes. Only the request counter serves as a true independent cross-check here. In the
earlier 100 ms run these two numbers happened to look similar (0.2425 s server-side against
0.2358 s client-side) purely because the artificial sleep sat inside the measured provider call and
dominated both.

## What this measured

These tests exercised the gateway's own request path — API-key authentication, team config lookup,
model and provider authorization, Redis-backed rate limiting, budget checking and spend accounting,
provider routing through the circuit breaker and health monitor, and Prometheus metric emission —
using the built-in `mock` provider so no real LLM API call was involved. That isolates gateway
performance from the network latency and variable inference time of any upstream provider, which
would otherwise dominate the numbers entirely. With the mock's simulated delay reduced to 1 ms, the
resulting figures are a direct measurement of that fixed overhead: about 22 ms per request at
concurrency 10, with a best observed case of 11 ms, and a sustained ceiling of roughly 450 requests
per second beyond which additional concurrency produces only queueing. All 7500 requests succeeded
at every level, so no failure or saturation-induced error mode was observed within this range —
the ceiling manifests as latency growth, not as dropped or rejected requests.

## Real-Provider Sanity Check

Two small tests against real providers under mild concurrency, to place the gateway's own overhead
in context against real inference latency. These are deliberately not stress tests: 20 requests per
provider at concurrency 5, capped to stay well inside Groq's free-tier allowance.

**Prompt:** `"Say hello in exactly 5 words"` for both. **Date:** 2026-09-09.

| Provider | Team | Model | Requests | Req/sec | P50 | P95 | Average | Fastest | Slowest | Errors |
|---|---|---|---|---|---|---|---|---|---|---|
| Groq   | team-alpha | `openai/gpt-oss-20b` | 20 (c=5) | 1.5774 | 1.2104 s | 8.3627 s | 1.7648 s | 0.7267 s | 8.3627 s | 0 non-200 |
| Ollama | team-beta  | `llama3.2` (local)   | 20 (c=5) | 1.5257 | 2.8800 s | 4.0000 s | 2.9467 s | 1.1799 s | 4.0000 s | 0 non-200 |

`hey` does not compute a P99 at n=20 — it emitted `0%% in 0.0000 secs` for that row in both runs — so
no P99 is reported here. Source files: `realworld_groq_results.txt`, `realworld_ollama_results.txt`.
The Ollama run used `-t 60` to accommodate CPU-bound local inference; nothing timed out, and the
provider's own HTTP client allows a 60 s read (`app/providers/ollama_provider.py:38-43`), so the
~4 s maximum is genuine clustering rather than a cap.

### Contrast with the gateway's own overhead

Measured against the mock-provider baseline at concurrency 10 (P50 0.0212 s, 447 req/s):

| | Mock baseline | Groq | Ollama |
|---|---|---|---|
| P50 latency | 0.0212 s | 1.2104 s | 2.8800 s |
| Multiple of baseline P50 | 1x | ~57x | ~136x |
| Throughput | 447.15 req/s | 1.58 req/s | 1.53 req/s |
| Gateway overhead as share of P50 | ~100% | ~1.8% | ~0.8% |

The gateway's fixed overhead of roughly 22 ms is between one and two orders of magnitude smaller
than the inference time of either real provider. At these latencies the gateway is not the
bottleneck and is not close to being one: real-provider throughput of about 1.5 req/s sits roughly
300x below the gateway's own measured ceiling of ~450 req/s. Groq's hosted inference returned a
median response about 2.4x faster than local CPU-bound Ollama, though Groq's tail was far worse —
its P95 of 8.36 s is an artifact discussed below rather than a representative figure. The practical
conclusion is that gateway overhead is not worth optimizing until provider latency is addressed
first; the ~450 req/s ceiling only becomes relevant with a provider fast enough to approach it.

### Test integrity: one silent fallback in the Groq run

Both runs reported `[200] 20 responses`, but the Groq run's status codes are misleading. Prometheus
shows `team-alpha` requests split 19 to `groq` and 1 to `mock`, with
`gateway_errors_total` recording one `provider_error` each for `groq` and `ollama`, and
`gateway_fallback_triggered_total{from_provider="groq",to_provider="mock"}` at 1. One request failed
on Groq, fell through to Ollama — which does not host `openai/gpt-oss-20b` locally — and was then
served by the mock provider, returning HTTP 200 with the fabricated string `"mock response"`. That
request is almost certainly the 8.3627 s outlier, so the Groq P95 above reflects the cost of
exhausting a fallback chain, not Groq's latency. The P50 of 1.2104 s is the trustworthy figure.

Groq also returned one HTTP 429 during the run, against the gateway's background health probe:

```
Rate limit reached for model `openai/gpt-oss-20b` ... service tier `on_demand`
on tokens per minute (TPM): Limit 8000, Used 7868, Requested 176
```

Twenty requests at concurrency 5 were enough to consume the free tier's 8000 tokens-per-minute
allowance, because the health monitor's own probes compete with test traffic for the same budget.

The Ollama run was clean by contrast: all 20 requests confirmed as `provider="ollama"`,
`model="llama3.2"`, `status="success"`, with no errors and no fallbacks recorded.

### Post-test health

`/admin/health` was checked after both runs. All three providers reported `status: "healthy"` with
`circuit_breaker_state: "closed"` and `consecutive_failures: 0`. Groq's `error_rate` briefly showed
0.111 (one failed probe in nine) immediately after its run and returned to 0.0 once the failure aged
out of the window. No circuit breaker tripped and no provider was left degraded or restricted as a
side effect of these tests.

### Configuration note

Running these tests required two temporary config changes, since `team-alpha` (10 req/min) and
`team-beta` (5 req/min) would otherwise have had most of the 20 requests rejected with 429 by the
gateway's own rate limiter. Both teams were raised to 30 req/min, and `team-beta`'s
`provider_priority` was corrected from `["mock", "ollama"]` to `["ollama"]` so its traffic would
reach Ollama rather than the mock provider. All three values were reverted after testing and
`config/teams.yaml` verified byte-identical to its pre-test state.

## Known Limitations

### `allowed_providers` is never enforced in routing

A team's `allowed_providers` list is loaded from config but never consulted when choosing a
provider. Routing is decided entirely by `provider_priority`.

`get_provider_candidates` at `app/main.py:168-186` builds its candidate list like this:

```python
    provider_priority = team_config.get(          # app/main.py:175-178
        "provider_priority",
        team_config.get("allowed_providers", []),
    )
    attempted_providers = [                        # app/main.py:179-181
        provider_name for provider_name in provider_priority if provider_name in providers
    ]
```

The filter at `app/main.py:179-181` tests `provider_name in providers` — the dict of *globally
registered* providers — not membership in the team's `allowed_providers`. `allowed_providers` is
loaded at `app/config.py:42` and thereafter used only as a *fallback default* for
`provider_priority` when that key is absent (`app/config.py:43-46`, and the same pattern at
`app/main.py:155-157`, `175-178`, `194-197`, `250-253`). If `provider_priority` is present, the
allowlist has no effect whatsoever.

Consequence: any provider named in `provider_priority` will be used even when the team's allowlist
excludes it. This was observed live — `team-beta` was configured `allowed_providers: ["ollama"]`
with `provider_priority: ["mock", "ollama"]`, and every request was served by `mock`, a provider
absent from its allowlist. The allowlist reads as a security control but does not act as one.

A fix would intersect the two lists before building candidates, e.g. filtering on
`provider_name in providers and provider_name in team_config.get("allowed_providers", [])`, and
rejecting the request when the intersection is empty rather than silently falling through.

### Fallback to the mock provider returns fabricated content as HTTP 200

When every real provider in a team's priority list fails, the chain continues into `mock` if it is
listed, and the client receives HTTP 200 containing the literal string `"mock response"` with no
indication the answer is synthetic. This is not hypothetical: during the Groq sanity check above,
1 of 20 requests failed on `groq`, then failed on `ollama`, then was served by `mock` and returned
200. `hey` recorded `[200] 20 responses`, so the failure was invisible from the client side and
visible only in `gateway_requests_total{provider="mock"}` and
`gateway_fallback_triggered_total{from_provider="groq",to_provider="mock"}`.

For any team whose traffic is not a load test, `mock` should not appear in `provider_priority` at
all, and exhausting the real providers should surface an error rather than a fabricated success.

### Health-monitor probes consume real provider quota

The background health monitor issues real chat completions against each registered provider,
including `groq`. These probes count against the account's token-per-minute allowance and compete
with live traffic — one probe was rate-limited by Groq during the sanity check purely because test
traffic had consumed the TPM budget. Provider quota consumption is therefore a function of uptime,
not just of request volume.

### `gateway_request_duration_seconds` is provider-only latency

Recorded around the provider call alone (`app/main.py:220-239`), excluding auth, rate limiting,
budget checks, serialization, and queueing. It is not an end-to-end latency metric and should not
be compared against client-side measurements. See the Prometheus cross-check section above.

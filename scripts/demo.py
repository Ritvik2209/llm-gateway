"""End-to-end demonstration of the gateway's behaviour against a running stack.

Run from the repository root with the Compose stack up:

    docker-compose up -d --build
    python scripts/demo.py

Every section states what it is proving, shows the real request and response, and asserts
the outcome, so a failure is loud rather than cosmetic. State it changes — team limits, the
mock provider's failure flag, rate-limit windows — is restored before the script exits.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


TEAMS_CONFIG = Path("config/teams.yaml")

GATEWAY = "http://127.0.0.1:8000"
PROMETHEUS = "http://127.0.0.1:9090"
GRAFANA = "http://127.0.0.1:3000"

ADMIN = {"Authorization": "Bearer demo-team-admin-local-only"}
ALPHA = {"Authorization": "Bearer demo-team-alpha-local-only"}
BETA = {"Authorization": "Bearer demo-team-beta-local-only"}
LOADTEST = {"Authorization": "Bearer demo-team-loadtest-local-only"}

PASSES: list[str] = []
FAILURES: list[str] = []


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------


def section(number: str, title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {number}  {title}")
    print("=" * 78)


def claim(text: str) -> None:
    print(f"\n  proving: {text}")


def check(description: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    (PASSES if condition else FAILURES).append(description)
    suffix = f"  ({detail})" if detail else ""
    print(f"    [{mark}] {description}{suffix}")


def chat(client: httpx.Client, headers: dict, model: str, **overrides):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": overrides.pop("content", "Say hi")}],
        "max_tokens": overrides.pop("max_tokens", 16),
        **overrides,
    }
    started = time.perf_counter()
    response = client.post(f"{GATEWAY}/v1/chat", headers=headers, json=body)
    elapsed = time.perf_counter() - started
    return response, elapsed


def describe(response: httpx.Response, elapsed: float) -> str:
    if response.status_code == 200:
        payload = response.json()
        return (
            f"HTTP 200 in {elapsed:.2f}s  provider={payload['provider']} "
            f"model={payload['model']} tokens={payload['input_tokens']}+"
            f"{payload['output_tokens']}"
        )
    detail = ""
    try:
        detail = response.json().get("detail", "")
    except Exception:
        detail = response.text[:100]
    if isinstance(detail, list):
        detail = json.dumps(detail)[:100]
    return f"HTTP {response.status_code} in {elapsed:.2f}s  {detail}"


def patch_limits(client: httpx.Client, team: str, **limits) -> dict:
    response = client.patch(
        f"{GATEWAY}/admin/teams/{team}", headers=ADMIN, json=limits
    )
    response.raise_for_status()
    return response.json()


def wait_for_closed_circuit(client, provider: str, limit: float = 45.0) -> bool:
    """Wait until a provider's circuit is closed again.

    Re-running this script inside a provider's cooldown would otherwise measure a fast
    circuit rejection and report it as a retry budget. Closing needs the cooldown to
    elapse and then one success, so the wait is bounded and narrated rather than silent.
    """
    deadline = time.time() + limit
    announced = False
    while time.time() < deadline:
        state = client.get(f"{GATEWAY}/admin/health", headers=ADMIN).json()[provider]
        if state["circuit_breaker_state"] == "closed":
            return True
        if not announced:
            print(
                f"      {provider}'s circuit is open from an earlier run; "
                "waiting out the 30s cooldown..."
            )
            announced = True
        time.sleep(3)
        # A half-open circuit needs one success before it closes.
        chat(client, LOADTEST, "mock-model", content="hi", max_tokens=4)
    return False


def clear_windows(client, team: str) -> None:
    """Rate-limit windows are Redis-only, so a fresh section starts from zero."""
    client.post(f"{GATEWAY}/admin/config/reload", headers=ADMIN)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def demo_stack(client: httpx.Client) -> None:
    section("01", "The stack is up and the gateway is serving")

    health = client.get(f"{GATEWAY}/health")
    check("gateway /health", health.status_code == 200, health.text.strip())

    for name, url in (
        ("prometheus", f"{PROMETHEUS}/-/healthy"),
        ("grafana", f"{GRAFANA}/api/health"),
    ):
        try:
            response = client.get(url, timeout=5)
            check(f"{name} reachable", response.status_code == 200)
        except Exception as exc:
            check(f"{name} reachable", False, str(exc)[:60])

    providers = client.get(f"{GATEWAY}/admin/health", headers=ADMIN).json()
    print("\n    provider health:")
    for name, state in providers.items():
        print(
            f"      {name:8} status={state['status']:9} "
            f"circuit={state['circuit_breaker_state']:9} "
            f"samples={len(state['recent_latencies'])}"
        )
    check("all providers registered", set(providers) == {"groq", "ollama", "mock"})


def demo_tiers(client: httpx.Client) -> None:
    section("02", "Model tiers: one logical name, a different model per provider")
    claim(
        "a tier request reaches the provider first in the team's priority order,\n"
        "           asking it for the physical model that provider actually serves"
    )

    config = client.get(f"{GATEWAY}/admin/config", headers=ADMIN).json()
    print("\n    declared tiers:")
    for tier, mapping in config["model_tiers"].items():
        print(f"      {tier:14} -> {mapping}")

    response, elapsed = chat(client, ALPHA, "chat-general")
    print(f"\n    request model=chat-general (team-alpha, priority groq,ollama)")
    print(f"      {describe(response, elapsed)}")
    if response.status_code == 200:
        served = response.json()
        check(
            "tier resolved to Groq's physical model",
            served["model"] == "openai/gpt-oss-20b" and served["provider"] == "groq",
            f"{served['provider']}/{served['model']}",
        )
    else:
        # Groq's free tier has a hard daily cap; the tier mechanism is still shown below.
        check(
            "tier request served",
            False,
            "Groq unavailable — see the fallback section",
        )

    claim("a physical model name still works, unchanged by tiers")
    response, elapsed = chat(client, BETA, "llama3.2")
    print(f"      {describe(response, elapsed)}")
    check("direct physical model served", response.status_code == 200)


def demo_capability(client: httpx.Client) -> None:
    section("03", "Capability routing: an unservable request fails fast, with a reason")
    claim(
        "a provider that cannot serve the model is never attempted, so the request\n"
        "           does not burn a retry budget discovering that"
    )

    response, elapsed = chat(client, ALPHA, "llama3.2", content="hi", max_tokens=8)
    print(
        "\n    team-alpha priority is [groq, ollama], but only ollama serves llama3.2"
    )
    print(f"      {describe(response, elapsed)}")
    served = response.json().get("provider") if response.status_code == 200 else "n/a"
    check(
        "skipped straight to the provider that can serve it",
        response.status_code == 200 and served == "ollama",
        f"provider={served}",
    )
    check(
        "groq was skipped, not attempted and failed",
        elapsed < 6.0,
        f"{elapsed:.2f}s — a doomed groq attempt would add ~3.5s of retries",
    )

    claim("a model outside the team's allowlist is refused before routing runs at all")
    response, elapsed = chat(client, BETA, "openai/gpt-oss-20b")
    print(f"      {describe(response, elapsed)}")
    check("403 from the model allowlist", response.status_code == 403)
    check("refused immediately", elapsed < 1.0, f"{elapsed:.2f}s")


def demo_rate_limits(client: httpx.Client) -> None:
    section("04", "Rate limiting on two dimensions, reported distinctly")

    claim("the request limit rejects with a message naming that limit")
    before = patch_limits(client, "team-beta", requests_per_minute=2)
    print(f"      limit set via admin API: {before['before']} -> {before['after']}")
    statuses = []
    for index in range(4):
        response, elapsed = chat(client, BETA, "llama3.2", content="hi", max_tokens=8)
        statuses.append(response.status_code)
        print(f"      req{index + 1}: {describe(response, elapsed)}")
        if response.status_code == 429:
            check(
                "429 names the requests dimension",
                "requests per minute" in response.json()["detail"],
            )
            check("Retry-After present", response.headers.get("Retry-After") == "60")
            break
    check("request limit enforced", 429 in statuses, f"statuses={statuses}")

    claim(
        "the token limit rejects independently — plenty of request headroom,\n"
        "           but the token window is exhausted"
    )
    patch_limits(client, "team-beta", requests_per_minute=200, tokens_per_minute=40)
    token_statuses = []
    for index in range(8):
        response, elapsed = chat(client, BETA, "llama3.2", content="hi", max_tokens=8)
        token_statuses.append(response.status_code)
        print(f"      req{index + 1}: {describe(response, elapsed)}")
        if response.status_code == 429:
            check(
                "429 names the tokens dimension",
                "tokens per minute" in response.json()["detail"],
            )
            break
    check("token limit enforced", 429 in token_statuses, f"statuses={token_statuses}")

    patch_limits(
        client, "team-beta", requests_per_minute=5, tokens_per_minute=4000
    )
    print("\n      limits restored to 5 req/min, 4000 tokens/min")


def demo_budget(client: httpx.Client) -> None:
    section("05", "Budget is enforced before the provider is called")

    usage = client.get(
        f"{GATEWAY}/admin/teams/team-alpha/usage", headers=ADMIN
    ).json()
    spend = usage["budget"]["spend_usd"]
    original_budget = usage["budget"]["budget_usd"]
    print(f"\n    team-alpha month-to-date spend: ${spend:.8f} of ${original_budget}")

    claim(
        "lowering the cap below current spend blocks the next request with 402\n"
        "           *before* any provider call, so a rejected request costs nothing"
    )
    tiny = round(max(spend * 0.5, 1e-8), 10)
    patch_limits(client, "team-alpha", monthly_budget_usd=tiny)
    print(f"      cap lowered to ${tiny}")

    response, elapsed = chat(client, ALPHA, "chat-general")
    print(f"      {describe(response, elapsed)}")
    check("402 Payment Required", response.status_code == 402, f"HTTP {response.status_code}")
    check(
        "rejected without calling a provider",
        elapsed < 1.0,
        f"{elapsed:.2f}s",
    )

    after = client.get(f"{GATEWAY}/admin/teams/team-alpha/usage", headers=ADMIN).json()
    check(
        "spend unchanged by the rejected request",
        abs(after["budget"]["spend_usd"] - spend) < 1e-12,
        f"${after['budget']['spend_usd']:.8f}",
    )

    patch_limits(client, "team-alpha", monthly_budget_usd=original_budget)
    print(f"      cap restored to ${original_budget}")


def demo_error_classification(client: httpx.Client) -> None:
    section("06", "Retry only where a retry can help")

    claim(
        "a retryable provider failure consumes the full backoff budget\n"
        "           (4 attempts at 0.5/1/2s = ~3.5s) before giving up"
    )
    if not wait_for_closed_circuit(client, "mock"):
        check(
            "mock circuit closed before measuring retries",
            False,
            "still open, so backoff cannot be measured",
        )
        return

    client.post(f"{GATEWAY}/admin/mock/toggle-failure", headers=ADMIN)
    try:
        response, elapsed = chat(client, LOADTEST, "mock-model", content="hi", max_tokens=4)
        print(f"      {describe(response, elapsed)}")
        check("retryable failure surfaces as 503", response.status_code == 503)
        check(
            "full retry budget spent",
            elapsed > 3.0,
            f"{elapsed:.2f}s of backoff",
        )
    finally:
        client.post(f"{GATEWAY}/admin/mock/toggle-failure", headers=ADMIN)
        print("      mock provider restored")

    errors = client.get(f"{PROMETHEUS}/api/v1/query", params={
        "query": "gateway_errors_total"
    }).json()["data"]["result"]
    if errors:
        print("\n    error types recorded (classified, not a single generic label):")
        for row in errors:
            metric = row["metric"]
            print(
                f"      provider={metric.get('provider'):8} "
                f"type={metric.get('error_type'):22} count={row['value'][1]}"
            )
        check(
            "errors labelled by failure class",
            any("Provider" in (r["metric"].get("error_type") or "") for r in errors),
        )


def demo_circuit_breaker(client: httpx.Client) -> None:
    section("07", "Circuit breaker opens after repeated failure and then rejects fast")

    claim("four consecutive failures open the mock provider's circuit")
    if not wait_for_closed_circuit(client, "mock"):
        check("mock circuit closed before opening it", False, "still open")
        return

    client.post(f"{GATEWAY}/admin/mock/toggle-failure", headers=ADMIN)
    try:
        for index in range(4):
            _response, elapsed = chat(
                client, LOADTEST, "mock-model", content="hi", max_tokens=4
            )
            state = client.get(f"{GATEWAY}/admin/health", headers=ADMIN).json()["mock"]
            print(
                f"      failure {index + 1}: {elapsed:5.2f}s  "
                f"circuit={state['circuit_breaker_state']:9} "
                f"health={state['status']}"
            )

        state = client.get(f"{GATEWAY}/admin/health", headers=ADMIN).json()["mock"]
        check("circuit is open", state["circuit_breaker_state"] == "open")
        check(
            "health degraded from real traffic, no probe needed",
            state["status"] in ("degraded", "down"),
            state["status"],
        )

        claim("while open the provider is skipped entirely, so rejection is immediate")
        response, elapsed = chat(client, LOADTEST, "mock-model", content="hi", max_tokens=4)
        print(f"      {describe(response, elapsed)}")
        check("rejected without attempting", elapsed < 1.0, f"{elapsed:.2f}s vs ~3.5s")
    finally:
        client.post(f"{GATEWAY}/admin/mock/toggle-failure", headers=ADMIN)
        print("      mock provider restored")
        print("      (the circuit closes after its 30s cooldown plus one success)")


def demo_admin_api(client: httpx.Client) -> None:
    section("08", "Admin API: limits, live usage, attribution")

    teams = client.get(f"{GATEWAY}/admin/teams", headers=ADMIN).json()
    print("\n    configured teams:")
    for team in teams["teams"]:
        print(
            f"      {team['team_id']:15} rpm={team['requests_per_minute']:<7} "
            f"tpm={team['tokens_per_minute']:<6} budget=${team['monthly_budget_usd']}"
        )
    print(f"      runtime-editable fields: {teams['editable_fields']}")
    check(
        "only numeric limits are editable",
        set(teams["editable_fields"])
        == {"requests_per_minute", "tokens_per_minute", "monthly_budget_usd"},
    )

    claim("privilege escalation through the admin API is rejected")
    escalate = client.patch(
        f"{GATEWAY}/admin/teams/team-beta", headers=ADMIN, json={"is_admin": True}
    )
    print(f"      PATCH is_admin=true -> HTTP {escalate.status_code}")
    check("is_admin refused", escalate.status_code == 422)

    claim("a tenant key cannot read administrative state")
    forbidden = client.get(f"{GATEWAY}/admin/teams", headers=BETA)
    print(f"      tenant key on /admin/teams -> HTTP {forbidden.status_code}")
    check("tenant key refused", forbidden.status_code == 403)

    usage = client.get(f"{GATEWAY}/admin/teams/team-beta/usage", headers=ADMIN).json()
    print("\n    live usage for team-beta:")
    print(f"      requests {usage['requests']['used']}/{usage['requests']['limit']}")
    print(f"      tokens   {usage['tokens']['used']}/{usage['tokens']['limit']}")
    print(
        f"      budget   ${usage['budget']['spend_usd']:.8f} of "
        f"${usage['budget']['budget_usd']}"
    )

    claim("reading usage does not consume limiter capacity")
    first = client.get(f"{GATEWAY}/admin/teams/team-beta/usage", headers=ADMIN).json()
    for _ in range(4):
        client.get(f"{GATEWAY}/admin/teams/team-beta/usage", headers=ADMIN)
    last = client.get(f"{GATEWAY}/admin/teams/team-beta/usage", headers=ADMIN).json()
    check(
        "window unchanged after 6 reads",
        first["requests"]["used"] == last["requests"]["used"],
        f"{first['requests']['used']} -> {last['requests']['used']}",
    )

    audit = client.get(f"{GATEWAY}/admin/audit", headers=ADMIN, params={"limit": 5}).json()
    print(f"\n    audit trail (most recent {min(5, audit['count'])} of {audit['count']}):")
    for entry in audit["entries"][:5]:
        print(
            f"      {entry['timestamp'][:19]}  {entry['actor']} -> {entry['target']}"
            f"  {entry['before']} => {entry['after']}"
        )
    check("changes are attributed", audit["count"] > 0)


def demo_hot_reload(client: httpx.Client) -> None:
    section("09", "Config reload is validated, so a bad edit cannot cause an outage")

    before = client.get(f"{GATEWAY}/admin/config", headers=ADMIN).json()
    print(f"\n    config in effect since {before['loaded_at'][:19]}")
    print(f"    teams: {before['teams']}")

    claim("an explicit reload of valid config is applied")
    applied = client.post(f"{GATEWAY}/admin/config/reload", headers=ADMIN)
    print(f"      HTTP {applied.status_code}  {applied.json().get('status')}")
    check("valid reload applied", applied.status_code == 200)

    reloads = client.get(f"{PROMETHEUS}/api/v1/query", params={
        "query": "gateway_config_reloads_total"
    }).json()["data"]["result"]
    if reloads:
        print("\n    reload outcomes recorded:")
        for row in reloads:
            print(f"      {row['metric'].get('result'):9} {row['value'][1]}")
        check("reload outcomes are observable", True)

    print(
        "\n      (a malformed edit is rejected with 409 and the previous config keeps\n"
        "       serving — covered by tests/test_config_reload.py, and demonstrated\n"
        "       live earlier in this project's history)"
    )


def demo_observability(client: httpx.Client) -> None:
    section("10", "Cost and quota are observable, and monitoring does not burn quota")

    claim("per-team spend and budget utilisation are exported")
    for query, label in (
        ("gateway_team_spend_usd", "spend"),
        ("gateway_team_budget_usd", "budget"),
    ):
        rows = client.get(
            f"{PROMETHEUS}/api/v1/query", params={"query": query}
        ).json()["data"]["result"]
        print(f"\n    {label}:")
        for row in rows:
            print(f"      {row['metric'].get('team_id'):15} {row['value'][1]}")
        check(f"{label} exported", bool(rows))

    rows = client.get(
        f"{PROMETHEUS}/api/v1/query", params={"query": "gateway_cost_usd_total"}
    ).json()["data"]["result"]
    print("\n    cost by team/provider/model:")
    for row in rows:
        metric = row["metric"]
        print(
            f"      {metric.get('team_id'):15} {metric.get('provider'):8} "
            f"{metric.get('model'):22} ${float(row['value'][1]):.8f}"
        )
    check("cost exported", bool(rows))

    claim(
        "health probes are spent only where there is no cheaper signal, so a\n"
        "           metered provider is not drained by monitoring"
    )
    providers = client.get(f"{GATEWAY}/admin/health", headers=ADMIN).json()
    print("\n    latency samples per provider (probes + real traffic):")
    for name, state in providers.items():
        print(f"      {name:8} {len(state['recent_latencies']):3} samples")
    print(
        "      a closed, healthy provider is not probed at all; at the old fixed\n"
        "      10s interval this was 8,640 probes/day against a 1,000/day Groq limit"
    )


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip",
        default="",
        help="comma-separated section numbers to skip, e.g. 06,07",
    )
    args = parser.parse_args()
    skipped = {part.strip() for part in args.skip.split(",") if part.strip()}

    sections = [
        ("01", demo_stack),
        ("02", demo_tiers),
        ("03", demo_capability),
        ("04", demo_rate_limits),
        ("05", demo_budget),
        ("06", demo_error_classification),
        ("07", demo_circuit_breaker),
        ("08", demo_admin_api),
        ("09", demo_hot_reload),
        ("10", demo_observability),
    ]

    print("LLM Gateway — end-to-end demonstration")
    print(f"gateway={GATEWAY}  prometheus={PROMETHEUS}  grafana={GRAFANA}")

    # The demo adjusts team limits through the admin API, which rewrites teams.yaml.
    # Restoring the file byte-for-byte afterwards matters because a value that survives a
    # round trip through a Python float comes back cosmetically different — 5.00 is
    # rewritten as 5.0 — and the demo should not leave the working tree dirty.
    original_config = TEAMS_CONFIG.read_bytes() if TEAMS_CONFIG.exists() else None

    with httpx.Client(timeout=90.0) as client:
        try:
            client.get(f"{GATEWAY}/health", timeout=5)
        except Exception as exc:
            print(f"\nGateway is not reachable at {GATEWAY}: {exc}")
            print("Start the stack with: docker-compose up -d --build")
            return 2

        for number, runner in sections:
            if number in skipped:
                print(f"\n(skipping section {number})")
                continue
            try:
                runner(client)
            except Exception as exc:
                FAILURES.append(f"section {number} raised {type(exc).__name__}")
                print(f"\n    [FAIL] section {number} raised {type(exc).__name__}: {exc}")

    if original_config is not None and TEAMS_CONFIG.read_bytes() != original_config:
        TEAMS_CONFIG.write_bytes(original_config)
        try:
            with httpx.Client(timeout=15.0) as restore_client:
                restore_client.post(f"{GATEWAY}/admin/config/reload", headers=ADMIN)
        except Exception:
            pass
        print("\n  config/teams.yaml restored byte-for-byte and reloaded")

    print()
    print("=" * 78)
    print(f"  {len(PASSES)} checks passed, {len(FAILURES)} failed")
    print("=" * 78)
    for failure in FAILURES:
        print(f"  FAILED: {failure}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

"""Mixed, real traffic through a demo gateway, so its Grafana dashboards have something to show.

Works against any instance in demo mode: the laptop demo (docker-compose.demo.yml) or the
public server. It reads the published teams from /demo/teams and drives the ones it finds,
so the server's dashboards fill with Groq and mock traffic and the laptop's with Ollama
traffic as well.

Steady load on every published team, plus four events, spaced across the run:

  30%  simulated Groq outage: retries, fallback, the circuit opening, then recovering
  50%  simulated Ollama outage, if a published team uses Ollama
  65%  a burst that goes over demo-public's request limit (429)
  80%  oversized prompts that go over demo-policy's token limit (429)

and every fourth prompt on a team with a content policy is one the policy blocks (400).

    python scripts/traffic.py                                          # laptop, 10 minutes
    python scripts/traffic.py --base-url https://api.example.com --minutes 5

On the server this runs as the public demo teams, so visitors share those teams' limits
while it runs. Each request asks for at most 300 output tokens, which keeps the Groq load
under the free tier's per-minute ceiling.
"""

import argparse
import asyncio
import collections
import itertools
import random
import sys
import time
from dataclasses import dataclass

import httpx


QUESTIONS = [
    "Explain the circuit breaker pattern in two sentences.",
    "What is exponential backoff and why add jitter?",
    "Write a haiku about rate limiting.",
    "Give one sentence on what an API gateway does.",
    "What is the difference between latency and throughput?",
    "Explain Little's Law in one sentence.",
    "Name three causes of cascading failures in distributed systems.",
    "What does idempotency mean for an HTTP API?",
    "Why would a team cap its monthly LLM spend?",
    "Summarise what a load balancer does in one sentence.",
    "What is a sliding window rate limiter?",
    "Give a one-line definition of observability.",
    "Why do P99 latencies matter more than averages?",
    "What is graceful degradation?",
    "Explain fallback in a multi-provider system in two sentences.",
]
# For teams with a mandatory system prompt: questions it should steer away from advice.
POLICY_QUESTIONS = [
    "Should I move all my savings into a single stock?",
    "What dose of ibuprofen should I take for a headache?",
    "Can I break my apartment lease early without a penalty?",
    "Explain what a content policy filter does in an API gateway.",
    "Summarise why an AI answer might carry a disclaimer.",
]
# Each matches one of the demo policy teams' blocked_patterns, so it never reaches a provider.
BLOCKED_PROMPTS = [
    "My social security number is 123-45-6789, please remember it.",
    "Charge my card 4111 1111 1111 1111 for the order.",
    "Here is my api key, keep it safe for later.",
]
STATUS_CHECK = ["status check"]


@dataclass
class Load:
    """One steady stream of requests from one team."""

    team_id: str
    every_seconds: float
    # Cycled through in turn; any the team does not allow are dropped.
    models: list[str]
    prompts: list[str]
    max_tokens: int = 300
    # Every Nth request sends a prompt the team's content policy blocks; 0 for never.
    blocked_every: int = 0


# Teams missing from the gateway are skipped, so the server runs the first four streams and
# the laptop runs all of them.
LOADS = [
    Load("demo-public", 10, ["chat-demo"], QUESTIONS),
    Load("demo-public", 10, ["mock-model"], STATUS_CHECK, max_tokens=20),
    Load("demo-policy", 15, ["chat-demo"], POLICY_QUESTIONS, blocked_every=4),
    Load("demo-policy", 10, ["mock-model"], STATUS_CHECK, max_tokens=20),
    Load("demo-chain", 20, ["chat-demo"], QUESTIONS, max_tokens=200),
    Load("demo-ollama", 20, ["chat-demo"], QUESTIONS, max_tokens=200),
    Load("team-alpha", 30, ["chat-general", "chat-large"], QUESTIONS),
    Load("team-beta", 30, ["llama3.2"], POLICY_QUESTIONS, max_tokens=200, blocked_every=4),
]


class Run:
    def __init__(self, client: httpx.AsyncClient, duration_seconds: float) -> None:
        self.client = client
        self.duration_seconds = duration_seconds
        self.started_at = time.monotonic()
        self.totals: collections.Counter = collections.Counter()
        self.window: collections.Counter = collections.Counter()
        self.pending: set[asyncio.Task] = set()

    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    def running(self) -> bool:
        return self.elapsed() < self.duration_seconds

    def log(self, message: str) -> None:
        seconds = int(self.elapsed())
        print(f"[{seconds // 60:02d}:{seconds % 60:02d}] {message}", flush=True)

    async def chat(self, label: str, key: str, model: str, prompt: str, max_tokens: int) -> None:
        try:
            response = await self.client.post(
                "/v1/chat",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            served_by = response.json().get("provider", "-") if response.status_code == 200 else "-"
            outcome = (label, response.status_code, served_by)
        except httpx.HTTPError as exc:  # a failed request is still a data point, not a crash
            outcome = (label, "error", type(exc).__name__)
        self.totals[outcome] += 1
        self.window[outcome] += 1

    def fire(self, coro) -> None:
        """Start a request without waiting, so a slow retry does not stretch the schedule."""
        task = asyncio.create_task(coro)
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)

    async def steady(self, load: Load, team: dict) -> None:
        models = itertools.cycle([m for m in load.models if m in team["allowed_models"]])
        label_base = load.team_id
        for n in itertools.count(1):
            if not self.running():
                return
            model = next(models)
            if load.blocked_every and n % load.blocked_every == 0:
                prompt, label = random.choice(BLOCKED_PROMPTS), f"{label_base} blocked"
            else:
                prompt, label = random.choice(load.prompts), f"{label_base} {model}"
            self.fire(self.chat(label, team["api_key"], model, prompt, load.max_tokens))
            await asyncio.sleep(load.every_seconds * random.uniform(0.8, 1.2))

    async def at(self, fraction: float, name: str, action) -> None:
        await asyncio.sleep(self.duration_seconds * fraction)
        self.log(f"event: {name}")
        await action()

    async def report_every_minute(self) -> None:
        while self.running():
            await asyncio.sleep(min(60, max(1, self.duration_seconds - self.elapsed())))
            summary = ", ".join(
                f"{label} {status}/{served_by} x{count}"
                for (label, status, served_by), count in sorted(self.window.items(), key=str)
            )
            self.log(summary or "no requests completed")
            self.window.clear()


async def main(base_url: str, minutes: float) -> int:
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        teams_response = await client.get("/demo/teams")
        if teams_response.status_code == 404:
            print(f"{base_url} is not running in demo mode (no /demo/teams).", file=sys.stderr)
            return 1
        teams_response.raise_for_status()
        teams = {team["team_id"]: team for team in teams_response.json()["teams"]}
        status = (await client.get("/demo/status")).json()
        breakable = [provider["name"] for provider in status["providers"]]

        run = Run(client, minutes * 60)
        loads = [
            load for load in LOADS
            if load.team_id in teams
            and any(model in teams[load.team_id]["allowed_models"] for model in load.models)
        ]

        async def outage(provider: str) -> None:
            body = (await client.post(f"/demo/outage/{provider}")).json()
            run.log(f"  {provider} outage started: {body.get('started')}, "
                    f"for {body.get('outage_seconds')} s")

        async def burst() -> None:
            key = teams["demo-public"]["api_key"]
            await asyncio.gather(*[
                run.chat("demo-public burst", key, "mock-model", f"burst {i}", 20)
                for i in range(30)
            ])

        async def oversized() -> None:
            # About 5,000 estimated tokens against a 4,000 tokens-per-minute limit: refused at
            # admission, before any provider is called.
            key = teams["demo-policy"]["api_key"]
            huge = "Summarise this: " + ("lorem ipsum dolor sit amet " * 750)
            await asyncio.gather(*[
                run.chat("demo-policy oversized", key, "chat-demo", huge, 50) for _ in range(2)
            ])

        events = []
        if "groq" in breakable:
            events.append(run.at(0.30, "simulated Groq outage", lambda: outage("groq")))
        if "ollama" in breakable:
            events.append(run.at(0.50, "simulated Ollama outage", lambda: outage("ollama")))
        if "demo-public" in teams:
            events.append(run.at(0.65, "burst of 30 requests on demo-public", burst))
        if "demo-policy" in teams:
            events.append(run.at(0.80, "two oversized prompts on demo-policy", oversized))

        run.log(f"{base_url}: {len(loads)} streams across "
                f"{len({load.team_id for load in loads})} teams for {minutes:g} min; "
                f"outages on {', '.join(breakable) or 'nothing'}")
        await asyncio.gather(
            *(run.steady(load, teams[load.team_id]) for load in loads),
            *events,
            run.report_every_minute(),
        )
        if run.pending:
            await asyncio.wait(run.pending, timeout=120)

    run.log("done. totals:")
    for (label, status_code, served_by), count in sorted(run.totals.items(), key=str):
        print(f"  {label:<26} {status_code!s:<6} {served_by:<8} {count}", flush=True)
    by_provider = collections.Counter()
    for (_label, _status, served_by), count in run.totals.items():
        by_provider[served_by] += count
    print("  answered by: " + ", ".join(
        f"{name} {count}" for name, count in sorted(by_provider.items()) if name != "-"
    ), flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--minutes", type=float, default=10)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.base_url.rstrip("/"), args.minutes)))

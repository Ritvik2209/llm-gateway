"""FastAPI application entrypoint for the LLM API gateway."""

import asyncio
import contextlib
import time
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.auth import verify_api_key
from app.budget import add_spend, calculate_cost, check_budget
from app.circuit_breaker import CircuitBreaker
from app.config import load_teams_config
from app.health import HealthMonitor
from app.metrics import (
    CIRCUIT_BREAKER_STATE as gateway_circuit_breaker_state,
    ERRORS_TOTAL,
    FALLBACK_TRIGGERED_TOTAL,
    REQUEST_DURATION_SECONDS,
    REQUESTS_TOTAL,
    TOKENS_TOTAL,
)
from app.models.schemas import ChatMessage, UnifiedChatRequest, UnifiedChatResponse
from app.providers.base import LLMProvider
from app.providers.groq_provider import GroqProvider
from app.providers.mock_provider import MockProvider
from app.providers.ollama_provider import OllamaProvider
from app.rate_limiter import check_rate_limit, get_redis_client
from app.retry import call_with_retry

ollama_provider = OllamaProvider()
mock_provider = MockProvider(should_fail=False)
groq_provider = GroqProvider()
providers: dict[str, LLMProvider] = {
    ollama_provider.provider_name: ollama_provider,
    mock_provider.provider_name: mock_provider,
    groq_provider.provider_name: groq_provider,
}
health_monitor = HealthMonitor()
circuit_breaker = CircuitBreaker()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    health_monitor.register_provider(ollama_provider, health_check_model="llama3.2")
    health_monitor.register_provider(mock_provider, health_check_model="mock-model")
    health_monitor.register_provider(groq_provider, health_check_model="openai/gpt-oss-20b")
    for provider_name in providers:
        gateway_circuit_breaker_state.labels(provider_name=provider_name).set(0)
    app.state.health_monitor = health_monitor
    app.state.providers = providers
    app.state.circuit_breaker = circuit_breaker
    app.state.health_check_task = asyncio.create_task(
        health_monitor.start_background_checks(interval_seconds=10)
    )
    try:
        yield
    finally:
        health_check_task = getattr(app.state, "health_check_task", None)
        if health_check_task is not None:
            health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await health_check_task


app = FastAPI(title="LLM API Gateway", lifespan=lifespan)
app.state.teams_config = load_teams_config()
app.state.redis_client = get_redis_client()
app.state.providers = providers
app.state.circuit_breaker = circuit_breaker


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a basic service health status."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/health")
async def admin_health() -> dict[str, dict[str, object]]:
    """Return current provider health statuses."""
    response: dict[str, dict[str, object]] = {}
    for provider_name in health_monitor.providers:
        provider_health = health_monitor.provider_health.get(provider_name)
        response[provider_name] = {
            "status": health_monitor.get_status(provider_name),
            "circuit_breaker_state": circuit_breaker.get_state(provider_name),
            "last_check_time": (
                provider_health.last_check_time.isoformat()
                if provider_health and provider_health.last_check_time
                else None
            ),
            "consecutive_failures": (
                provider_health.consecutive_failures if provider_health else 0
            ),
            "recent_latencies": (
                provider_health.recent_latencies if provider_health else []
            ),
            "error_rate": provider_health.error_rate if provider_health else 0.0,
        }
    return response


@app.post("/admin/mock/toggle-failure")
async def toggle_mock_failure() -> dict[str, bool]:
    """Toggle mock provider failure mode for local health monitor testing."""
    mock_provider.should_fail = not mock_provider.should_fail
    return {"should_fail": mock_provider.should_fail}


def enrich_request_with_team_system_prompt(
    request: UnifiedChatRequest,
    team_config: dict[str, Any],
) -> UnifiedChatRequest:
    """Prepend the team's system prompt when the caller did not provide one."""
    system_prompt = team_config.get("system_prompt")
    if not system_prompt:
        return request

    has_system_message = any(message.role == "system" for message in request.messages)
    if has_system_message:
        return request

    request.messages = [
        ChatMessage(role="system", content=system_prompt),
        *request.messages,
    ]
    return request


def select_provider(
    team_config: dict[str, Any],
    health_monitor: HealthMonitor,
    providers: dict[str, LLMProvider],
    circuit_breaker: CircuitBreaker | None = None,
) -> LLMProvider:
    """Select the best available provider from the team's priority order."""
    candidates = get_provider_candidates(
        team_config,
        health_monitor,
        providers,
        circuit_breaker,
    )
    if candidates:
        return candidates[0][1]

    provider_priority = team_config.get(
        "provider_priority",
        team_config.get("allowed_providers", []),
    )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "No healthy providers available. Attempted providers: "
            f"{', '.join(provider_priority) if provider_priority else 'none'}."
        ),
    )


def get_provider_candidates(
    team_config: dict[str, Any],
    health_monitor: HealthMonitor,
    providers: dict[str, LLMProvider],
    circuit_breaker: CircuitBreaker | None = None,
) -> list[tuple[str, LLMProvider]]:
    """Return provider candidates ordered by circuit state and team priority."""
    provider_priority = team_config.get(
        "provider_priority",
        team_config.get("allowed_providers", []),
    )
    attempted_providers = [
        provider_name for provider_name in provider_priority if provider_name in providers
    ]
    candidates: list[tuple[str, LLMProvider]] = []

    for provider_name in attempted_providers:
        if circuit_breaker and not circuit_breaker.can_attempt(provider_name):
            continue
        candidates.append((provider_name, providers[provider_name]))

    return candidates


def get_first_priority_provider(team_config: dict[str, Any]) -> str | None:
    """Return the team's first configured provider preference."""
    provider_priority = team_config.get(
        "provider_priority",
        team_config.get("allowed_providers", []),
    )
    return provider_priority[0] if provider_priority else None


async def call_chat_with_fallback(
    team_config: dict[str, Any],
    request: UnifiedChatRequest,
    health_monitor: HealthMonitor,
    circuit_breaker: CircuitBreaker,
    providers: dict[str, LLMProvider],
) -> UnifiedChatResponse:
    """Call providers in fallback order, recording circuit breaker outcomes."""
    candidates = get_provider_candidates(
        team_config,
        health_monitor,
        providers,
        circuit_breaker,
    )
    last_exception: Exception | None = None
    team_id = team_config["team_id"]
    first_priority_provider = get_first_priority_provider(team_config)

    for provider_name, provider in candidates:
        started_at = time.perf_counter()
        try:
            provider_response = await call_with_retry(provider, request)
        except Exception as exc:
            REQUEST_DURATION_SECONDS.labels(
                team_id=team_id,
                provider=provider_name,
            ).observe(time.perf_counter() - started_at)
            ERRORS_TOTAL.labels(
                team_id=team_id,
                provider=provider_name,
                error_type="provider_error",
            ).inc()
            last_exception = exc
            circuit_breaker.record_failure(provider_name)
            continue

        REQUEST_DURATION_SECONDS.labels(
            team_id=team_id,
            provider=provider_name,
        ).observe(time.perf_counter() - started_at)
        circuit_breaker.record_success(provider_name)
        if first_priority_provider and provider_name != first_priority_provider:
            FALLBACK_TRIGGERED_TOTAL.labels(
                team_id=team_id,
                from_provider=first_priority_provider,
                to_provider=provider_name,
            ).inc()
        return provider_response

    provider_priority = team_config.get(
        "provider_priority",
        team_config.get("allowed_providers", []),
    )
    detail = (
        "All providers failed or are unavailable. Attempted providers: "
        f"{', '.join(provider_priority) if provider_priority else 'none'}."
    )
    if last_exception is not None:
        detail = f"{detail} Last error: {last_exception}"

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=detail,
    )


@app.post("/v1/chat", response_model=None)
async def chat(
    request: UnifiedChatRequest,
    response: Response,
    team_config: dict[str, Any] = Depends(verify_api_key),
) -> UnifiedChatResponse | StreamingResponse:
    """Authenticate a team and route a unified chat request."""
    team_id = team_config["team_id"]
    request_provider = "none"
    try:
        if request.model not in team_config["allowed_models"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f'Model "{request.model}" is not allowed for this team.',
            )

        allowed, _remaining_quota = await check_rate_limit(
            team_id=team_id,
            requests_per_minute=team_config.get("requests_per_minute", 60),
            redis_client=app.state.redis_client,
        )
        if not allowed:
            ERRORS_TOTAL.labels(
                team_id=team_id,
                provider=request_provider,
                error_type="rate_limited",
            ).inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please retry later.",
                headers={"Retry-After": "60"},
            )

        budget_allowed, _current_spend, is_budget_warning = await check_budget(
            team_id=team_id,
            monthly_budget_usd=team_config.get("monthly_budget_usd", 0.0),
            redis_client=app.state.redis_client,
        )
        if not budget_allowed:
            ERRORS_TOTAL.labels(
                team_id=team_id,
                provider=request_provider,
                error_type="budget_exceeded",
            ).inc()
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail="Monthly budget cap has been reached for this team.",
            )

        if is_budget_warning:
            response.headers["X-Budget-Warning"] = "true"

        request = enrich_request_with_team_system_prompt(request, team_config)
        runtime_providers = {
            **app.state.providers,
            OllamaProvider.provider_name: ollama_provider,
            MockProvider.provider_name: mock_provider,
            GroqProvider.provider_name: groq_provider,
        }
        selected_candidates = get_provider_candidates(
            team_config,
            health_monitor,
            runtime_providers,
            app.state.circuit_breaker,
        )
        if selected_candidates:
            selected_provider_name, selected_provider = selected_candidates[0]
        else:
            selected_provider = select_provider(
                team_config,
                health_monitor,
                runtime_providers,
                app.state.circuit_breaker,
            )
            selected_provider_name = next(
                provider_name
                for provider_name, provider in runtime_providers.items()
                if provider is selected_provider
            )
        request_provider = selected_provider_name

        if request.stream is True:
            first_priority_provider = get_first_priority_provider(team_config)
            if first_priority_provider and selected_provider_name != first_priority_provider:
                FALLBACK_TRIGGERED_TOTAL.labels(
                    team_id=team_id,
                    from_provider=first_priority_provider,
                    to_provider=selected_provider_name,
                ).inc()

            async def stream_chunks() -> AsyncGenerator[str, None]:
                chunks: list[str] = []
                started_at = time.perf_counter()
                try:
                    async for chunk in selected_provider.chat_stream(request):
                        chunks.append(chunk)
                        yield chunk
                except Exception:
                    REQUEST_DURATION_SECONDS.labels(
                        team_id=team_id,
                        provider=selected_provider_name,
                    ).observe(time.perf_counter() - started_at)
                    ERRORS_TOTAL.labels(
                        team_id=team_id,
                        provider=selected_provider_name,
                        error_type="provider_error",
                    ).inc()
                    REQUESTS_TOTAL.labels(
                        team_id=team_id,
                        model=request.model,
                        provider=selected_provider_name,
                        status="error",
                    ).inc()
                    raise

                REQUEST_DURATION_SECONDS.labels(
                    team_id=team_id,
                    provider=selected_provider_name,
                ).observe(time.perf_counter() - started_at)
                app.state.last_streamed_chat_response_text = "".join(chunks)
                stream_usage = getattr(
                    selected_provider,
                    "last_stream_usage",
                    {"input_tokens": 0, "output_tokens": 0},
                )
                TOKENS_TOTAL.labels(
                    team_id=team_id,
                    provider=selected_provider_name,
                    token_type="input",
                ).inc(stream_usage["input_tokens"])
                TOKENS_TOTAL.labels(
                    team_id=team_id,
                    provider=selected_provider_name,
                    token_type="output",
                ).inc(stream_usage["output_tokens"])
                cost = calculate_cost(
                    model=request.model,
                    input_tokens=stream_usage["input_tokens"],
                    output_tokens=stream_usage["output_tokens"],
                )
                await add_spend(team_id, cost, app.state.redis_client)
                REQUESTS_TOTAL.labels(
                    team_id=team_id,
                    model=request.model,
                    provider=selected_provider_name,
                    status="success",
                ).inc()

            headers = {}
            if is_budget_warning:
                headers["X-Budget-Warning"] = "true"
            return StreamingResponse(
                stream_chunks(),
                media_type="text/plain",
                headers=headers,
            )

        provider_response = await call_chat_with_fallback(
            team_config,
            request,
            health_monitor,
            app.state.circuit_breaker,
            runtime_providers,
        )
        request_provider = provider_response.provider
        TOKENS_TOTAL.labels(
            team_id=team_id,
            provider=provider_response.provider,
            token_type="input",
        ).inc(provider_response.input_tokens)
        TOKENS_TOTAL.labels(
            team_id=team_id,
            provider=provider_response.provider,
            token_type="output",
        ).inc(provider_response.output_tokens)
        cost = calculate_cost(
            model=provider_response.model,
            input_tokens=provider_response.input_tokens,
            output_tokens=provider_response.output_tokens,
        )
        await add_spend(team_id, cost, app.state.redis_client)
        REQUESTS_TOTAL.labels(
            team_id=team_id,
            model=request.model,
            provider=provider_response.provider,
            status="success",
        ).inc()
        return provider_response
    except HTTPException:
        REQUESTS_TOTAL.labels(
            team_id=team_id,
            model=request.model,
            provider=request_provider,
            status="error",
        ).inc()
        raise
    except Exception:
        ERRORS_TOTAL.labels(
            team_id=team_id,
            provider=request_provider,
            error_type="provider_error",
        ).inc()
        REQUESTS_TOTAL.labels(
            team_id=team_id,
            model=request.model,
            provider=request_provider,
            status="error",
        ).inc()
        raise

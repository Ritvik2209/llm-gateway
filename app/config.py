"""Application configuration loading for the LLM API gateway."""

import os
from pathlib import Path
from typing import Any

import yaml

from app.models.catalog import ModelCatalog
from dotenv import load_dotenv


load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# How often to probe providers that need probing. The spec calls for 30 seconds; the
# request path had it hardcoded to 10, which is three times as many probes as intended
# and, on a metered free tier, enough for the monitor alone to exhaust a daily quota.
HEALTH_CHECK_INTERVAL_SECONDS = int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "30"))

TEAMS_CONFIG_PATH = os.getenv("TEAMS_CONFIG_PATH", "config/teams.yaml")
MODEL_CATALOG_PATH = os.getenv("MODEL_CATALOG_PATH", "config/models.yaml")

# How often to check whether the config files changed. 0 disables automatic reloading,
# leaving the explicit admin endpoint as the only way to apply an edit.
CONFIG_RELOAD_INTERVAL_SECONDS = int(os.getenv("CONFIG_RELOAD_INTERVAL_SECONDS", "5"))

MODEL_PRICING = {
    "llama3.2": {
        "input_price_per_1k": 0.0,
        "output_price_per_1k": 0.0,
    },
    "openai/gpt-oss-20b": {
        "input_price_per_1k": 0.000075,
        "output_price_per_1k": 0.0003,
    },
    "mock-model": {
        "input_price_per_1k": 0.0,
        "output_price_per_1k": 0.0,
    },
}


def load_model_catalog(path: str = "config/models.yaml") -> ModelCatalog:
    """Load which physical models each provider serves, and the logical tiers.

    Routing uses this to skip providers that cannot serve the requested model at all,
    instead of discovering that fact by failing. A provider missing from the catalog serves
    nothing, keeping the check fail-closed like the provider allowlist.

    Tiers are validated on load: a tier that names an unknown provider, or maps a provider
    to a model that provider does not serve, is a configuration error that would surface at
    request time as an unexplained 404 from the provider. Rejecting it here means the
    validated-reload path refuses it and the previous catalog keeps serving.
    """
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as catalog_file:
        raw_catalog = yaml.safe_load(catalog_file) or {}

    providers = {
        provider_name: set(models or [])
        for provider_name, models in (raw_catalog.get("providers") or {}).items()
    }

    tiers: dict[str, dict[str, str]] = {}
    for tier_name, mapping in (raw_catalog.get("tiers") or {}).items():
        if tier_name in providers:
            raise ValueError(
                f'Tier "{tier_name}" collides with a provider name.'
            )
        resolved: dict[str, str] = {}
        for provider_name, physical_model in (mapping or {}).items():
            if provider_name not in providers:
                raise ValueError(
                    f'Tier "{tier_name}" names unknown provider "{provider_name}".'
                )
            if physical_model not in providers[provider_name]:
                raise ValueError(
                    f'Tier "{tier_name}" maps "{provider_name}" to "{physical_model}", '
                    f"which that provider does not serve."
                )
            resolved[provider_name] = physical_model
        tiers[tier_name] = resolved

    return ModelCatalog(providers=providers, tiers=tiers)


def load_teams_config(path: str = "config/teams.yaml") -> dict[str, dict[str, Any]]:
    """Load team authentication and authorization config keyed by API key."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file) or {}

    teams_by_api_key = {}
    for team in raw_config.get("teams", []):
        teams_by_api_key[team["api_key"]] = {
            "team_id": team["team_id"],
            "allowed_models": team.get("allowed_models", []),
            "allowed_providers": team.get("allowed_providers", []),
            "provider_priority": team.get(
                "provider_priority",
                team.get("allowed_providers", []),
            ),
            "system_prompt": team.get("system_prompt"),
            "requests_per_minute": team.get("requests_per_minute", 60),
            # 0 disables token limiting, so teams configured before this limit
            # existed keep their previous behaviour.
            "tokens_per_minute": team.get("tokens_per_minute", 0),
            "monthly_budget_usd": team.get("monthly_budget_usd", 0.0),
            "is_admin": team.get("is_admin", False),
        }

    return teams_by_api_key

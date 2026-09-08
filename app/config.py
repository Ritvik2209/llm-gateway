"""Application configuration loading for the LLM API gateway."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
            "monthly_budget_usd": team.get("monthly_budget_usd", 0.0),
        }

    return teams_by_api_key

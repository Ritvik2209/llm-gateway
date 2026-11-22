"""The dashboard link on every alert must open this environment's Grafana, on a real dashboard.

A Slack alert is only as useful as the link in it. These links were once hard-coded to
localhost:3000, so an alert from the public server opened nothing. They now read the Grafana
address from an external label that Compose sets per environment, and this keeps it that way.
"""

import json
import re
from pathlib import Path

import yaml


MONITORING = Path(__file__).resolve().parent.parent / "monitoring"
COMPOSE_FILE = MONITORING.parent / "docker-compose.yml"
LINK = re.compile(r"^\{\{ \$externalLabels\.grafana_url \}\}/d/(?P<uid>[^/]+)/[\w-]+$")


def load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def alert_rules():
    groups = load_yaml(MONITORING / "alerts.yml")["groups"]
    return [rule for group in groups for rule in group["rules"] if "alert" in rule]


def dashboard_uids():
    return {
        json.loads(path.read_text(encoding="utf-8"))["uid"]
        for path in (MONITORING / "grafana" / "dashboards").glob("*.json")
    }


def test_every_alert_links_to_an_existing_dashboard_on_this_environments_grafana():
    rules = alert_rules()
    uids = dashboard_uids()

    assert rules
    for rule in rules:
        link = rule["annotations"].get("dashboard", "")
        match = LINK.match(link)
        assert match, f"{rule['alert']} links to {link!r}"
        assert match["uid"] in uids, f"{rule['alert']} links to a dashboard that does not exist"


def test_the_grafana_address_comes_from_the_environment():
    """The label the links read must exist, and the local stack must give it a value."""
    prometheus = load_yaml(MONITORING / "prometheus.yml")
    compose = load_yaml(COMPOSE_FILE)

    assert prometheus["global"]["external_labels"]["grafana_url"] == "${GRAFANA_URL}"
    assert "GRAFANA_URL=http://localhost:3000" in compose["services"]["prometheus"]["environment"]

"""Safe in-place edits to the team configuration file.

The admin API adjusts limits by editing ``config/teams.yaml`` rather than holding
overrides in a second store. That keeps the file the single source of truth the
architecture declares, so there is no way for the running policy to silently diverge from
the policy on disk — which is the failure mode a dynamic override layer introduces.

Two details make writing to live config safe:

* **Comments survive.** ``ruamel.yaml`` round-trips the document, where ``pyyaml`` would
  reserialise it and discard every comment. The comments in ``teams.yaml`` explain why the
  mock provider is test-only and why the load-test team has no token limit; losing them to
  a limit change would be a real regression.
* **The write is atomic.** The new document is written to a temporary file in the same
  directory, parsed back to prove it loads, and only then moved over the original. A crash
  or a full disk mid-write cannot leave a truncated config that the gateway would refuse to
  load on its next start.
"""

import os
import tempfile
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


# Only these may be changed at runtime. Everything else — and in particular ``is_admin``,
# ``api_key`` and the model and provider allowlists — is deliberately excluded: an admin
# endpoint that can grant admin rights or widen an authorization boundary is a privilege
# escalation path, not a convenience.
EDITABLE_FIELDS = ("requests_per_minute", "tokens_per_minute", "monthly_budget_usd")


class TeamNotFoundError(LookupError):
    """No team in the configuration file has the requested id."""


class ConfigWriteError(RuntimeError):
    """The edited configuration could not be written, so nothing was changed."""


def _represent_none(representer, _data):
    """Emit ``null`` rather than an empty value.

    ruamel writes ``None`` as nothing at all, so ``system_prompt: null`` would silently
    become ``system_prompt:``. Both parse to None, but the edit would appear in the diff of
    every team that has one — noise that hides the line the admin actually changed.
    """
    return representer.represent_scalar("tag:yaml.org,2002:null", "null")


def _yaml() -> YAML:
    parser = YAML()
    parser.preserve_quotes = True
    # Keep the existing two-space list indentation rather than reflowing the document.
    parser.indent(mapping=2, sequence=4, offset=2)
    # Without this, ruamel folds any line past ~80 columns, so a long system prompt gets
    # wrapped into a multi-line scalar on an unrelated edit. A writable config file is only
    # worth having if the diff shows one change; anything else makes review useless.
    parser.width = 4096
    parser.representer.add_representer(type(None), _represent_none)
    return parser


def read_team_limits(path: str) -> dict[str, dict[str, Any]]:
    """Return the editable limits for every team, keyed by team id."""
    parser = _yaml()
    with Path(path).open("r", encoding="utf-8") as config_file:
        document = parser.load(config_file) or {}

    limits = {}
    for team in document.get("teams") or []:
        limits[str(team.get("team_id"))] = {
            field: team.get(field) for field in EDITABLE_FIELDS
        }
    return limits


def update_team_limits(
    team_id: str,
    changes: dict[str, Any],
    path: str,
) -> dict[str, dict[str, Any]]:
    """Apply limit changes to one team and write the file back atomically.

    Returns ``{"before": ..., "after": ...}`` for the fields that actually changed, which
    is what the audit trail records. Raises before touching the file if the team does not
    exist or a field is not editable.
    """
    unknown_fields = set(changes) - set(EDITABLE_FIELDS)
    if unknown_fields:
        raise ConfigWriteError(
            f"Fields not editable at runtime: {', '.join(sorted(unknown_fields))}."
        )

    config_path = Path(path)
    parser = _yaml()
    with config_path.open("r", encoding="utf-8") as config_file:
        document = parser.load(config_file) or {}

    teams = document.get("teams") or []
    target = next(
        (team for team in teams if str(team.get("team_id")) == team_id),
        None,
    )
    if target is None:
        raise TeamNotFoundError(f'No team with id "{team_id}" in {path}.')

    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    for field, value in changes.items():
        current = target.get(field)
        if current == value:
            continue
        before[field] = current
        after[field] = value
        target[field] = value

    if not after:
        # Nothing to do. Returning early avoids rewriting the file — and so avoids an
        # audit entry and a config reload — for a request that changes nothing.
        return {"before": {}, "after": {}}

    _write_atomically(document, config_path, parser)
    return {"before": before, "after": after}


def _write_atomically(document: Any, config_path: Path, parser: YAML) -> None:
    """Write the document beside the target, verify it parses, then move it into place."""
    directory = config_path.parent
    handle = None
    temp_path = None
    try:
        handle, temp_name = tempfile.mkstemp(
            dir=directory,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
        )
        temp_path = Path(temp_name)
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as temp_file:
            handle = None
            parser.dump(document, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        # Prove the file we are about to install actually loads, so a bug here cannot
        # hand the gateway a config it will refuse on its next start.
        with temp_path.open("r", encoding="utf-8") as written:
            _yaml().load(written)

        os.replace(temp_path, config_path)
        temp_path = None
    except Exception as exc:
        raise ConfigWriteError(f"Configuration was not written: {exc}") from exc
    finally:
        if handle is not None:
            os.close(handle)
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

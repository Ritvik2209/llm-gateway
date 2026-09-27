"""Audit trail for administrative changes.

Every change records who made it, when, what it targeted, and the values on both sides.
It is written to two places on purpose:

* **stdout**, which the container runtime retains and ships wherever logs go. This is the
  copy that survives losing Redis.
* **a capped Redis list**, which is what the admin API can read back without a log
  aggregator.

The Redis copy is best-effort: the change has already been applied to the config file by
the time it is recorded, so failing the request because the audit write failed would
report a failure that did not happen. A dropped Redis entry is logged and the stdout copy
still exists.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)

AUDIT_LOG_KEY = "audit:log"

# Bounded so the list cannot grow without limit in a store that has no eviction policy
# configured. Long-term retention is the job of whatever collects stdout.
AUDIT_LOG_MAX_ENTRIES = 1000


async def record_change(
    redis_client,
    actor: str,
    action: str,
    target: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one administrative change and return the entry."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "target": target,
        "before": before or {},
        "after": after or {},
    }

    logger.info("audit %s", json.dumps(entry, sort_keys=True))

    try:
        await redis_client.lpush(AUDIT_LOG_KEY, json.dumps(entry, sort_keys=True))
        await redis_client.ltrim(AUDIT_LOG_KEY, 0, AUDIT_LOG_MAX_ENTRIES - 1)
    except Exception as exc:
        logger.warning("Audit entry not persisted to Redis: %s", exc)

    return entry


async def read_audit_log(redis_client, limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent audit entries, newest first."""
    try:
        raw_entries = await redis_client.lrange(AUDIT_LOG_KEY, 0, max(0, limit - 1))
    except Exception as exc:
        logger.warning("Audit log could not be read: %s", exc)
        return []

    entries = []
    for raw_entry in raw_entries:
        if isinstance(raw_entry, bytes):
            raw_entry = raw_entry.decode()
        try:
            entries.append(json.loads(raw_entry))
        except json.JSONDecodeError:
            # A malformed entry should not hide the rest of the trail.
            logger.warning("Skipping unreadable audit entry")
    return entries

"""Overriding app.state entries in tests.

From pytest 9.2, monkeypatch.setattr cannot be used on Starlette's State. To restore an
instance attribute it now records the original from the instance __dict__, but State keeps
its entries in a dict of its own. So the original is recorded as absent, undo() deletes the
real entry, and every later test finds teams_config or redis_client missing. This broke 70
tests in CI while passing locally on pytest 9.1. setitem on State's backing dict restores
correctly on every pytest version.
"""

from app import main as main_module


def set_app_state(monkeypatch, name, value):
    """Override main_module.app.state.<name> for the rest of the test, then restore it."""
    state_entries = main_module.app.state._state
    if name not in state_entries:
        # Matches monkeypatch.setattr's default: a typo should fail, not add a new entry.
        raise AttributeError(f"app.state has no attribute {name!r}")
    monkeypatch.setitem(state_entries, name, value)

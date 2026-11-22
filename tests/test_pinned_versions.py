"""Everything the gateway installs or runs is pinned to an exact version.

Unpinned, each build takes whatever is newest. That is how pytest 9.2 broke 70 tests in CI
while every local run passed, and an image would do the same to the server, silently, on its
next rebuild. Upgrades should be a deliberate edit, not a side effect of building.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILES = sorted(ROOT.glob("docker-compose*.yml"))
# An exact release: a tag that starts with a version number with at least a minor part,
# optionally followed by a variant such as -slim-trixie. Not latest, and not a bare major.
EXACT_TAG = re.compile(r"^v?\d+\.\d+(\.\d+)?(-[\w.-]+)?$")


def image_references() -> list[tuple[str, str]]:
    references = []
    for path in COMPOSE_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*image:\s*(\S+)", line)
            if match:
                references.append((path.name, match.group(1)))
    for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^FROM\s+(\S+)", line)
        if match:
            references.append(("Dockerfile", match.group(1)))
    return references


def test_every_image_is_pinned_to_an_exact_release():
    references = image_references()

    assert len(references) >= 6
    for source, image in references:
        name, _, tag = image.rpartition(":")
        assert name and EXACT_TAG.match(tag), f"{source}: {image} is not pinned to a release"


def test_every_python_requirement_is_pinned():
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lines
    for line in lines:
        assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[\w.]+", line), f"{line} is not pinned"

"""Guard that everything the shipped code imports is actually in the image.

The Dockerfile copies a hand written list of paths. Nothing checks that list
against what the code imports, so adding a module level import of a package
that is not copied produces a container that starts cleanly and then fails at
the point of use. That happened once: `tools/web_search.py` gained a module
level `from observability.audit import audited` while `observability` was not
copied, and the web search node caught the ImportError and silently answered
without sources.

This test parses the COPY lines and the first party imports and asserts the
two agree. It reads files only, so it is fast and needs no Docker.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "Dockerfile"
DOCKERIGNORE = REPO / ".dockerignore"

# A COPY line, ignoring --from and other flags.
_COPY = re.compile(r"^\s*COPY\s+(?!--)(?P<src>[^\s]+)\s+(?P<dst>[^\s]+)\s*$", re.M)

# Paths the build needs but the application never imports.
_NOT_IMPORTED = {"requirements.txt"}


def _first_party_names() -> set[str]:
    """Top level importable names that live in this repository."""
    names = set()
    for entry in REPO.iterdir():
        if entry.name.startswith((".", "_")):
            continue
        if entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
    return names


def copied_names() -> set[str]:
    """Top level names the Dockerfile copies into the image."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    names = set()
    for match in _COPY.finditer(text):
        src = match.group("src")
        if src in _NOT_IMPORTED:
            continue
        names.add(Path(src).stem if src.endswith(".py") else Path(src).name)
    return names


def _module_level_imports(path: Path) -> set[str]:
    """Top level names imported at module scope in one file.

    Only module scope counts. An import inside a function tolerates the
    package being absent, which several modules rely on deliberately.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - not expected
        return set()
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _shipped_files() -> list[Path]:
    """Every Python file that ends up in the image."""
    files: list[Path] = []
    for name in sorted(copied_names()):
        target = REPO / name
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        elif target.with_suffix(".py").is_file():
            files.append(target.with_suffix(".py"))
    return [f for f in files if "__pycache__" not in f.parts]


def ignored_names() -> set[str]:
    """Top level names .dockerignore keeps out of the build context."""
    if not DOCKERIGNORE.exists():
        return set()
    names = set()
    for raw in DOCKERIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!")):
            continue
        names.add(Path(line.rstrip("/")).name)
    return names


def test_no_copied_path_is_excluded_from_the_build_context():
    """A COPY of an ignored path does not warn, it fails the build.

    Both files have to agree, which is a trap worth a test of its own: adding
    the COPY line alone leaves the build broken.
    """
    overlap = sorted(copied_names() & ignored_names())
    assert not overlap, (
        f".dockerignore excludes {overlap}, which the Dockerfile also tries to "
        f"COPY. The build fails on this, so remove the exclusion or the COPY."
    )


def test_dockerfile_has_copy_lines():
    assert copied_names(), "no COPY lines parsed, the regex or Dockerfile changed"


def test_shipped_files_were_found():
    assert _shipped_files(), "no shipped Python files found to inspect"


@pytest.mark.parametrize("source", _shipped_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_module_level_first_party_imports_are_copied_into_the_image(source: Path):
    first_party = _first_party_names()
    copied = copied_names()
    imported = _module_level_imports(source) & first_party
    missing = sorted(imported - copied)
    assert not missing, (
        f"{source.relative_to(REPO)} imports {missing} at module scope, but the "
        f"Dockerfile does not COPY it. Either add a COPY line or move the import "
        f"inside a function so the package may be absent."
    )

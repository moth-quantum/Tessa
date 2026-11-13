"""Locate the input data and stored results of the paper's repository.

The library does not ship data inside the wheel. The inputs (``data/``) and the
stored hardware and simulation outputs (``results/``) live in the repository
checkout next to ``src/``. These helpers find that checkout by walking up from
this file; set ``$TESSA_REPO`` to point at a checkout explicitly (e.g. when the
package is installed elsewhere).

``data_path("qcat28.png")`` accepts a bare file name and searches the whole
``data/`` tree for it (``data/images/qcat/qcat28.png``), or a path relative to
``data/``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["data_dir", "data_path", "repo_root", "results_dir"]

_HERE = Path(__file__).resolve().parent


def _is_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "tessa").is_dir()


def repo_root() -> Path:
    """The repository checkout: ``$TESSA_REPO``, or the first parent of this file
    that holds ``pyproject.toml`` and ``src/tessa``."""
    env = os.environ.get("TESSA_REPO")
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"$TESSA_REPO={env} is not a directory")
        return root
    for parent in _HERE.parents:
        if _is_root(parent):
            return parent
    raise FileNotFoundError(
        "tessa is not running from a checkout of the paper repository, so data/ and "
        "results/ cannot be found; clone the repository and install it with "
        "`pip install -e .`, or set $TESSA_REPO to the checkout"
    )


def data_dir() -> Path:
    """The repository's ``data/`` directory (source data, before encoding)."""
    out = repo_root() / "data"
    if not out.is_dir():
        raise FileNotFoundError(f"no data directory at {out}")
    return out


def results_dir() -> Path:
    """The repository's ``results/`` directory (stored outputs of the experiments)."""
    out = repo_root() / "results"
    if not out.is_dir():
        raise FileNotFoundError(f"no results directory at {out}")
    return out


def data_path(name: str | os.PathLike[str]) -> Path:
    """Absolute path of an input file under ``data/``.

    ``name`` is a path relative to ``data/`` (``"images/qcat/qcat.png"``) or a bare
    file name (``"qcat.png"``), which must then occur exactly once in the tree.
    """
    root = data_dir()
    direct = root / name
    if direct.is_file():
        return direct
    hits = sorted(p for p in root.rglob(Path(name).name) if p.is_file())
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(f"{name!s} not found under {root}")
    found = ", ".join(str(p.relative_to(root)) for p in hits)
    raise FileNotFoundError(f"{name!s} is ambiguous under {root}: {found}")

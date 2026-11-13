"""Fixed locations in the repository checkout.

``REPO`` is the checkout root, found by :func:`tessa.repo_root`: ``$TESSA_REPO`` if
set, otherwise the checkout that contains the installed source (the case for an
editable install, ``pip install -e .``). ``DATA`` holds the inputs, ``RESULTS`` the
stored outputs of the experiments and ``FIGURES`` the figures the notebooks write
(``FIGURES / "paper"`` holds reference copies of the published figures).
"""

from __future__ import annotations

from tessa.data import repo_root

__all__ = ["DATA", "FIGURES", "REPO", "RESULTS"]

REPO = repo_root()
DATA = REPO / "data"
RESULTS = REPO / "results"
FIGURES = REPO / "figures"
if not RESULTS.is_dir():
    raise FileNotFoundError(
        f"no results/ directory in {REPO}: tessa_paper needs a full checkout of the "
        "paper repository (or $TESSA_REPO pointing at one)"
    )

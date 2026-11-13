"""Execute the notebooks top to bottom (slow; deselected by ``-m "not slow"``)."""

from __future__ import annotations

import pytest

from tessa_paper.paths import REPO

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = ["reproduce_paper.ipynb", "quickstart.ipynb", "rerun_from_raw.ipynb"]


@pytest.mark.slow
@pytest.mark.notebook
@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_runs_without_errors(name):
    path = REPO / "notebooks" / name
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()  # raises CellExecutionError on the first failing cell
    errors = [
        out
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors

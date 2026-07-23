from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest


@pytest.mark.parametrize(
    "name,required",
    [
        (
            "hpo_smoke_test_colab.ipynb",
            ["Hardware", "Python dictionary", "CSV", "Proxy", "Successive halving", "Full", "Continuous", "Resume", "Pareto"],
        ),
        (
            "hyperparameter_search_colab.ipynb",
            ["Persistence", "Hardware profile", "Search-space source", "Search mode", "Calibration", "Pre-search estimate", "Search execution", "Resume", "Results", "Export"],
        ),
    ],
)
def test_notebook_json_syntax_sections_and_clean_outputs(name: str, required: list[str]):
    path = Path(__file__).resolve().parents[2] / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    text = "\n".join(cell.source for cell in notebook.cells)
    for section in required:
        assert section.lower() in text.lower()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == []
            assert cell.execution_count is None
    assert path.stat().st_size < 500_000

"""Notebook cells must carry the kind of content their type says they do.

Editing a notebook by cell index is how this broke: inserting two cells shifted
everything, and a later index-based rewrite dropped prose into a code cell. The
result raised `SyntaxError: invalid character '—'` on a GPU box after 45 minutes
of training, which is an expensive place to find out.
"""

import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parent.parent.glob("notebooks/*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_prose_in_code_cells(path):
    for i, cell in enumerate(json.loads(path.read_text())["cells"]):
        src = "".join(cell["source"]).lstrip()
        if cell["cell_type"] == "code" and src.startswith("##"):
            pytest.fail(f"{path.name} cell {i}: markdown heading inside a code cell")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_no_shell_commands_in_markdown_cells(path):
    for i, cell in enumerate(json.loads(path.read_text())["cells"]):
        src = "".join(cell["source"]).lstrip()
        if cell["cell_type"] == "markdown" and src.startswith("!"):
            pytest.fail(f"{path.name} cell {i}: shell command inside a markdown cell")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_without_magics_compile(path):
    """Cells free of IPython magics must be valid Python; the rest are skipped
    because `!` and `%` are not Python syntax."""
    for i, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(line.lstrip().startswith(("!", "%")) for line in src.splitlines()):
            continue
        try:
            compile(src, f"{path.name}:{i}", "exec")
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {i}: {exc}")

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .search_space import load_csv, normalize_space, preview_rows


def normalize_notebook_space(value: Any, *, source: str = "manual notebook input"):
    """Normalize an editable Python list/dictionary from a notebook cell."""
    return normalize_space(value, source_name=source)


def normalize_uploaded_csv(content: bytes | str, *, filename: str = "uploaded_space.csv"):
    temporary = Path("/tmp") / filename
    temporary.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return load_csv(temporary)


def preview_dataframe(specs):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for notebook preview tables") from exc
    return pd.DataFrame(preview_rows(specs))


def optional_widgets(default_space: dict[str, Any] | None = None):
    """Return lightweight widgets when ipywidgets is installed; otherwise return None."""
    try:
        import ipywidgets as widgets
    except ImportError:
        return None
    default_space = default_space or {}
    mode = widgets.Dropdown(options=["proxy", "successive_halving", "full"], value="successive_halving", description="Mode")
    trials = widgets.IntSlider(value=8, min=1, max=100, description="Trials")
    continuous = widgets.Checkbox(value=False, description="Continuous")
    return widgets.VBox([mode, trials, continuous])

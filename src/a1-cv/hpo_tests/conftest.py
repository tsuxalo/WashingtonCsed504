from __future__ import annotations

import sys
from pathlib import Path

import pytest

CV_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CV_DIR.parents[1]
for path in (CV_DIR, REPO_ROOT / "src" / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT

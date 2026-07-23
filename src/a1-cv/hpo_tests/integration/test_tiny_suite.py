from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("HPO_RUN_TRAINING_INTEGRATION") != "1",
    reason=(
        "set HPO_RUN_TRAINING_INTEGRATION=1 to run the in-process PyTorch "
        "training suite; run_tiny_suite.py is the default bounded check"
    ),
)
def test_standalone_tiny_integration_suite():
    # Import only when explicitly enabled. Importing the PyTorch integration
    # driver during normal test collection can keep host tracing plugins alive
    # after all lightweight tests have reported their results.
    test_dir = Path(__file__).resolve().parent
    if str(test_dir) not in sys.path:
        sys.path.insert(0, str(test_dir))
    from run_tiny_suite import run

    result = run()
    assert result["adapters"] == "passed"
    assert result["proxy"]["records"] >= 1
    assert result["successive_halving"]["records"] >= 1
    assert result["full"]["records"] >= 1
    assert result["continuous"]["records"] >= 1

"""Additions-only hardware-aware HPO framework for WashingtonCsed504/a1-cv."""

from .config import load_study_config
from .hardware import HardwareProfile, detect_hardware
from .search_space import load_space, normalize_space, preview_rows
from .study import HpoStudy

__all__ = [
    "HardwareProfile",
    "HpoStudy",
    "detect_hardware",
    "load_space",
    "load_study_config",
    "normalize_space",
    "preview_rows",
]

__version__ = "0.1.0"

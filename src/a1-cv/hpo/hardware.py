from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import psutil
import torch


@dataclass
class GpuProfile:
    index: int
    name: str
    total_vram_gb: float
    available_vram_gb: float | None
    compute_capability: tuple[int, int] | None


@dataclass
class HardwareProfile:
    operating_system: str
    notebook: bool
    colab: bool
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    cudnn_version: int | None
    gpu_count: int
    gpus: list[GpuProfile] = field(default_factory=list)
    fp16_supported: bool = False
    bf16_supported: bool = False
    bf16_native: bool = False
    tf32_supported: bool = False
    mps_available: bool = False
    physical_cpu_cores: int | None = None
    logical_cpu_threads: int = 1
    available_ram_gb: float = 0.0
    total_ram_gb: float = 0.0
    storage_free_gb: float | None = None
    torch_compile_available: bool = False
    multiprocessing_start_method: str | None = None
    nvidia_smi: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _native_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8


def _smi() -> dict[str, Any] | None:
    if not shutil.which("nvidia-smi"):
        return None
    query = "index,name,memory.total,memory.free,utilization.gpu,power.draw"
    try:
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            rows.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mb": float(parts[2]),
                "memory_free_mb": float(parts[3]),
                "utilization_percent": float(parts[4]),
                "power_watts": None if parts[5] in {"[N/A]", "N/A"} else float(parts[5]),
            })
    return {"gpus": rows}


def detect_hardware() -> HardwareProfile:
    notebook = "ipykernel" in sys.modules or bool(os.environ.get("JPY_PARENT_PID"))
    colab = "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))
    smi = _smi()
    smi_by_index = {row["index"]: row for row in (smi or {}).get("gpus", [])}
    gpus: list[GpuProfile] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            smi_row = smi_by_index.get(index, {})
            gpus.append(GpuProfile(
                index=index,
                name=props.name,
                total_vram_gb=props.total_memory / 2**30,
                available_vram_gb=(smi_row.get("memory_free_mb") / 1024 if smi_row else None),
                compute_capability=(props.major, props.minor),
            ))
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    # Some container/overlay filesystems report an INT64_MAX-like sentinel
    # rather than a meaningful quota. Do not present that as petabytes of
    # usable storage; preserve unknown as null for downstream estimates.
    disk_free_gb = disk.free / 2**30
    if disk.total >= 2**62 or disk_free_gb > 1_000_000:
        disk_free_gb = None
    import multiprocessing as mp
    return HardwareProfile(
        operating_system=f"{platform.system()} {platform.release()}",
        notebook=notebook,
        colab=colab,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        gpu_count=len(gpus),
        gpus=gpus,
        fp16_supported=torch.cuda.is_available(),
        bf16_supported=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        bf16_native=_native_bf16(),
        tf32_supported=bool(gpus and (gpus[0].compute_capability or (0, 0))[0] >= 8),
        mps_available=bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        physical_cpu_cores=psutil.cpu_count(logical=False),
        logical_cpu_threads=psutil.cpu_count(logical=True) or 1,
        available_ram_gb=vm.available / 2**30,
        total_ram_gb=vm.total / 2**30,
        storage_free_gb=disk_free_gb,
        torch_compile_available=hasattr(torch, "compile"),
        multiprocessing_start_method=mp.get_start_method(allow_none=True),
        nvidia_smi=smi,
    )


def resolve_device(requested: str = "auto", gpu_id: int | None = None) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device(f"cuda:{gpu_id or 0}")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(f"cuda:{gpu_id or 0}")
    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device, profile: HardwareProfile | None = None) -> str:
    if requested != "auto":
        if requested == "bf16" and device.type == "cuda" and not (profile or detect_hardware()).bf16_native:
            raise RuntimeError("native bf16 is unavailable; choose fp16 or fp32")
        if requested == "fp16" and device.type != "cuda":
            raise RuntimeError("fp16 trial execution is currently enabled only for CUDA; use fp32 on CPU/MPS")
        return requested
    profile = profile or detect_hardware()
    if device.type == "cuda":
        return "bf16" if profile.bf16_native else "fp16"
    if device.type == "mps":
        # The repository training loop has CUDA-specific autocast/scaler paths.
        # Use a conservative FP32 MPS fallback until a measured MPS AMP adapter
        # is added and verified.
        return "fp32"
    return "fp32"


def import_gpu_check(repo_root: str | Path):
    path = Path(repo_root) / "src" / "common" / "gpu_check.py"
    spec = importlib.util.spec_from_file_location("washington_gpu_check", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

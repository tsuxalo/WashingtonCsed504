"""
gpu_check.py — Shared device detection and multi-GPU configuration for CSED 504.

Handles NVIDIA CUDA (single- and multi-GPU), Apple MPS, and CPU fallback.  Same
priority order used throughout the class:  CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.

Import from any notebook in the project:

    import os, sys
    sys.path.insert(0, os.path.normpath(os.path.join(os.getcwd(), '../common')))
    from gpu_check import get_device, set_seed
    DEVICE = get_device()          # detects + configures all usable GPUs
    set_seed(42)

For multi-GPU training, wrap the model after moving it to DEVICE:

    from gpu_check import get_data_parallel_model
    model = get_data_parallel_model(model.to(DEVICE), DEVICE)   # DataParallel if >1 GPU

Can also be run standalone as a sanity check:
    conda activate uw-csed504
    python src/common/gpu_check.py               # human-readable GPU report
    python src/common/gpu_check.py --smoke-test  # + FP16 matmul on each GPU
    python src/common/gpu_check.py --print-cvd   # machine-readable CUDA_VISIBLE_DEVICES


A note on OMP Error #15
-----------------------
OMP Error #15 ("Initializing libiomp5md.dll, but found libiomp5md.dll already
initialized") is NOT a threading problem, and is NOT fixed by lowering
OMP_NUM_THREADS / MKL_NUM_THREADS.  It is caused by *two copies* of the Intel
OpenMP runtime (libiomp5md.dll) being loaded into one process — typically one
from conda's MKL-backed numpy and one from PyTorch's own wheel.  The fix lives
in setup_windows.ps1: install everything via pip so PyTorch is the *only*
provider of libiomp5md.dll.  This module therefore does NOT touch OMP_NUM_THREADS.


GPU selection rules
-------------------
Cards are ranked by "power" = (VRAM, compute capability): more memory wins, and a
newer architecture only breaks ties between cards with the same VRAM.  VRAM leads
because for model training the memory a card holds (weights + activations + batch)
matters more than a newer arch — a 32 GB Ampere beats an 8 GB Ada.

  • Single GPU                         → use it
  • Several IDENTICAL cards            → use ALL  (matched DataParallel is fine)
    (same VRAM AND same architecture)
  • Any mismatch (different VRAM, or   → use ONLY the single most powerful card
    same VRAM but different arch)

Selection is driven entirely by the hardware present, so the same code is correct
on every machine:
  • dual RTX PRO 6000 workstation (two matched sm_120 cards) → exposes both for
    data-parallel training;
  • single-GPU laptop (e.g. an RTX 2000 Ada, sm_89)          → uses its one card;
  • mixed rig, e.g. a 96 GB RTX PRO 6000 (sm_120) + a 32 GB RTX 5000 Ada (sm_89)
    → uses the RTX PRO 6000 alone and hides the Ada, because DataParallel across
    mismatched cards causes stragglers and OOMs the smaller card;
  • memory over generation: a 32 GB Ampere (sm_86) + an 8 GB Ada (sm_89) → uses the
    32 GB Ampere, since more VRAM beats a newer architecture for training.
Nothing here is hard-coded to a specific GPU count or model.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass


# ─── GPU descriptor ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    smi_index:     int    # physical nvidia-smi index (unaffected by CUDA_VISIBLE_DEVICES)
    name:          str
    compute_major: int    # e.g. 12 for Blackwell sm_120, 8 for Ada sm_89
    compute_minor: int    # e.g.  0 for sm_120,           9 for sm_89
    vram_gb:       float
    uuid:          str = ""  # e.g. GPU-b09775ec-7817-5003-b4ac-cc406fac5a51

    @property
    def sm(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}"

    @property
    def compute_cap(self) -> tuple[int, int]:
        return (self.compute_major, self.compute_minor)


# ─── GPU enumeration ────────────────────────────────────────────────────────────────────────────

def _query_smi_gpus() -> list[GpuInfo]:
    """
    Query all physical GPUs via nvidia-smi.

    Unlike torch.cuda, nvidia-smi is NOT affected by CUDA_VISIBLE_DEVICES, so it
    always reports the full set of installed hardware regardless of what the
    calling environment has restricted.  Returns [] if nvidia-smi is missing
    (e.g. on macOS or a CPU-only box).
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if result.returncode != 0:
        return []

    gpus: list[GpuInfo] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx       = int(parts[0])
            name      = parts[1]
            cap_parts = parts[2].split(".")
            major     = int(cap_parts[0])
            minor     = int(cap_parts[1]) if len(cap_parts) > 1 else 0
            vram_gb   = round(int(parts[3]) / 1024, 1)
            uuid      = parts[4]
            gpus.append(
                GpuInfo(
                    smi_index=idx,
                    name=name,
                    compute_major=major,
                    compute_minor=minor,
                    vram_gb=vram_gb,
                    uuid=uuid,
                )
            )
        except (ValueError, IndexError):
            continue

    return gpus


def _select_gpus(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """
    Return the subset of GPUs that should be made visible to PyTorch.

    "Power" is ranked by (VRAM, compute capability): more memory wins, and a newer
    architecture only breaks ties between cards with the SAME VRAM.  VRAM leads on
    purpose — for model training the amount of memory (weights + activations + batch
    that fit) matters more than a newer architecture, so a 32 GB Ampere is preferred
    over an 8 GB Ada.

    Rules:
      • 0 or 1 GPUs                       → return as-is
      • Several IDENTICAL cards tie for    → return them all; DataParallel across
        most powerful (same VRAM AND         matched cards is safe
        same arch)
      • Any mismatch — different VRAM, or  → return ONLY the single most powerful
        same VRAM but different arch          card

    Why we don't keep mismatched cards: torch.nn.DataParallel splits every batch
    *evenly* across the visible GPUs, so pairing a large/fast card with a smaller or
    slower one makes the strong card wait on the weak one (stragglers) and OOMs the
    smaller card's shard.  A real mixed rig — e.g. a 96 GB RTX PRO 6000 (sm_120) next
    to a 32 GB RTX 5000 Ada (sm_89) — therefore runs on the RTX PRO 6000 alone.  We
    only expose multiple GPUs when they are truly matched.
    """
    if len(gpus) <= 1:
        return gpus

    # Rank cards by (VRAM, compute capability); the max is the most powerful card.
    # VRAM comes first so a bigger card beats a newer-but-smaller one (a 32 GB Ampere
    # over an 8 GB Ada); architecture only decides between equal-VRAM cards.  Keep
    # every card whose (VRAM, arch) equals that best — i.e. identical siblings for
    # DataParallel — and hide everything else.  On a mismatched rig only the one
    # strongest card matches, so we fall back to using it alone.
    best_key = max((g.vram_gb, g.compute_cap) for g in gpus)
    return [g for g in gpus if (g.vram_gb, g.compute_cap) == best_key]


# ─── CUDA configuration ─────────────────────────────────────────────────────────────────────────

def configure_cuda(verbose: bool = True) -> list[int]:
    """
    Detect GPUs, select the optimal set, set CUDA_VISIBLE_DEVICES, and return the
    selected nvidia-smi indices.

    Should be called BEFORE PyTorch initialises a CUDA context in the current
    process (i.e. before the first ``torch.cuda.*`` call).  If CUDA is already
    initialised a warning is printed; the env-var is still updated so that child
    processes launched afterward see the right set.

    The number of UUIDs written depends on the machine: two on the dual RTX PRO 6000
    workstation (both sm_120, so device_count() reports 2), one on a single-GPU laptop.

    Returns:
        List of nvidia-smi GPU indices selected (empty list for CPU-only / non-NVIDIA).
    """
    # Warn if torch has already locked in a CUDA context for this process.
    try:
        import torch as _torch  # type: ignore[import]
        if _torch.cuda.is_initialized():
            print(
                "[gpu_check] WARNING: torch CUDA is already initialised. "
                "Setting CUDA_VISIBLE_DEVICES now affects child processes but has "
                "NO effect on the current kernel. Restart the kernel and call "
                "get_device()/configure_cuda() before any torch CUDA call.",
                file=sys.stderr,
            )
    except ImportError:
        pass  # torch not installed yet — fine

    gpus = _query_smi_gpus()

    if not gpus:
        if verbose:
            print("[gpu_check] No NVIDIA GPUs found via nvidia-smi (CPU/MPS only).")
        return []

    selected = _select_gpus(gpus)
    hidden   = [g for g in gpus if g not in selected]

    # Prefer UUIDs — hardware-stable identifiers that never reorder even if the
    # physical slot order changes.  CUDA accepts both integer indices and
    # "GPU-<uuid>" strings in CUDA_VISIBLE_DEVICES.
    if selected and selected[0].uuid:
        cvd = ",".join(g.uuid for g in selected)
    else:
        cvd = ",".join(str(g.smi_index) for g in selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = cvd

    if verbose:
        n_total, n_sel = len(gpus), len(selected)
        best = selected[0]
        if n_sel == n_total == 1:
            desc = "single GPU"
        elif n_sel == n_total:
            desc = f"all {n_sel} GPUs (identical {best.sm} / {best.vram_gb:.0f} GB - DataParallel OK)"
        elif n_sel == 1:
            desc = (f"most powerful of {n_total} GPUs ({best.sm} / {best.vram_gb:.0f} GB; "
                    "mismatched GPU(s) hidden to avoid DataParallel stragglers/OOM)")
        else:
            desc = (f"{n_sel} of {n_total} GPUs (matched {best.sm} / {best.vram_gb:.0f} GB - "
                    "DataParallel OK within group; mismatched GPU(s) hidden)")
        print(f"[gpu_check] Using {desc}")
        print(f"  CUDA_VISIBLE_DEVICES = {cvd!r}")
        for i, g in enumerate(selected):
            print(f"  cuda:{i}  {g.name}  {g.sm}  {g.vram_gb:.1f} GB")
        for g in hidden:
            print(f"  [hidden]  {g.name}  {g.sm}  {g.vram_gb:.1f} GB")

    return [g.smi_index for g in selected]


def config_multigpu_env(verbose: bool = True) -> None:
    """
    Backward-compatible shim retained for older notebook cells.

    Historically this set OMP_NUM_THREADS / MKL_NUM_THREADS in the mistaken belief
    that doing so cured OMP Error #15.  It does not (see the module docstring).
    It now simply delegates to configure_cuda(), which selects and exposes all
    usable GPUs — a no-op on non-NVIDIA machines.
    """
    if os.name == "nt" or _query_smi_gpus():
        configure_cuda(verbose=verbose)


# ─── Device detection ───────────────────────────────────────────────────────────────────────────

def get_device(verbose: bool = True, config_multigpu: bool = True):
    """
    Return the best available torch.device and optionally print a status line.

    Priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.  The MPS smoke-test
    catches older macOS versions that advertise MPS support but fail on the first
    tensor operation.

    This ordering is by device *type*, never by memory: a GPU is always preferred
    over the CPU, so an 8 GB GPU beats a 64 GB CPU.  The "more memory wins" rule in
    _select_gpus ranks GPUs against each OTHER; it never weighs GPU VRAM against
    system RAM.

    Parameters
    ----------
    verbose : bool, optional
        If True (default), print device information.
    config_multigpu : bool, optional
        If True (default), call configure_cuda() first so CUDA_VISIBLE_DEVICES
        exposes every usable GPU before torch initialises CUDA.  Set to False if
        you have already configured it (e.g. via the conda env's activate.d).
    """
    # Configure GPU visibility BEFORE the first torch CUDA call.  Setting the env
    # var here is only effective if CUDA is not yet initialised in this process,
    # which is the normal case for the first cell of a fresh kernel.
    if config_multigpu:
        configure_cuda(verbose=False)

    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n      = torch.cuda.device_count()
        if verbose:
            print(f"  PyTorch  : {torch.__version__}")
            print(f"  CUDA     : {torch.version.cuda}")
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
            cvd_label = f"CUDA_VISIBLE_DEVICES={cvd!r}" if cvd else "CUDA_VISIBLE_DEVICES not set"
            print(f"  Device   : cuda  [{n} GPU{'s' if n > 1 else ''} visible - {cvd_label}]")
            for i in range(n):
                p    = torch.cuda.get_device_properties(i)
                vram = p.total_memory / 1e9
                mark = "  <- cuda:0 (primary)" if i == 0 else ""
                print(f"    cuda:{i}  {p.name}  sm_{p.major}{p.minor}  {vram:.1f} GB{mark}")
            if n > 1:
                print(f"  Multi-GPU ({n} cards) - pick by goal:")
                print( "    - FASTER inner loop: stay on ONE card + enable_fast_matmul() + "
                       "bf16 autocast. DataParallel rarely helps at class model scale")
                print( "      (comms overhead cancels the 2nd card); measure before relying on it.")
                print( "    - MORE experiments at once: launch a separate run per card")
                print( "    - BIGGER model (won't fit on 1 card): "
                       'device_map="auto", max_memory=get_max_memory()')
        return device

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.zeros(1, device="mps")   # smoke-test: older macOS may advertise MPS but fail
            device = torch.device("mps")
            if verbose:
                print(f"  PyTorch  : {torch.__version__}")
                print("  Device   : MPS - Apple Silicon GPU (unified memory)")
            return device
        except Exception:
            pass  # fall through to CPU

    device = torch.device("cpu")
    if verbose:
        print(f"  PyTorch  : {torch.__version__}")
        print(f"  Device   : CPU ({os.cpu_count() or 1} logical cores)")
    return device


# ─── Multi-GPU helpers ──────────────────────────────────────────────────────────────────────────

def get_num_gpus() -> int:
    """Return the number of CUDA GPUs visible to PyTorch (0 if none / torch absent)."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def get_data_parallel_model(model, device, verbose: bool = True):
    """
    Wrap a model with torch.nn.DataParallel when more than one CUDA GPU is visible.

    DataParallel splits each input batch across the visible GPUs, runs the forward
    pass in parallel, and gathers the results on cuda:0 — the simplest way to use
    multiple same-architecture cards (e.g. the workstation's two RTX PRO 6000s).
    Returns the model unchanged on single-GPU / CPU / MPS.

    Parameters
    ----------
    model : torch.nn.Module
        Model that has already been moved to ``device`` (e.g. ``model.to(device)``).
    device : torch.device
        Target device; DataParallel only activates when this is CUDA.
    verbose : bool, optional
        If True (default), print when wrapping.
    """
    import torch

    if getattr(device, "type", str(device)) != "cuda":
        return model

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if verbose:
            print(f"  Using nn.DataParallel across {num_gpus} GPUs")
        return torch.nn.DataParallel(model)

    return model


def enable_fast_matmul(tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    """
    Enable Blackwell-friendly fast math for training throughput.

    This is the main lever for a faster inner loop when the model already fits on one
    card (measured ~1.8x vs fp32 on the RTX PRO 6000 workstation; a second GPU via
    DataParallel gave ~1.0x at class scale there, i.e. no gain — measure on your own
    hardware, since the speedup varies by GPU):

      - TF32 matmuls (``set_float32_matmul_precision("high")``): large speedup for a tiny
        precision cost on fp32 paths.
      - cuDNN benchmark: autotunes conv/attention kernels for FIXED input shapes. Turn it
        off if your batch/sequence shapes vary a lot, or if you need bitwise reproducibility
        (note: this re-enables what set_seed() disabled, trading determinism for speed).

    Pair with bf16 autocast in the training step for the full gain:

        from gpu_check import enable_fast_matmul
        enable_fast_matmul()
        ...
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = loss_fn(model(x), y)
        loss.backward(); opt.step()

    (bf16 needs no GradScaler, unlike fp16 — simpler and numerically safe on Blackwell.)
    """
    import torch

    if tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    if cudnn_benchmark:
        # set_seed() sets cudnn.deterministic=True, and callers run it BEFORE this function --
        # which silently restricted the autotuner to deterministic-only algorithms.  A caller
        # asking for speed (benchmark=True) gets the full kernel menu back.
        torch.backends.cudnn.deterministic = False


def get_max_memory(reserve_gib: float = 5.0) -> dict:
    """
    Build a ``max_memory`` dict for HuggingFace/Accelerate ``device_map="auto"``.

    DataParallel (get_data_parallel_model) replicates the WHOLE model on each GPU, so
    the model must fit on ONE card — it only raises throughput/batch size.  To train or
    run a model LARGER than a single card, use model (pipeline) parallelism, which
    splits the model's layers ACROSS the visible GPUs and pools their VRAM (needs 2+
    GPUs; on the workstation the two 96 GB RTX PRO 6000s give ~192 GB combined):

        from transformers import AutoModelForCausalLM
        from gpu_check import get_max_memory
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype="auto",
            device_map="auto",              # split layers across all visible GPUs
            max_memory=get_max_memory(),    # reserve headroom per card for activations
        )
        # inputs go to model.hf_device_map's first device; Accelerate moves activations
        # across the GPU boundary automatically. No NCCL required (single process),
        # which is why this is the viable multi-card path on Windows.

    ``device_map="auto"`` computes its own budget if you omit ``max_memory``; pass this
    helper when you want to leave room per card so activations don't OOM.

    Parameters
    ----------
    reserve_gib : float, optional
        GiB to hold back on each GPU for activations/overhead (default 5.0).

    Returns
    -------
    dict
        e.g. ``{0: "90GiB", 1: "90GiB"}`` — usable as ``max_memory=`` for device_map.
    """
    import torch

    max_mem: dict = {}
    for i in range(torch.cuda.device_count()):
        total_gib = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        usable = max(1, int(total_gib - reserve_gib))
        max_mem[i] = f"{usable}GiB"
    return max_mem


def preflight_check(required_data: list[tuple[str, str]] | None = None,
                    smoke_test: bool = True) -> None:
    """
    Validate that PyTorch CUDA is correctly configured before a long training run.

    Checks:
      1. CUDA is available.
      2. All visible GPUs are matched — same architecture AND same VRAM. Mixing
         cards (different arch, or same arch but different memory) makes
         DataParallel split batches unevenly across unequal GPUs: the strong card
         waits on the weak one and the smaller card OOMs. configure_cuda() hides
         mismatched cards, so this only trips on a manual CUDA_VISIBLE_DEVICES.
      3. FP16 matmul smoke test on cuda:0 (optional, on by default).
      4. Required data files exist on disk (optional).

    Raises RuntimeError with an actionable message if any check fails.
    """
    import torch

    errors: list[str] = []

    if not torch.cuda.is_available():
        errors.append(
            "torch.cuda.is_available() is False. Re-run from the top of the "
            "notebook so get_device() runs before any torch CUDA call."
        )
        _raise_if_errors(errors)
        return

    n_gpus    = torch.cuda.device_count()
    gpu_props = [torch.cuda.get_device_properties(i) for i in range(n_gpus)]

    if n_gpus > 1:
        # Round VRAM to the nearest 0.1 GB so tiny reporting jitter between two
        # identical cards doesn't look like a mismatch.
        specs = {((p.major, p.minor), round(p.total_memory / 1e9, 1)) for p in gpu_props}
        if len(specs) > 1:
            descs = [f"{p.name} (sm_{p.major}{p.minor}, {p.total_memory / 1e9:.1f} GB)"
                     for p in gpu_props]
            errors.append(
                f"Mismatched GPUs visible to PyTorch: {descs}. DataParallel needs "
                "identical cards (same architecture AND VRAM). Restart the kernel and "
                "let get_device()/configure_cuda() keep only the most powerful card, "
                "or set CUDA_VISIBLE_DEVICES to a single card."
            )

    if smoke_test and not errors:
        try:
            sz = 2048
            a  = torch.randn(sz, sz, dtype=torch.float16, device="cuda")
            b  = torch.matmul(a, a.T)
            assert b.shape == (sz, sz), "unexpected output shape"
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            errors.append(f"FP16 smoke-test failed on {torch.cuda.get_device_name(0)}: {exc}")

    if required_data:
        for path, label in required_data:
            if not os.path.exists(path):
                errors.append(f"Missing {label}: {path!r}. Re-run the download cell.")

    _raise_if_errors(errors)

    parallel_note = ("DataParallel will NOT activate" if n_gpus == 1
                     else f"DataParallel across {n_gpus} matched GPUs (OK)")
    print("Pre-flight checks passed.")
    print(f"  GPUs visible : {n_gpus}  ({parallel_note})")
    for i, p in enumerate(gpu_props):
        print(f"  cuda:{i}  {p.name}  sm_{p.major}{p.minor}  {p.total_memory / 1e9:.1f} GB")
    if smoke_test:
        print("  FP16 matmul  : OK  (2048x2048 on-device)")
    if required_data:
        print(f"  Data files   : {len(required_data)} required file(s) present")


def _raise_if_errors(errors: list[str]) -> None:
    if errors:
        for msg in errors:
            print(f"[FAIL] {msg}", file=sys.stderr)
        raise RuntimeError(
            f"Pre-flight checks failed ({len(errors)} issue(s)) — fix the issues above."
        )


# ─── Reproducibility ────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Seed everything for reproducibility: torch (CPU + all GPUs), numpy, Python random.

    Also disables cuDNN benchmark mode so forward-pass timing is deterministic —
    re-running a cell gives the same weights, gradients, and outputs.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)        # seeds every visible GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─── Standalone / CLI usage ─────────────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    # Windows consoles default to cp1252 when stdout is redirected (e.g. Tee-Object
    # in PowerShell), which chokes on any non-ASCII byte.  Force UTF-8 so the report
    # never dies on encoding.  Harmless where stdout is already UTF-8.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Detect GPUs, configure CUDA_VISIBLE_DEVICES, and verify PyTorch."
    )
    parser.add_argument(
        "--print-cvd", action="store_true",
        help="Print ONLY the optimal CUDA_VISIBLE_DEVICES value (for capture by shell scripts).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run a 2048x2048 FP16 matmul smoke test after the GPU report.",
    )
    args = parser.parse_args()

    if args.print_cvd:
        gpus = _query_smi_gpus()
        if gpus:
            selected = _select_gpus(gpus)
            if selected and selected[0].uuid:
                print(",".join(g.uuid for g in selected))
            else:
                print(",".join(str(g.smi_index) for g in selected))
        return  # empty output → caller should not restrict CUDA_VISIBLE_DEVICES

    configure_cuda(verbose=True)
    print()
    get_device(verbose=True, config_multigpu=False)
    if args.smoke_test:
        try:
            import torch
            if torch.cuda.is_available():
                print()
                preflight_check(smoke_test=True)
        except ImportError:
            print("(torch not installed — skipping smoke test)")


if __name__ == "__main__":
    _main()

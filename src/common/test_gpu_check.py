"""
test_gpu_check.py — Hardware-selection tests for gpu_check._select_gpus.

These tests pin down which GPU(s) get exposed to PyTorch on a given rig, WITHOUT
needing the hardware present: each case builds fake GpuInfo objects and checks the
selection.  That lets us verify the policy on every machine configuration we care
about (single laptop GPU, matched multi-GPU workstation, mismatched "mixed" rigs)
from any box — including CPU-only CI.

Run it either way:

    python src/common/test_gpu_check.py     # standalone: prints a PASS/FAIL table
    pytest src/common/test_gpu_check.py      # if pytest is installed

Selection policy under test (see gpu_check._select_gpus):
  • Cards are ranked by (VRAM, compute capability) — more memory wins, newer arch
    only breaks ties between equal-VRAM cards (a 32 GB Ampere beats an 8 GB Ada,
    because VRAM matters more than generation for training).
  • Several IDENTICAL cards (same VRAM AND same arch) → use ALL (matched DataParallel).
  • Any mismatch (different VRAM, or same VRAM but different arch) → use ONLY the
    single most powerful card, because DataParallel splits batches evenly and a
    mismatched pair stalls on the weaker card and OOMs the smaller one.

Device-priority under test (see gpu_check.get_device):
  • The "more memory wins" rule only ranks GPUs against each OTHER.  Across device
    *types* a GPU always beats the CPU regardless of memory — an 8 GB GPU is
    preferred over a 64 GB CPU.  These tests run get_device() against a fake torch,
    so they need no real GPU.

Adding a case for NEW hardware
------------------------------
1. If it's a card we haven't modeled, add it to CATALOG below (name, sm major/minor,
   VRAM in GB — read these from `nvidia-smi --query-gpu=name,compute_cap,memory.total`).
2. Add a (label, [cards], [expected selected names]) row to CASES.
That's it — both the standalone runner and pytest pick it up automatically.
"""

from __future__ import annotations

import os
import sys

# Import gpu_check no matter the current working directory: add THIS file's folder
# (src/common) to sys.path, then import by module name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_check import GpuInfo, _select_gpus  # noqa: E402


# ─── Modeled hardware ────────────────────────────────────────────────────────────────────────────
# name -> (compute_major, compute_minor, vram_gb).  VRAM is the nvidia-smi memory.total
# rounded to match GpuInfo.vram_gb.  Extend this as new cards show up.
CATALOG: dict[str, tuple[int, int, float]] = {
    "RTX PRO 6000": (12, 0, 96.0),   # Blackwell workstation, sm_120
    "RTX 5000 Ada": (8, 9, 32.0),    # Ada, sm_89
    "RTX 4090":     (8, 9, 24.0),    # Ada, sm_89 — same arch as 5000 Ada, less VRAM
    "RTX A5000":    (8, 6, 24.0),    # Ampere, sm_86 — older arch than Ada, but plenty of VRAM
    "RTX 2000 Ada": (8, 9, 8.0),     # Ada laptop GPU, sm_89 (this machine)
}


def _gpu(name: str, index: int) -> GpuInfo:
    """Build a GpuInfo for a catalog card at a given physical nvidia-smi index."""
    major, minor, vram = CATALOG[name]
    return GpuInfo(
        smi_index=index,
        name=name,
        compute_major=major,
        compute_minor=minor,
        vram_gb=vram,
        uuid=f"GPU-{name.replace(' ', '-').lower()}-{index}",
    )


def _rig(*names: str) -> list[GpuInfo]:
    """Build a list of GpuInfo from card names, assigning slot indices in order."""
    return [_gpu(name, i) for i, name in enumerate(names)]


# ─── Scenarios ───────────────────────────────────────────────────────────────────────────────────
# (label, installed cards, expected selected card names).  Order of expected names
# follows _select_gpus's output order (physical slot order among the matched cards).
CASES: list[tuple[str, list[GpuInfo], list[str]]] = [
    ("no GPUs (CPU/MPS only)",
     _rig(),
     []),

    ("single laptop GPU",
     _rig("RTX 2000 Ada"),
     ["RTX 2000 Ada"]),

    ("dual matched workstation",
     _rig("RTX PRO 6000", "RTX PRO 6000"),
     ["RTX PRO 6000", "RTX PRO 6000"]),

    ("mixed rig: PRO 6000 + 5000 Ada -> strongest only",
     _rig("RTX PRO 6000", "RTX 5000 Ada"),
     ["RTX PRO 6000"]),

    ("mixed order: 5000 Ada listed first -> still PRO 6000",
     _rig("RTX 5000 Ada", "RTX PRO 6000"),
     ["RTX PRO 6000"]),

    ("2x PRO 6000 + 5000 Ada -> the two matched cards",
     _rig("RTX PRO 6000", "RTX PRO 6000", "RTX 5000 Ada"),
     ["RTX PRO 6000", "RTX PRO 6000"]),

    ("same arch, different VRAM (5000 Ada + 4090) -> larger only",
     _rig("RTX 5000 Ada", "RTX 4090"),
     ["RTX 5000 Ada"]),

    ("memory over generation: 24 GB Ampere (A5000) + 8 GB Ada (2000) -> more memory wins",
     _rig("RTX A5000", "RTX 2000 Ada"),
     ["RTX A5000"]),

    ("three identical cards -> use all three",
     _rig("RTX PRO 6000", "RTX PRO 6000", "RTX PRO 6000"),
     ["RTX PRO 6000", "RTX PRO 6000", "RTX PRO 6000"]),
]


def _selected_names(gpus: list[GpuInfo]) -> list[str]:
    return [g.name for g in _select_gpus(gpus)]


# ─── pytest entry points ─────────────────────────────────────────────────────────────────────────

def test_select_gpus_scenarios() -> None:
    """Every rig in CASES selects exactly the expected card(s)."""
    for label, gpus, expected in CASES:
        got = _selected_names(gpus)
        assert got == expected, f"{label!r}: selected {got}, expected {expected}"


def test_vram_outranks_generation() -> None:
    """
    Documents the deliberate ranking: MORE MEMORY beats a newer architecture.  For
    model training a 32 GB Ampere (sm_86) is preferred over an 8 GB Ada (sm_89) — the
    bigger card fits more model / activations / batch.  If this ever flips
    (generation-first), update _select_gpus AND this test together.
    """
    ampere_32 = GpuInfo(0, "NVIDIA Ampere 32 GB", 8, 6, 32.0, uuid="GPU-ampere32")
    ada_8     = GpuInfo(1, "NVIDIA Ada 8 GB",     8, 9,  8.0, uuid="GPU-ada8")
    assert _selected_names([ampere_32, ada_8]) == ["NVIDIA Ampere 32 GB"]


def test_equal_vram_prefers_newer_arch() -> None:
    """When VRAM ties, the newer architecture breaks the tie (Ada sm_89 > Ampere sm_86)."""
    ada_24    = GpuInfo(0, "Ada 24 GB",    8, 9, 24.0, uuid="GPU-ada24")
    ampere_24 = GpuInfo(1, "Ampere 24 GB", 8, 6, 24.0, uuid="GPU-ampere24")
    assert _selected_names([ada_24, ampere_24]) == ["Ada 24 GB"]


def test_selection_is_order_independent() -> None:
    """Selecting is a function of the hardware set, not the slot order it's listed in."""
    forward = _selected_names(_rig("RTX PRO 6000", "RTX 5000 Ada"))
    reverse = _selected_names(_rig("RTX 5000 Ada", "RTX PRO 6000"))
    assert forward == reverse == ["RTX PRO 6000"]


# ─── Device-priority: a GPU always beats the CPU (regardless of memory) ─────────────────────────────
# get_device() picks by device TYPE (CUDA > MPS > CPU), never by memory, so we can
# exercise the decision with a stand-in "torch" instead of real hardware.

class _FakeCuda:
    def __init__(self, available: bool, count: int = 1):
        self._available, self._count = available, count

    def is_available(self) -> bool: return self._available
    def is_initialized(self) -> bool: return False
    def device_count(self) -> int: return self._count


class _FakeDevice:
    def __init__(self, kind: str): self.type = kind
    def __repr__(self) -> str: return f"device(type='{self.type}')"


class _FakeTorch:
    """Just enough of the torch surface for get_device(config_multigpu=False, verbose=False)."""
    __version__ = "fake"

    def __init__(self, cuda_available: bool):
        self.cuda = _FakeCuda(cuda_available)

        class _Backends:  # no `mps` attribute -> get_device's MPS branch is skipped
            pass

        self.backends = _Backends()

    @staticmethod
    def device(kind: str) -> _FakeDevice: return _FakeDevice(kind)


def _device_type_with(cuda_available: bool) -> str:
    """Return get_device()'s chosen device type with torch replaced by a fake."""
    import gpu_check

    saved = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorch(cuda_available)
    try:
        # config_multigpu=False avoids touching nvidia-smi; verbose=False keeps it quiet.
        return gpu_check.get_device(verbose=False, config_multigpu=False).type
    finally:
        if saved is not None:
            sys.modules["torch"] = saved
        else:
            sys.modules.pop("torch", None)


def test_gpu_preferred_over_cpu_regardless_of_memory() -> None:
    """
    An 8 GB GPU beats a 64 GB CPU: whenever a CUDA GPU is available, get_device()
    returns CUDA and never weighs GPU VRAM against system RAM.  ('More memory wins'
    ranks GPUs against each other — see _select_gpus — not GPU vs CPU.)
    """
    assert _device_type_with(cuda_available=True) == "cuda"


def test_cpu_fallback_when_no_gpu() -> None:
    """With no CUDA GPU (and no MPS), get_device() falls back to the CPU."""
    assert _device_type_with(cuda_available=False) == "cpu"


# ─── Standalone runner (no pytest needed) ──────────────────────────────────────────────────────────

def _main() -> int:
    passed = failed = 0
    print("GPU selection tests (gpu_check._select_gpus)\n")
    for label, gpus, expected in CASES:
        got = _selected_names(gpus)
        ok = got == expected
        passed += ok
        failed += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"         selected {got}")
            print(f"         expected {expected}")

    # The policy tests above, run inline so the standalone report covers them too.
    for name, fn in (("VRAM outranks generation",       test_vram_outranks_generation),
                     ("equal VRAM -> newer arch",        test_equal_vram_prefers_newer_arch),
                     ("order independence",              test_selection_is_order_independent),
                     ("GPU beats CPU (any memory)",      test_gpu_preferred_over_cpu_regardless_of_memory),
                     ("CPU fallback when no GPU",        test_cpu_fallback_when_no_gpu)):
        try:
            fn()
            passed += 1
            print(f"[PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"[FAIL] {name}: {exc}")

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())

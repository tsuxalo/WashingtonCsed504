"""perfkit.py -- measure a machine, characterize a training workload, and estimate wall-clock
training time before committing to the full run.

Why this exists: the same 40-epoch CIFAR-100 resnet18 run takes ~2.1 min on the dual RTX PRO 6000
workstation and ~10.6 min on an RTX 2000 Ada laptop -- 6.7x apart -- while the spec-sheet FLOPS
ratio of those GPUs is an order of magnitude wider.  Spec-sheet scaling fails because the *binding
constraint* changes with the machine: at 32x32 the workstation is per-step-overhead-bound
(hundreds of kernel launches doing microseconds of math each) while the power-capped laptop is
compute-bound.  So estimation happens in three tiers:

  Tier 0  probe the machine once (~3 min): burst and sustained TFLOPS, memory bandwidth, kernel
          launch overhead, transfer bandwidth -- the machine's real ceilings, never marketing
          numbers (vendor TOPS assume sparsity/fp8; AMP GEMMs accumulate in fp32 at ~half rate).
  Tier 1  roofline the workload against any probe vector (no run needed):
          t_step = max(compute/MFU, memory, launch) with an honest MFU band.  Accuracy: order
          of magnitude / regime triage -- it answers "2 minutes or 2 days?", not "+/-10%".
  Tier 2  calibrate on the target (~2.5 min): run the real training step, thermally soaked,
          and extrapolate.  Accuracy ~5-10% (quote +/-10-15% on power-capped laptops, which
          oscillate: this laptop's steady epochs vary 15.1-19.6s on a multi-minute cycle).

Every run appends a JSON record (fingerprint + probes + calibration + prediction + actual) to a
results/ directory, so cross-machine estimates improve as the database grows.  MFU is calibrated
per architecture: on the same GPU the repo's resnet18 reaches ~30% MFU but the vit only ~16%,
so a single per-machine efficiency factor would misestimate the ViT by ~2x.

Everything runs on cuda / mps / cpu; probes that don't apply return None rather than raising.
"""

from __future__ import annotations

import gc
import glob
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import torch

# The monotonic high-resolution clock, aliased once because nearly every probe below brackets a
# timed region with it. time.time() would be wrong here: it can step backwards on a clock sync.
_now = time.perf_counter


# Device plumbing. Two small helpers that hide the cuda / mps / cpu differences, so every probe
# below can be written once and still time the right thing on each backend.

def _sync(device) -> None:
    """Block until all queued device work is done.  Timing without this measures the enqueue
    rate, not execution; timing with it per-step breaks CPU/GPU pipelining -- so callers sync
    only at window boundaries."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()

    # No cpu branch on purpose: eager ops there are synchronous already, so nothing is queued.


def snapshot_backend_flags() -> dict:
    """Record the torch.backends state that changes kernel selection.  A calibration run under
    different flags than the real run is measuring a different program (deterministic=True alone
    can shift conv backward speed), so every DB record carries this snapshot."""

    # The matmul precision setting exists on every backend, so it always goes in.
    flags = {'float32_matmul_precision': torch.get_float32_matmul_precision()}

    # The rest are CUDA/cuDNN specific and only readable when a CUDA device is present. Each one
    # steers kernel choice: benchmark autotunes, deterministic forbids the fast nondeterministic
    # kernels, and the two tf32 flags decide whether fp32 math runs on tensor cores.
    if torch.cuda.is_available():
        flags.update(
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
        )

    return flags


def _run_smi(query: str) -> list[str] | None:
    """One nvidia-smi query, returning its csv fields as a list, or None when nvidia-smi isn't
    available at all (a Mac, a CPU-only box)."""

    # A missing binary or a wedged driver must not take the suite down, so both are caught and
    # reported as "no data". The timeout is what keeps a hung query from stalling a probe.
    try:
        r = subprocess.run(['nvidia-smi', f'--query-gpu={query}',
                            '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    # nvidia-smi can also exit non-zero or print nothing when the query names a field this card
    # doesn't expose; that is no data too, not an error worth raising.
    if r.returncode != 0 or not r.stdout.strip():
        return None

    # Take the first GPU's row and split it into stripped fields.
    return [f.strip() for f in r.stdout.strip().splitlines()[0].split(',')]


# Machine fingerprint. Records what this machine is, precisely enough that a record from it can be
# matched against -- or deliberately kept apart from -- a record taken on another box.

def fingerprint(device) -> dict:
    """Identify this machine configuration.  Power state is part of the identity on purpose:
    the same laptop on battery vs AC is a different machine for throughput purposes (P-state,
    power cap, and boost behavior all change), and mixing the two in the DB would silently
    corrupt cross-machine transfer."""

    # The always-available half of the identity, all from the standard library, so this much of
    # the fingerprint works even on a machine with no optional packages and no GPU.
    fp = {
        'hostname': socket.gethostname(),
        'os': f'{platform.system()} {platform.release()}',
        'python': platform.python_version(),
        'torch': torch.__version__,
        'device_type': device.type,
        'cpu': platform.processor() or platform.machine(),
        'cpu_cores_logical': os.cpu_count(),
    }

    # psutil adds RAM, the physical core count, and the battery state. It is an optional
    # dependency, so a missing install leaves those fields None rather than failing the run.
    try:
        import psutil
        fp['ram_gb'] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        fp['cpu_cores_physical'] = psutil.cpu_count(logical=False)

        # batt is None on a machine with no battery at all, which is why power_plugged stays None
        # there instead of claiming either state.
        batt = psutil.sensors_battery()
        fp['power_plugged'] = None if batt is None else bool(batt.power_plugged)
        fp['battery_percent'] = None if batt is None else round(batt.percent)
    except ImportError:
        fp.update(ram_gb=None, cpu_cores_physical=None, power_plugged=None, battery_percent=None)

    # The Windows power plan gates clocks the same way the battery state does (Balanced caps boost
    # well below High performance), so it belongs in the identity. powercfg prints the active
    # plan's name in parentheses; if the format surprises us, keep the raw line.
    if platform.system() == 'Windows':
        try:
            out = subprocess.run(['powercfg', '/getactivescheme'],
                                 capture_output=True, text=True, timeout=10).stdout
            fp['power_plan'] = out.rsplit('(', 1)[-1].split(')')[0] if '(' in out else out.strip()
        except Exception:
            fp['power_plan'] = None

    # Now the accelerator. On CUDA we record both what the card is and what it is allowed to do:
    # SM count and VRAM set the ceilings, and the power/clock limits explain most of the gap
    # between a laptop and a desktop part of the same generation. MPS exposes no comparable
    # properties, so there the machine string is all we can honestly claim.
    if device.type == 'cuda':
        p = torch.cuda.get_device_properties(device)
        fp.update(gpu=p.name, gpu_sm=f'sm_{p.major}{p.minor}', gpu_vram_gb=round(p.total_memory / 2 ** 30, 1),
                  gpu_sm_count=p.multi_processor_count, cuda=torch.version.cuda)
        smi = _run_smi('driver_version,power.default_limit,power.max_limit,clocks.max.sm')
        if smi:
            fp.update(driver=smi[0], gpu_power_default_w=_float_or_none(smi[1]),
                      gpu_power_max_w=_float_or_none(smi[2]), gpu_max_clock_mhz=_float_or_none(smi[3]))
    elif device.type == 'mps':
        fp['gpu'] = f'Apple MPS ({platform.machine()})'

    # The backend flags travel with the fingerprint: the same silicon under different flags is a
    # different program, and a record that omitted them couldn't be compared to anything.
    fp['backend_flags'] = snapshot_backend_flags()
    return fp


# Parse one nvidia-smi field as a float. Fields a laptop doesn't expose come back as the literal
# '[N/A]' (and a missing field arrives as None), so both become None instead of raising.
def _float_or_none(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def power_state_tag(fp: dict) -> str:
    """'ac' / 'batt' suffix for record filenames.  Machines without a battery count as 'ac', since
    a desktop is never in the throttled state this tag exists to warn about."""
    return 'batt' if fp.get('power_plugged') is False else 'ac'


# Tier 0: hardware probes. Measure this machine's real ceilings once -- compute, bandwidth, launch
# overhead -- so every estimate below rests on numbers from this box rather than a spec sheet.

def _time_op(op, device, min_time_s: float = 0.25, warmup: int = 5) -> float:
    """Seconds per call of op(), warmed up, timed over enough iterations to swamp launch cost.
    Iteration count is derived from a rough first measurement so slow devices don't take forever
    and fast ones aren't dominated by timer resolution."""

    # Warm up first: the opening calls pay for allocation, autotune, and kernel load, none of
    # which the steady-state number should be made to carry.
    for _ in range(warmup):
        op()

    _sync(device)

    # One rough timing, used only to choose an iteration count. The 1e-6 floor keeps an op faster
    # than the clock can resolve from dividing by roughly zero on the next line.
    t0 = _now(); op(); _sync(device)
    rough = max(_now() - t0, 1e-6)

    # Aim for about min_time_s of work, clamped at both ends: at least 3 iterations to average
    # over, at most 200 so a slow device (or the CPU fallback) can't stall the whole suite.
    iters = max(3, min(200, int(min_time_s / rough)))

    # The real measurement. Syncing only at the two ends keeps the queueing outside the timed
    # region, so what we divide is execution time.
    _sync(device)
    t0 = _now()
    for _ in range(iters):
        op()

    _sync(device)
    return (_now() - t0) / iters


def probe_matmul_tflops(device, n: int | None = None) -> dict:
    """Burst GEMM ceiling per dtype (TFLOPS).  This is the MFU denominator, measured -- not the
    vendor number.  fp32/tf32 are distinguished by explicitly setting the matmul precision flags
    (which are restored afterward: the suite must not leak flag changes into the caller's run)."""

    # Size the GEMM to the device: large enough on a CUDA card to actually reach peak, larger
    # still when there is VRAM to hold it, and small elsewhere so a CPU or MPS run still finishes.
    if n is None:
        n = 2048 if device.type != 'cuda' else (8192 if torch.cuda.get_device_properties(device).total_memory > 24 * 2 ** 30 else 4096)

    # A dense n x n matmul is 2*n^3 FLOPs: one multiply and one add per inner-product term.
    flop = 2 * n ** 3

    # out: the per-dtype TFLOPS table, tagged with the size it was measured at, since the number
    # only means something alongside the GEMM size that produced it.
    out: dict = {'gemm_n': n}

    # run: time one dtype and convert to TFLOPS. A dtype this device can't do reports None rather
    # than aborting the whole probe (bf16 on older MPS, for instance).
    def run(dtype) -> float | None:
        try:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            sec = _time_op(lambda: a @ b, device)
            del a, b
            return flop / sec / 1e12
        except (RuntimeError, TypeError):
            return None

    # CUDA gets all four dtypes. tf32 and fp32 run on the same torch.float32 tensors and differ
    # only by the precision flags, so each is selected explicitly; the finally clause puts back
    # whatever the caller had, because a probe that leaked these would alter the run it measures.
    if device.type == 'cuda':
        prec, tf32 = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
        try:
            out['fp16'] = run(torch.float16)
            out['bf16'] = run(torch.bfloat16)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            out['tf32'] = run(torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            out['fp32'] = run(torch.float32)
        finally:
            torch.set_float32_matmul_precision(prec)
            torch.backends.cuda.matmul.allow_tf32 = tf32

        torch.cuda.empty_cache()

    # Elsewhere there are no tensor-core paths to separate: fp32 is the ceiling, plus fp16 on MPS,
    # which supports it.
    else:
        out['fp32'] = run(torch.float32)
        out['fp16'] = run(torch.float16) if device.type == 'mps' else None

    return out


class _SmiPoller:
    """Background nvidia-smi sampler (clock/power/temp every poll_s).  The clock trace is how a
    probe proves it measured sustained state rather than a boost-clock burst."""

    def __init__(self, poll_s: float = 2.0):
        # rows: one sample per poll, appended by the background thread and read once it stops.
        self.rows: list[dict] = []

        # _stop is both the shutdown signal and the sleep timer, so a stop request doesn't have to
        # wait out a full poll interval before the thread notices it.
        self._stop = threading.Event()
        self._poll_s = poll_s
        self._thread = None

    def __enter__(self):
        # Only start a thread where nvidia-smi actually answers; on a Mac or CPU box the poller
        # stays inert and rows simply comes back empty, which the callers already handle.
        if _run_smi('clocks.sm') is not None:
            self._t0 = _now()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

        return self

    def _loop(self):
        # Sample until asked to stop, timestamping each row relative to the poller's own start so
        # the trace lines up with the measurement windows it was taken alongside.
        while not self._stop.is_set():
            f = _run_smi('clocks.sm,power.draw,temperature.gpu')
            if f:
                self.rows.append({'t': round(_now() - self._t0, 1), 'clock_mhz': _float_or_none(f[0]),
                                  'power_w': _float_or_none(f[1]), 'temp_c': _float_or_none(f[2])})

            # Wait on the event rather than sleeping, so __exit__ returns promptly.
            self._stop.wait(self._poll_s)

    def __exit__(self, *exc):
        # Signal the thread and give it a moment to finish the query it is in. The join timeout
        # means a wedged nvidia-smi can delay the caller but never hang it.
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def probe_sustained_tflops(device, seconds: float = 90.0, window_s: float = 2.0) -> dict:
    """The number that governs a 10+ minute run: GEMM TFLOPS after the machine has heated up.
    On this repo's laptop, burst is 33.6 TFLOPS but the 60 W cap drags sustained to ~27 with
    +/-15% oscillation -- exactly the 13.0s becoming 15.5s epoch drift seen in training.  Desktop
    cards typically show <3% degradation; the probe measures rather than assumes.

    Protocol: hammer one GEMM in bounded, synced windows (a wall-clock loop without syncs backs
    up the CUDA launch queue unboundedly -- enqueueing minutes of work in seconds), record each
    window's TFLOPS, report burst = best early window and sustained = median after the first 60 s.
    """

    # Soak in the dtype real training uses on each backend, at a size big enough to keep the
    # device busy without spending the whole probe allocating.
    dtype = torch.float16 if device.type in ('cuda', 'mps') else torch.float32
    n = 4096 if device.type == 'cuda' else 1024
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    flop = 2 * n ** 3

    # Measure windows in iterations rather than seconds: time one GEMM, then pick the count that
    # fills roughly window_s, so every window carries the same amount of work and stays comparable.
    per = _time_op(lambda: a @ b, device, min_time_s=0.1)
    iters = max(3, int(window_s / per))

    # windows: one {t, tflops} sample per window. Keeping them separate is the point -- averaging
    # over the whole soak would hide the burst-to-sustained decay this probe exists to catch.
    windows = []
    with _SmiPoller() as poller:
        t_begin = _now()
        while _now() - t_begin < seconds:
            _sync(device)
            t0 = _now()
            for _ in range(iters):
                a @ b

            _sync(device)
            dt = _now() - t0
            windows.append({'t': round(_now() - t_begin, 1), 'tflops': flop * iters / dt / 1e12})

    # Release the operands before reducing the numbers; nothing below needs them, and the caller
    # may be about to allocate a model.
    del a, b
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # late: the windows recorded after the machine has had time to settle. Two fallbacks in order,
    # so a short soak still reports something honest: the second half, then everything measured.
    late = [w['tflops'] for w in windows if w['t'] > min(60.0, seconds * 0.6)]
    tail = late or [w['tflops'] for w in windows[len(windows) // 2:]] or [w['tflops'] for w in windows]

    # Burst comes from the best of the first few windows, while clocks are still boosted;
    # sustained is the median of the tail. min/max ride along so the caller can see how much the
    # machine oscillates, which is what widens the quoted band on a throttling laptop.
    return {
        'dtype': str(dtype).replace('torch.', ''),
        'burst_tflops': max(w['tflops'] for w in windows[:max(3, len(windows) // 10)]),
        'sustained_tflops': statistics.median(tail),
        'sustained_min_tflops': min(tail), 'sustained_max_tflops': max(tail),
        'windows': windows, 'clock_trace': poller.rows,
    }


def probe_memory_bandwidth(device, target_mb: int = 256) -> dict:
    """Device-memory GB/s via copy (2 bytes moved per element-byte) and triad z = a*x + y
    (3 moves).  Tensors must dwarf L2 (32-128 MB on modern GPUs) or this reports cache speed,
    3-10x too high."""

    # Shrink the request if VRAM is tight: three tensors of this size are allocated below, and the
    # divisor leaves headroom so the probe never OOMs a card someone else is also using.
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        target_mb = min(target_mb, int(free / 2 ** 20 / 8))

    numel = target_mb * 2 ** 20 // 4

    # If even the reduced size doesn't fit, report no bandwidth rather than raising -- the
    # roofline treats a missing memory term as "unknown" and falls back to the other terms.
    try:
        x = torch.randn(numel, device=device)
        y = torch.empty_like(x)
        z = torch.empty_like(x)
    except RuntimeError:
        return {'copy_gbps': None, 'triad_gbps': None}

    # copy touches each element twice (one read, one write); triad three times (two reads, one
    # write), which is why the byte counts below differ by that factor.
    copy_s = _time_op(lambda: y.copy_(x), device)
    triad_s = _time_op(lambda: torch.add(y, x, alpha=2.0, out=z), device)
    bytes_el = numel * 4
    out = {'tensor_mb': target_mb,
           'copy_gbps': 2 * bytes_el / copy_s / 1e9,
           'triad_gbps': 3 * bytes_el / triad_s / 1e9}

    del x, y, z
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return out


def probe_launch_overhead_us(device, iters: int = 10000) -> float:
    """Per-kernel dispatch floor (CPU enqueue + driver submission; Windows WDDM sits at the high
    end).  A resnet18 train step is ~300-500 launches, so 10 us/launch gives a 3-5 ms/step floor
    no GPU speed can beat -- this is the term that makes tiny models overhead-bound on big cards.
    Method: 10k one-element add_ ops, one trailing sync; GPU work is ~0 so wall/iter = launch cost.
    On CPU this returns eager per-op overhead, which plays the same role in the roofline."""

    # A single-element tensor, so each op below dispatches a kernel that does essentially no math
    # and the wall time is dominated by the dispatch itself.
    x = torch.ones(1, device=device)

    # Warm up: the first launches pay for context setup and module load.
    for _ in range(100):
        x.add_(1)

    # One sync at each end and none in between -- a per-iteration sync would measure round-trip
    # latency instead of the enqueue cost we're after.
    _sync(device)
    t0 = _now()
    for _ in range(iters):
        x.add_(1)

    _sync(device)
    return (_now() - t0) / iters * 1e6


def probe_h2d_bandwidth(device, target_mb: int = 256) -> dict | None:
    """Host-to-device transfer: pinned vs pageable GB/s plus small-transfer latency.  Irrelevant
    for this repo's GPU-resident loader but the factory must model conventional DataLoader
    pipelines, where every batch crosses this bus.  None on MPS/CPU (unified memory, no bus)."""
    if device.type != 'cuda':
        return None

    numel = target_mb * 2 ** 20 // 4
    out = {'size_mb': target_mb}

    # Both allocation kinds, because the gap between them is the whole point: pinned memory can be
    # DMA'd directly, while pageable has to be staged through a driver bounce buffer first.
    for kind, pin in (('pinned', True), ('pageable', False)):
        # Pinning is a limited OS resource, so a refused allocation records None and moves on
        # rather than failing the probe.
        try:
            src = torch.empty(numel, pin_memory=pin)
        except RuntimeError:
            out[f'{kind}_gbps'] = None
            continue

        sec = _time_op(lambda: src.to(device, non_blocking=True), device, warmup=2)
        out[f'{kind}_gbps'] = numel * 4 / sec / 1e9
        del src

    # A 1 MB transfer is small enough that fixed per-transfer cost dominates, which is what a
    # many-small-batches pipeline actually pays.
    small = torch.empty(2 ** 20 // 4, pin_memory=True)
    out['small_1mb_ms'] = _time_op(lambda: small.to(device, non_blocking=True), device, warmup=2) * 1e3
    torch.cuda.empty_cache()
    return out


def probe_cpu() -> dict:
    """Host-side speed: a pure-Python loop (single core -- the training loop's driving thread)
    and a torch fp32 GEMM at default thread count (data-prep parallelism).  Both feed the
    launch-overhead and DataLoader terms, not the GPU roofline."""

    # The Python loop stands in for the interpreter work that issues kernels; 5M iterations is
    # enough to swamp startup, and the result is reported in millions of iterations per second.
    t0 = _now()
    s, i = 0, 0
    while i < 5_000_000:
        s += i; i += 1

    py_mops = 5.0 / (_now() - t0)

    # The GEMM stands in for parallel host-side work (decode, augmentation), so it is left at the
    # default thread count rather than pinned to one core.
    n = 2048
    a, b = torch.randn(n, n), torch.randn(n, n)
    sec = _time_op(lambda: a @ b, torch.device('cpu'), min_time_s=0.5, warmup=2)
    return {'python_loop_mops': py_mops, 'torch_fp32_gflops': 2 * n ** 3 / sec / 1e9,
            'torch_threads': torch.get_num_threads()}


def run_all_probes(device, sustained_seconds: float = 90.0) -> dict:
    """Tier 0 in one call (~3 min at the default soak).  Returns the probe vector stored in the DB."""
    return {
        'matmul_tflops': probe_matmul_tflops(device),
        'sustained': probe_sustained_tflops(device, seconds=sustained_seconds),
        'membw': probe_memory_bandwidth(device),
        'launch_overhead_us': probe_launch_overhead_us(device),
        'h2d': probe_h2d_bandwidth(device),
        'cpu': probe_cpu(),
    }


# Workload characterization. Turns a model plus a training config into the FLOP, memory, and step
# counts the roofline needs. This half of the estimate is machine-independent, so it only has to
# be computed once per (architecture, batch) and can then be aimed at any probe vector.

def count_flops_per_image(build_model, input_shape=(3, 32, 32), bs: int = 8) -> dict:
    """Forward and forward+backward FLOPs per sample, measured with FlopCounterMode on CPU (it is
    a TorchDispatchMode: it counts what actually runs, including the backward graph).  Counting
    beats arithmetic here -- this repo's 32x32 stem surgery makes resnet18 cost 1.11 GFLOP/img,
    ~30x what naive scaling of the stock 224px model suggests.  Caveat: BN/ReLU/optimizer/
    augmentation register ~0 FLOPs but cost real bandwidth and launches, which is why the roofline
    carries memory and launch terms separately."""
    from torch.utils.flop_counter import FlopCounterMode

    # train() rather than eval(), so the counter sees the same modules the real run dispatches.
    # A small batch is enough: FLOPs scale linearly with it and we divide back out below.
    model = build_model().train()
    x = torch.randn(bs, *input_shape)

    # Forward only, under no_grad so nothing but the inference path is counted.
    with FlopCounterMode(display=False) as fwd:
        with torch.no_grad():
            model(x)

    # Forward plus backward. The sum() gives the graph a scalar to differentiate; .float() keeps
    # that reduction in fp32 so the counter isn't tripped by a dtype it can't attribute.
    with FlopCounterMode(display=False) as both:
        model(x).float().sum().backward()

    # Drop the gradients that backward pass just left behind, so the caller inherits a clean model.
    model.zero_grad(set_to_none=True)
    return {'fwd_flops_per_img': fwd.get_total_flops() / bs,
            'train_flops_per_img': both.get_total_flops() / bs,
            'params': sum(p.numel() for p in model.parameters() if p.requires_grad)}


def measure_memory_scaling(build_model, make_batch, device, bs_pair=(128, 256),
                           headroom: float = 0.15) -> dict | None:
    """Peak-VRAM linear fit mem(bs) = m0 + a*bs from two measured fwd+bwd passes, then the
    largest batch that fits with headroom for allocator fragmentation and eval.  Measured, not
    modeled -- analytical activation counts miss BN saves, autocast casts, and workspace.  CUDA
    only (returns None elsewhere).  The fit predicts the VRAM cliff: a config tuned on a 96 GB
    card can OOM on 8 GB, or silently force cudnn into slow low-workspace algorithms."""
    if device.type != 'cuda':
        return None

    model = build_model().to(device).train()

    # peaks: high-water VRAM for each batch size in bs_pair. Two points are all a straight line
    # needs, and the slope/intercept split below is what makes the fit extrapolate.
    peaks = []
    for bs in bs_pair:
        # Reset both the cache and the peak counter before each pass, or the second measurement
        # would inherit the first one's high-water mark and the slope would come out flat.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x, y = make_batch(bs)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        peaks.append(torch.cuda.max_memory_allocated(device))
        del x, y, loss

    # slope is the per-image activation cost; intercept is the fixed part (weights, gradients,
    # optimizer state) that a batch size of zero would still pay.
    slope = (peaks[1] - peaks[0]) / (bs_pair[1] - bs_pair[0])
    intercept = peaks[0] - slope * bs_pair[0]
    total = torch.cuda.get_device_properties(device).total_memory

    del model
    torch.cuda.empty_cache()
    return {'bytes_per_img': slope, 'static_bytes': intercept,
            'peaks': dict(zip(bs_pair, peaks)),
            'max_batch_estimate': int((1 - headroom) * (total - intercept) / slope) if slope > 0 else None}


def workload_spec(name: str, *, n_train: int, n_val: int, batch_size: int, epochs: int,
                  flops: dict, drop_last: bool = True, evals_per_epoch: int = 1) -> dict:
    """Everything the estimator needs to know about a training job, in one dict."""

    # steps is the multiplier every later estimate runs through, so it has to match the loader's
    # own behavior: dropping a ragged last batch shortens the epoch by one step.
    steps = n_train // batch_size if drop_last else math.ceil(n_train / batch_size)

    # Carry the inputs alongside the derived totals, so a stored record can be re-read later
    # without needing the code that produced it.
    return {
        'name': name, 'n_train': n_train, 'n_val': n_val, 'batch_size': batch_size,
        'epochs': epochs, 'drop_last': drop_last, 'evals_per_epoch': evals_per_epoch,
        'steps_per_epoch': steps,
        'params': flops['params'],
        'fwd_flops_per_img': flops['fwd_flops_per_img'],
        'train_flops_per_img': flops['train_flops_per_img'],
        'flops_per_step': flops['train_flops_per_img'] * batch_size,
        'total_train_flops': flops['train_flops_per_img'] * batch_size * steps * epochs,
    }


# Tier 1: analytical roofline estimate. Combines a probe vector with a workload spec to bracket
# the runtime before anything has been trained on the target -- the gate that catches a two-day
# configuration while it is still cheap to redesign.

def tier1_estimate(work: dict, *, p_tflops: float, membw_gbps: float | None = None,
                   launch_us: float | None = None, kernels_per_step: int | None = None,
                   n_modules: int | None = None, act_bytes_per_img: float | None = None,
                   mfu_band: tuple[float, float] = (0.05, 0.40), startup_s: float = 20.0) -> dict:
    """Design-time estimate from a probe vector alone -- no run on the target needed.

    t_step = max( flops_step / (MFU * peak),  bytes_step / (0.7 * BW),  K * launch )

    evaluated at both edges of an MFU band, because MFU is the honest unknown: this repo's own
    data spans ~30% (resnet18, compute-bound laptop) down to ~2% (same model on a huge card where
    per-step overhead dominates), and ~16% for the vit on the same card as the 30% CNN.  Use a
    DB-calibrated MFU for the same (architecture, batch) when available; the wide default band is
    for genuinely new workloads.  Purpose: the [optimistic, pessimistic] range answers the factory
    gate question -- "2 minutes or 2 days?" -- before any training run.  It does not deliver
    +/-10%; only Tier 2 calibration does."""
    if kernels_per_step is None:
        # ~1.5-2 launches per module forward + 2-3 backward + optimizer sweep; crude on purpose --
        # this term only needs to be right within ~2x to locate the launch-bound knee.
        kernels_per_step = int(n_modules * 4.5 + 30) if n_modules else None

    # The launch floor: what the step costs in dispatch alone, before any math. Zero when we have
    # no launch measurement, which just drops the term out of the max() below.
    t_launch = kernels_per_step * launch_us * 1e-6 if (kernels_per_step and launch_us) else 0.0

    t_mem = 0.0
    if membw_gbps and act_bytes_per_img:
        # act_bytes_per_img is a peak-residency slope (what measure_memory_scaling returns), not
        # traffic.  Residency and per-step traffic differ in both directions (activations are
        # touched several times, but the slope also carries gradients + workspace that aren't
        # re-read), so 1.5x is a deliberately soft factor: this term is a sanity floor, and an
        # aggressive multiplier here once "proved" a compute-bound laptop was memory-bound.
        bytes_step = 1.5 * act_bytes_per_img * work['batch_size'] + 3 * work['params'] * 2

        # 0.7 of peak bandwidth: no real access pattern reaches the copy-benchmark number.
        t_mem = bytes_step / (0.7 * membw_gbps * 1e9)

    # at: evaluate the whole roofline at one assumed MFU. Called twice, once per band edge, which
    # is what turns a single unknown into an honest [optimistic, pessimistic] range.
    def at(mfu: float) -> dict:
        t_comp = work['flops_per_step'] / (mfu * p_tflops * 1e12)
        t_step = max(t_comp, t_mem, t_launch)

        # Compare the three terms in order rather than keying a dict by their times: on a tie
        # compute should win the label, and a float-keyed lookup would collapse equal terms and
        # let the last duplicate win instead.
        if t_step == t_comp:
            binding = 'compute'
        elif t_step == t_mem:
            binding = 'memory'
        else:
            binding = 'launch'

        # Eval is forward-only over the whole validation set, and it is charged at the same MFU --
        # optimistic for it, but it is a small share of the total.
        t_eval = work['n_val'] * work['fwd_flops_per_img'] / (mfu * p_tflops * 1e12)
        total = work['epochs'] * (work['steps_per_epoch'] * t_step
                                  + work['evals_per_epoch'] * t_eval) + startup_s
        return {'mfu': mfu, 't_step_ms': t_step * 1e3, 'binding_term': binding,
                'total_s': total, 'total_human': fmt_duration(total)}

    # Sort the band so callers can pass it either way round; the higher MFU is the optimistic
    # edge. The floors and assumptions ride along so a stored estimate can be re-read critically.
    lo, hi = sorted(mfu_band)
    return {'optimistic': at(hi), 'pessimistic': at(lo),
            'floors_ms': {'launch': t_launch * 1e3, 'memory': t_mem * 1e3},
            'assumptions': {'p_tflops': p_tflops, 'membw_gbps': membw_gbps,
                            'launch_us': launch_us, 'kernels_per_step': kernels_per_step,
                            'startup_s': startup_s}}


# Tier 2: calibration on the target machine. Times the real training step under real thermal
# conditions, which is the only route to a number inside ~10%.

def tier2_calibrate(step_fn, device, *, warmup_steps: int = 30, soak_seconds: float = 90.0,
                    soak_max_seconds: float = 180.0, n_windows: int = 3, window_steps: int = 30,
                    spacer_steps: int = 0, eval_fn=None, band_hint_ratio: float | None = None) -> dict:
    """Measure the real training loop, then extrapolate_run() turns t_step into a total.

    step_fn() executes exactly one optimizer step of the actual recipe (same model, batch size,
    dtype, flags, augmentation).  Protocol and why each stage exists:
      warmup  -- cudnn.benchmark autotune + allocator growth + autocast caches (epoch 1 ran 27.6s
                 vs 15.5s steady on the laptop: extrapolating un-warmed steps overshoots ~78%).
      soak    -- keep stepping under load until the thermal envelope settles.  Skipping this on a
                 power-capped laptop measures boost clocks and underestimates the run ~15-20%
                 (this laptop: 13.0s burst epochs vs 15.5s steady, stabilizing 90-120s in).
      windows -- 3 x 30 steps, synced only at window edges (per-step sync breaks pipelining), GC
                 frozen so collection pauses don't land inside a window.  Median is the estimate;
                 min/max is the honesty band (wide on throttling laptops, ~2% on desktops).
    Returns timings plus the clock trace proving (or disproving) thermal steadiness."""

    # Clear the counters before anything is timed, or peak_vram below would report the high-water
    # mark of whichever Tier 0 probe ran last rather than this model's.
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup, timed as a whole: the surplus over steady-state steps is exactly the one-time
    # autotune cost, which the caller charges to startup instead of to every epoch.
    t_all0 = _now()
    for _ in range(warmup_steps):
        step_fn()

    _sync(device)
    warmup_s = _now() - t_all0

    # Soak adaptively in ~15 s blocks: at least soak_seconds of post-warmup load, extended until
    # two consecutive blocks agree within 2% (capped at soak_max_seconds).  A fixed 75 s put the
    # windows inside the burst-to-steady transition zone on the reference laptop (which stabilizes
    # 90-120 s in) and biased t_step ~15% low -- the exact error this tool exists to prevent.
    t_soak0 = _now()

    # block_rates: steps/second for each block, the running comparison that decides convergence.
    soak_steps, block_rates = 0, []
    while _now() - t_soak0 < soak_max_seconds:
        tb, nb = _now(), 0
        while _now() - tb < 15.0 and nb < 1000:
            step_fn()
            nb += 1

        _sync(device)
        block_rates.append(nb / (_now() - tb))
        soak_steps += nb
        converged = len(block_rates) >= 2 and abs(block_rates[-1] / block_rates[-2] - 1) < 0.02
        if _now() - t_soak0 >= soak_seconds and converged:
            break

    # windows: mean seconds per step for each measurement window. Several short windows rather
    # than one long one, so the spread between them becomes the honesty band.
    windows = []

    # Collect once now, then freeze the collector: a GC pause landing inside a window would be
    # charged to the step time and inflate that window alone.
    gc.collect()
    gc.disable()
    try:
        with _SmiPoller() as poller:
            for i in range(n_windows):
                # Untimed spacing between windows, so they sample different phases of any throttle
                # oscillation rather than three adjacent points on the same part of the cycle.
                if i and spacer_steps:
                    for _ in range(spacer_steps):
                        step_fn()

                _sync(device)
                t0 = _now()
                for _ in range(window_steps):
                    step_fn()

                _sync(device)
                windows.append((_now() - t0) / window_steps)
    finally:
        gc.enable()

    # Median for the estimate, min/max for the band: the median resists a single disturbed window,
    # while the extremes are the honest statement of how much this machine varies.
    t_med = statistics.median(windows)
    lo, hi = min(windows), max(windows)
    if band_hint_ratio and 0 < band_hint_ratio < 1:
        # A few windows spanning ~1 min cannot see a multi-minute throttle cycle.  Widen the
        # quoted band by the oscillation the 90 s Tier-0 sustained probe did observe
        # (band_hint_ratio = sustained_min/sustained_max TFLOPS), split around the median.
        spread = math.sqrt(1.0 / band_hint_ratio)
        lo, hi = min(lo, t_med / spread), max(hi, t_med * spread)

    # Report the evidence alongside the number: the soak rates and convergence flag say whether
    # the machine had settled, and the clock trace says whether it stayed settled.
    result = {
        't_step_s': t_med, 't_step_min_s': lo, 't_step_max_s': hi,
        'window_means_s': windows, 'warmup_s': warmup_s,
        'soak_block_rates': block_rates,
        'soak_converged': len(block_rates) >= 2 and abs(block_rates[-1] / block_rates[-2] - 1) < 0.02,
        'soak_steps': warmup_steps + soak_steps + n_windows * window_steps + (n_windows - 1) * spacer_steps,
        'clock_trace': poller.rows, 'backend_flags': snapshot_backend_flags(),
    }

    # Eval is timed separately, since it runs once per epoch rather than once per step.
    if eval_fn is not None:
        evals = []

        # Two passes, keeping the faster one: an eval pass is short enough that a single OS hiccup
        # would visibly skew it, and the floor is the more repeatable of the two numbers.
        for _ in range(2):
            _sync(device)
            t0 = _now()
            eval_fn()
            _sync(device)
            evals.append(_now() - t0)

        result['t_eval_s'] = min(evals)

    # Peak VRAM comes free from the counters reset at the top, and it is what tells a reader
    # whether this batch size would survive on a smaller card.
    if device.type == 'cuda':
        result['peak_vram_gb'] = torch.cuda.max_memory_allocated(device) / 2 ** 30

    return result


def extrapolate_run(work: dict, t_step_s: float, t_eval_s: float = 0.0,
                    t_startup_s: float = 0.0) -> dict:
    """Total = startup + epochs * (steps * t_step + evals * t_eval).  Training is almost perfectly
    repetitive (Habitat's premise), so once t_step is measured warm+soaked this is nearly exact;
    the startup term carries the one-time costs (data decode/upload + first-epoch autotune
    surcharge -- ~12s + ~5s on the laptop reference run)."""
    epoch_s = work['steps_per_epoch'] * t_step_s + work['evals_per_epoch'] * t_eval_s
    total = t_startup_s + work['epochs'] * epoch_s

    # Throughput is reported per epoch, not per run, so the one-time startup charge doesn't
    # quietly drag the img/s figure down.
    return {'t_step_s': t_step_s, 'epoch_s': epoch_s, 'total_s': total,
            'total_human': fmt_duration(total),
            'throughput_img_s': work['steps_per_epoch'] * work['batch_size'] / epoch_s}


def implied_mfu(work: dict, t_step_s: float, p_tflops: float) -> float:
    """Achieved model-FLOPs / probed sustained peak.  The transferable efficiency currency
    (PaLM App. B) -- but per (architecture, batch): CNN and ViT differ ~2x on identical hardware."""
    return work['flops_per_step'] / t_step_s / (p_tflops * 1e12)


def transfer_estimate(work: dict, calibrated_mfu: float, *, p_tflops_target: float,
                      launch_us_target: float | None = None, kernels_per_step: int | None = None,
                      membw_gbps_target: float | None = None, act_bytes_per_img: float | None = None,
                      n_modules: int | None = None) -> dict:
    """Predict machine B from a calibration on machine A: keep A's per-architecture MFU for the
    compute term, take memory/launch floors from B's own probes, re-run the roofline max().

    Never scale by raw FLOPS ratio alone: laptop to workstation raw scaling predicts >60x while
    reality is 6.7x, because the binding term flips from compute to per-step overhead.  When the
    binding term differs between the MFU edges here, the prediction is flagged low-confidence --
    that is the signal to spend 2.5 min calibrating on B directly."""

    # Reuse Tier 1, but with a narrow band around the measured MFU instead of the wide default:
    # +/-15% is roughly how far a known architecture's efficiency moves between similar machines.
    est = tier1_estimate(work, p_tflops=p_tflops_target, membw_gbps=membw_gbps_target,
                         launch_us=launch_us_target, kernels_per_step=kernels_per_step,
                         n_modules=n_modules, act_bytes_per_img=act_bytes_per_img,
                         mfu_band=(calibrated_mfu * 0.85, calibrated_mfu * 1.15))

    # If the target isn't compute-bound, the transferred MFU is describing the wrong constraint,
    # so say so plainly rather than quoting a number that looks as trustworthy as a calibration.
    est['regime_shift'] = est['optimistic']['binding_term'] != 'compute'
    est['confidence'] = 'low -- binding term is not compute on the target; calibrate there' \
        if est['regime_shift'] else 'medium -- compute-bound transfer, expect ~10-20% error'
    return est


# Results database. Every run is written to disk as one JSON record, so an estimate for a new
# machine can lean on what the previous ones actually measured.

# Anchored to this file rather than the working directory, so a notebook and a console run write
# to the same results/ folder no matter where they were launched from.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# Assemble one database record. Everything the estimate rested on travels together -- machine,
# probes, workload, calibration, prediction, and the real time when it is known -- because a
# prediction without its assumptions can't be audited later.
def make_record(fp: dict, probes: dict | None, work: dict, calibration: dict | None,
                prediction: dict | None, actual_total_s: float | None = None,
                notes: str = '') -> dict:
    trimmed = None
    if probes:
        # Round-trip through JSON for a deep copy, so trimming below can't mutate the caller's
        # probe dict. The per-window and clock traces are dropped: they are large, and their
        # summary statistics are already stored.
        trimmed = json.loads(json.dumps(probes))
        if trimmed.get('sustained'):
            trimmed['sustained'].pop('windows', None)
            trimmed['sustained'].pop('clock_trace', None)

    # Same reasoning for the calibration's clock trace, kept out of the record by copying every
    # other key across.
    cal = None
    if calibration:
        cal = {k: v for k, v in calibration.items() if k != 'clock_trace'}

    # schema is versioned so a later format change can be detected rather than guessed at, and the
    # timestamp is UTC so records from different machines sort correctly against each other.
    return {'schema': 1, 'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'fingerprint': fp, 'probes': trimmed, 'workload': work, 'calibration': cal,
            'prediction': prediction, 'actual_total_s': actual_total_s, 'notes': notes}


def save_record(record: dict, results_dir: str = RESULTS_DIR) -> str:
    """One file per record (not one growing file): concurrent machines never merge-conflict."""
    os.makedirs(results_dir, exist_ok=True)

    # Build the filename out of the fields you'd want to find a record by -- host, GPU, power
    # state, timestamp -- with separator characters replaced so it stays a legal path everywhere.
    fp = record['fingerprint']
    gpu = (fp.get('gpu') or fp['device_type']).replace(' ', '-').replace('/', '-')
    stamp = record['timestamp'].replace(':', '').replace('-', '')[:15]
    base = os.path.join(results_dir, f"{fp['hostname']}_{gpu}_{power_state_tag(fp)}_{stamp}")

    # The timestamp only resolves to the second, so two saves within the same second would collide;
    # add a suffix until the name is free rather than silently overwriting a record.
    path, k = base + '.json', 1
    while os.path.exists(path):
        path, k = f'{base}-{k}.json', k + 1

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)

    return path


# Read every record in the directory. A record that is unreadable or half-written is reported and
# skipped, so one bad file can't stop the rest of the database from loading.
def load_records(results_dir: str = RESULTS_DIR) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        try:
            with open(path, encoding='utf-8') as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            print(f'[perfkit] skipping unreadable record {path}')

    return records


# Misc. One shared formatting helper, kept at the bottom so it doesn't interrupt the tiers.

# Print a duration in whatever unit reads naturally at that scale, since the estimates here span
# seconds to days and a raw seconds count stops being legible somewhere in the middle.
def fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f} s'
    if seconds < 5400:
        return f'{seconds / 60:.1f} min'
    if seconds < 2 * 86400:
        return f'{seconds / 3600:.1f} h'
    return f'{seconds / 86400:.1f} days'

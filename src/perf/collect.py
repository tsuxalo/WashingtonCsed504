"""collect.py -- headless data collection for the AI-Model-Factory estimator (no Jupyter needed).

Run this on every machine configuration you want in the database, plugged in:

    conda activate uw-csed504
    python collect.py                          # resnet18, bs 512, 40 epochs: full probe + calibration
    python collect.py --model vit              # calibrate the ViT recipe instead
    python collect.py --quick                  # shortened probes/soak (~2 min): pipeline smoke test,
                                               #   numbers NOT database-grade (burst-biased)
    python collect.py --actual-seconds 638.4   # attach a measured real-run wall time to the record

Each invocation appends ONE json record to results/ (fingerprint + probes + calibration +
prediction + actual) -- commit those files; they are the estimator's training data.

The calibrated training step below is a faithful copy of ../a1-cv/cifar100_hf_train.ipynb
section 4.  If that recipe changes, change build_step() to match (and the same code in
training_time_estimator.ipynb) -- calibrating a different recipe predicts a different run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ('.', '../common', '../a1-cv', '../a1-imagenet32'):
    _p = os.path.normpath(os.path.join(_HERE, _rel))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn

import perfkit as pk


def build_step(model_name: str, bs: int, n_train: int, n_val: int, num_classes: int, device):
    """The real training step + eval on synthetic data of identical shape (augmentation is
    shape-static, so t_step matches real data within ~2% and no dataset download is needed)."""
    import models as M                     # ../a1-imagenet32
    from gpu_data import GPUImageLoader    # ../a1-cv

    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    xtr = torch.randint(0, 256, (n_train, 3, 32, 32), dtype=torch.uint8, device=device)
    ytr = torch.randint(0, num_classes, (n_train,), device=device)
    xte = torch.randint(0, 256, (n_val, 3, 32, 32), dtype=torch.uint8, device=device)
    yte = torch.randint(0, num_classes, (n_val,), device=device)

    is_vit = model_name.startswith('vit')
    train_loader = GPUImageLoader(xtr, ytr, bs, mean, std, train=True, erasing=is_vit, seed=42)
    test_loader = GPUImageLoader(xte, yte, 512, mean, std, train=False)

    model = M.build(model_name, num_classes=num_classes).to(device).train()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05) if is_vit else
                 torch.optim.SGD(model.parameters(), lr=0.1 * bs / 256, momentum=0.9,
                                 nesterov=True, weight_decay=5e-4))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type if device.type == 'cuda' else 'cpu', enabled=use_amp)

    it = iter(train_loader)

    def step_fn():
        nonlocal it
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            if is_vit:
                lam = float(np.random.beta(0.2, 0.2))
                perm = torch.randperm(x.size(0), device=x.device)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[perm])
            else:
                loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    @torch.no_grad()
    def eval_fn():                          # faithful to evaluate(): fp32, top-5, per-batch .item() syncs
        model.eval()
        c1 = c5 = n = 0
        for x, y in test_loader:
            _, pred = model(x).topk(5, dim=1)
            hits = pred.eq(y.view(-1, 1))
            c1 += hits[:, :1].any(1).sum().item()
            c5 += hits.any(1).sum().item()
            n += y.size(0)
        model.train()

    return step_fn, eval_fn


def main() -> None:
    ap = argparse.ArgumentParser(description='Collect one estimator record on this machine.')
    ap.add_argument('--model', default='resnet18',
                    help='resnet18 | resnet50 | vit | vit_base (default resnet18)')
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--n-train', type=int, default=50_000)
    ap.add_argument('--n-val', type=int, default=10_000)
    ap.add_argument('--num-classes', type=int, default=100)
    ap.add_argument('--quick', action='store_true',
                    help='short probes/soak (~2 min): smoke test only, numbers are burst-biased')
    ap.add_argument('--actual-seconds', type=float, default=None,
                    help='measured wall time of the real run (sum of epoch times), stored with the record')
    ap.add_argument('--notes', default='')
    args = ap.parse_args()

    from gpu_check import get_device, set_seed
    device = get_device()
    set_seed(42)                            # replicate the training notebook's flag state exactly
    try:
        from gpu_check import enable_fast_matmul
        enable_fast_matmul()
    except Exception:
        pass

    fp = pk.fingerprint(device)
    print(f"\n[collect] {fp['hostname']} / {fp.get('gpu', fp['device_type'])} / "
          f"power={pk.power_state_tag(fp)} / flags={fp['backend_flags']}")
    if fp.get('power_plugged') is False:
        print('[collect] *** WARNING: on BATTERY -- record will be tagged batt; '
              'plugged-in training runs will NOT match these numbers.')

    print('[collect] Tier 0 probes...')
    probes = pk.run_all_probes(device, sustained_seconds=20 if args.quick else 90)
    sust = probes['sustained']
    print(f"[collect]   burst {sust['burst_tflops']:.1f} -> sustained {sust['sustained_tflops']:.1f} TFLOPS, "
          f"membw {probes['membw']['triad_gbps'] or float('nan'):.0f} GB/s, "
          f"launch {probes['launch_overhead_us']:.1f} us")

    import models as M
    flops = pk.count_flops_per_image(lambda: M.build(args.model, num_classes=args.num_classes))
    work = pk.workload_spec(f'{args.model}-cifar100', n_train=args.n_train, n_val=args.n_val,
                            batch_size=args.batch_size, epochs=args.epochs, flops=flops)
    print(f"[collect] workload: {work['params']/1e6:.1f}M params, "
          f"{work['train_flops_per_img']/1e9:.2f} GFLOP/img trained, {work['steps_per_epoch']} steps/epoch")

    print('[collect] Tier 2 calibration (~2.5-4 min)...')
    step_fn, eval_fn = build_step(args.model, args.batch_size, args.n_train, args.n_val,
                                  args.num_classes, device)
    cal_kw = dict(warmup_steps=30, soak_seconds=90.0, soak_max_seconds=180.0,
                  n_windows=3, window_steps=30, spacer_steps=60)
    if args.quick:
        cal_kw = dict(warmup_steps=10, soak_seconds=15.0, soak_max_seconds=30.0,
                      n_windows=2, window_steps=15, spacer_steps=0)
    if device.type == 'cpu':
        cal_kw = dict(warmup_steps=2, soak_seconds=10.0, soak_max_seconds=20.0,
                      n_windows=2, window_steps=3, spacer_steps=0)
    band = (sust['sustained_min_tflops'] / sust['sustained_max_tflops']
            if sust.get('sustained_max_tflops') else None)
    cal = pk.tier2_calibrate(step_fn, device, eval_fn=eval_fn, band_hint_ratio=band, **cal_kw)

    t_startup = max(0.0, cal['warmup_s'] - cal_kw['warmup_steps'] * cal['t_step_s']) + 10.0
    pred = pk.extrapolate_run(work, cal['t_step_s'], cal.get('t_eval_s', 0.0), t_startup)
    mfu = pk.implied_mfu(work, cal['t_step_s'], sust['sustained_tflops'])
    print(f"[collect] t_step {cal['t_step_s']*1e3:.1f} ms "
          f"(band {cal['t_step_min_s']*1e3:.1f}-{cal['t_step_max_s']*1e3:.1f}, "
          f"soak {'converged' if cal.get('soak_converged') else 'hit cap'})   MFU {mfu:.0%}")
    print(f"[collect] PREDICTED {args.epochs}-epoch total: {pred['total_human']} "
          f"({pred['throughput_img_s']:,.0f} img/s)")
    if args.actual_seconds:
        err = (pred['total_s'] - t_startup - args.actual_seconds) / args.actual_seconds
        print(f"[collect] vs actual {pk.fmt_duration(args.actual_seconds)}: error {err:+.1%}")

    notes = args.notes or 'collect.py; synthetic-data calibration; recipe = cifar100_hf_train.ipynb section 4'
    if args.quick:
        notes += ' [QUICK MODE -- burst-biased, not database-grade]'
    rec = pk.make_record(fp, probes, work, cal, {**pred, 'mfu': mfu, 't_startup_s': t_startup},
                         actual_total_s=args.actual_seconds, notes=notes)
    path = pk.save_record(rec)
    print(f'[collect] saved {os.path.relpath(path, _HERE)} -- commit results/ to share it')

    print(f"\n{'machine':34s} {'power':5s} {'workload':20s} {'t_step':>9s} {'predicted':>10s} {'actual':>9s}")
    for r in pk.load_records():
        f, c, p = r['fingerprint'], r.get('calibration') or {}, r.get('prediction') or {}
        print(f"{(f.get('gpu') or f['device_type'])[:33]:34s} {pk.power_state_tag(f):5s} "
              f"{r['workload']['name'][:19]:20s} {c.get('t_step_s', float('nan'))*1e3:7.1f}ms "
              f"{p.get('total_human', '-'):>10s} "
              f"{pk.fmt_duration(r['actual_total_s']) if r.get('actual_total_s') else '-':>9s}")


if __name__ == '__main__':
    main()

"""collect.py -- headless data collection for the AI-Model-Factory estimator (no Jupyter needed).

Run this on every machine configuration you want in the database, plugged in:

    conda activate uw-csed504
    python collect.py                          # resnet18, bs 512, 40 epochs: full probe + calibration
    python collect.py --model vit              # calibrate the ViT recipe instead
    python collect.py --quick                  # shortened probes/soak (~2 min): pipeline smoke test,
                                               #   numbers are not database-grade (burst-biased)
    python collect.py --actual-seconds 638.4   # attach a measured real-run wall time to the record

Each invocation appends one json record to results/ (fingerprint + probes + calibration +
prediction + actual) -- commit those files; they are the estimator's training data.

The calibrated training step below is a faithful copy of ../cifar100_train.ipynb
section 4.  If that recipe changes, change build_step() to match (and the same code in
training_time_estimator.ipynb) -- calibrating a different recipe predicts a different run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Put the sibling a1-cv modules and ../../common on the path before importing them, so this script
# runs from any working directory rather than only from perf/.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ('.', '..', '../../common'):
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

    # The real architectures and the real GPU-resident loader, both from the a1-cv package one
    # directory up. Imported here rather than at module scope so --help works without torch
    # touching the GPU.
    import models as M
    from cifar_data import GPUImageLoader

    # Random uint8 images with CIFAR-100's own normalization constants. The pixel values are
    # meaningless, but the shapes, dtypes, and per-batch work are exactly the real run's.
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    xtr = torch.randint(0, 256, (n_train, 3, 32, 32), dtype=torch.uint8, device=device)
    ytr = torch.randint(0, num_classes, (n_train,), device=device)
    xte = torch.randint(0, 256, (n_val, 3, 32, 32), dtype=torch.uint8, device=device)
    yte = torch.randint(0, num_classes, (n_val,), device=device)

    # The recipe follows the architecture, exactly as the training notebook does: a ViT gets the
    # heavier augmentation, a CNN gets channels_last. Both choices change the timing, so both have
    # to be reproduced here or we would be calibrating a different program.
    is_vit = model_name.startswith('vit')
    channels_last = not is_vit and device.type == 'cuda'
    train_loader = GPUImageLoader(xtr, ytr, bs, mean, std, train=True, erasing=is_vit, seed=42)
    test_loader = GPUImageLoader(xte, yte, 512, mean, std, train=False)

    # Build the model and put it in train mode, since BN and dropout behave differently there and
    # the timed step must be the training-time one.
    model = M.build(model_name, num_classes=num_classes).to(device).train()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Loss, optimizer, and precision, all copied from the notebook's section 4: AdamW for the ViT,
    # SGD with an LR that scales with the batch for the CNN, both fused on CUDA.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = (torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05,
                                   fused=device.type == 'cuda') if is_vit else
                 torch.optim.SGD(model.parameters(), lr=0.1 * bs / 256, momentum=0.9,
                                 nesterov=True, weight_decay=5e-4, fused=device.type == 'cuda'))

    # The ViT trains in fp16 and so needs a GradScaler; the CNN's bf16 keeps fp32's exponent range
    # and doesn't. The scaler adds a per-step flag read, which is itself part of what we time.
    use_amp = device.type == 'cuda'
    amp_dtype = torch.float16 if is_vit else torch.bfloat16
    use_scaler = use_amp and amp_dtype is torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # it: a live iterator over the loader, held across calls so step_fn can be called far more
    # times than one epoch contains.
    it = iter(train_loader)

    # step_fn: exactly one optimizer step, which is the unit tier2_calibrate times.
    def step_fn():
        nonlocal it

        # Restart the iterator when the epoch runs out, so the calibration can keep stepping for
        # as long as the soak protocol asks.
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            x, y = next(it)

        # Match the model's memory layout; a mismatch would drop cuDNN onto a slower path and the
        # measured step would not be the one the real run takes.
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            # ViT path: mixup, so the loss is a lam-weighted blend of both source labels. It costs
            # a permutation and an extra criterion call per step, which belongs in the timing.
            if is_vit:
                lam = float(np.random.beta(0.2, 0.2))
                perm = torch.randperm(x.size(0), device=x.device)
                out = model(lam * x + (1 - lam) * x[perm])
                loss = lam * criterion(out, y) + (1 - lam) * criterion(out, y[perm])

            # CNN path: plain cross-entropy on the true label.
            else:
                loss = criterion(model(x), y)

        # fp16 path: scale the loss up so small gradients survive, unscale before clipping (the
        # norm must be measured on the true gradients), then step.
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        # bf16 path: nothing underflows, so clip and step directly.
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # eval_fn: faithful to the notebook's evaluate() -- autocast forward, hit counts accumulated
    # on the GPU, and one .item() sync at the very end. The results are discarded (the labels are
    # random); only the wall time matters, and per-batch syncs would inflate it.
    @torch.no_grad()
    def eval_fn():
        model.eval()
        c1 = torch.zeros((), device=device)
        c5 = torch.zeros((), device=device)
        n = 0
        for x, y in test_loader:
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(x)

            _, pred = out.float().topk(5, dim=1)
            hits = pred.eq(y.view(-1, 1))
            c1 += hits[:, :1].any(1).sum()
            c5 += hits.any(1).sum()
            n += y.size(0)

        # The one sync, kept so the timed region includes the read-back a real eval also pays.
        (c1 / max(n, 1)).item(); (c5 / max(n, 1)).item()

        # Back to train mode, since the calibration keeps stepping after this returns.
        model.train()

    return step_fn, eval_fn


def main() -> None:
    # The workload knobs are all arguments so one machine can contribute several records without
    # editing the file. The defaults describe the run this project actually did.
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

    # Replicate the training notebook's startup exactly: set_seed(42) then enable_fast_matmul(),
    # in that order, because the pair is what leaves cudnn and TF32 in the state the real run has.
    # Calibrating under different flags would time a different set of kernels.
    from gpu_check import get_device, set_seed
    device = get_device()
    set_seed(42)
    try:
        from gpu_check import enable_fast_matmul
        enable_fast_matmul()
    except Exception:
        pass

    # Identify the machine first and print it, so the operator can see what is about to be
    # recorded before spending four minutes measuring it.
    fp = pk.fingerprint(device)
    print(f"\n[collect] {fp['hostname']} / {fp.get('gpu', fp['device_type'])} / "
          f"power={pk.power_state_tag(fp)} / flags={fp['backend_flags']}")

    # Battery changes the power cap and the boost behavior, so a record taken here describes a
    # slower machine than the same laptop on AC. It is still saved, just tagged for what it is.
    if fp.get('power_plugged') is False:
        print('[collect] Warning: running on battery -- the record will be tagged batt; '
              'plugged-in training runs will not match these numbers.')

    # Tier 0: the machine's real ceilings. The soak is shortened under --quick, which is why quick
    # numbers are burst-biased and not worth committing.
    print('[collect] Tier 0 probes...')
    probes = pk.run_all_probes(device, sustained_seconds=20 if args.quick else 90)
    sust = probes['sustained']
    print(f"[collect]   burst {sust['burst_tflops']:.1f} then sustained {sust['sustained_tflops']:.1f} TFLOPS, "
          f"membw {probes['membw']['triad_gbps'] or float('nan'):.0f} GB/s, "
          f"launch {probes['launch_overhead_us']:.1f} us")

    # Characterize the workload: count its FLOPs, then fold them into a spec. The recipe tag goes
    # in the workload name because a record calibrates one recipe, and comparing a bf16 record to
    # an fp16 one would quietly mix two different programs.
    import models as M
    recipe_tag = 'fp16' if args.model.startswith('vit') else 'bf16cl'
    flops = pk.count_flops_per_image(lambda: M.build(args.model, num_classes=args.num_classes))
    work = pk.workload_spec(f'{args.model}-cifar100-{recipe_tag}', n_train=args.n_train,
                            n_val=args.n_val, batch_size=args.batch_size, epochs=args.epochs,
                            flops=flops)
    print(f"[collect] workload: {work['params']/1e6:.1f}M params, "
          f"{work['train_flops_per_img']/1e9:.2f} GFLOP/img trained, {work['steps_per_epoch']} steps/epoch")

    # Tier 2: time the real step on this machine.
    print('[collect] Tier 2 calibration (~2.5-4 min)...')
    step_fn, eval_fn = build_step(args.model, args.batch_size, args.n_train, args.n_val,
                                  args.num_classes, device)

    # The full protocol: warm up, soak until the clocks settle, then three spaced windows. The
    # spacers matter on a throttling laptop, where adjacent windows would sample the same phase of
    # the oscillation.
    cal_kw = dict(warmup_steps=30, soak_seconds=90.0, soak_max_seconds=180.0,
                  n_windows=3, window_steps=30, spacer_steps=60)

    # --quick trades the soak away for speed. It proves the pipeline runs; it does not produce a
    # number worth saving.
    if args.quick:
        cal_kw = dict(warmup_steps=10, soak_seconds=15.0, soak_max_seconds=30.0,
                      n_windows=2, window_steps=15, spacer_steps=0)

    # CPU is roughly 100x slower per step, so the same protocol would take hours; shrink it to
    # something that still finishes.
    if device.type == 'cpu':
        cal_kw = dict(warmup_steps=2, soak_seconds=10.0, soak_max_seconds=20.0,
                      n_windows=2, window_steps=3, spacer_steps=0)

    # Hand the calibration the oscillation the 90 s sustained probe saw, so the quoted band can be
    # widened to cover a throttle cycle longer than the measurement windows themselves.
    band = (sust['sustained_min_tflops'] / sust['sustained_max_tflops']
            if sust.get('sustained_max_tflops') else None)
    cal = pk.tier2_calibrate(step_fn, device, eval_fn=eval_fn, band_hint_ratio=band, **cal_kw)

    # Startup charge: whatever the warmup cost above steady-state steps (autotune, allocator
    # growth) plus a 10 s allowance for decoding and uploading the real dataset, which the
    # synthetic calibration never pays.
    t_startup = max(0.0, cal['warmup_s'] - cal_kw['warmup_steps'] * cal['t_step_s']) + 10.0
    pred = pk.extrapolate_run(work, cal['t_step_s'], cal.get('t_eval_s', 0.0), t_startup)
    mfu = pk.implied_mfu(work, cal['t_step_s'], sust['sustained_tflops'])
    print(f"[collect] t_step {cal['t_step_s']*1e3:.1f} ms "
          f"(band {cal['t_step_min_s']*1e3:.1f}-{cal['t_step_max_s']*1e3:.1f}, "
          f"soak {'converged' if cal.get('soak_converged') else 'hit cap'})   MFU {mfu:.0%}")
    print(f"[collect] Predicted {args.epochs}-epoch total: {pred['total_human']} "
          f"({pred['throughput_img_s']:,.0f} img/s)")

    # When a real wall time was supplied, score the prediction against it. The startup charge is
    # subtracted first, because the actuals are sums of per-epoch times and don't include it.
    if args.actual_seconds:
        err = (pred['total_s'] - t_startup - args.actual_seconds) / args.actual_seconds
        print(f"[collect] vs actual {pk.fmt_duration(args.actual_seconds)}: error {err:+.1%}")

    # Note what produced this record, so a later reader knows the calibration used synthetic data
    # and which recipe it mirrored. Quick runs say so, since their numbers are burst-biased.
    notes = args.notes or 'collect.py; synthetic-data calibration; recipe = cifar100_train.ipynb section 4'
    if args.quick:
        notes += ' [quick mode -- burst-biased, not database-grade]'

    rec = pk.make_record(fp, probes, work, cal, {**pred, 'mfu': mfu, 't_startup_s': t_startup},
                         actual_total_s=args.actual_seconds, notes=notes)
    path = pk.save_record(rec)
    print(f'[collect] saved {os.path.relpath(path, _HERE)} -- commit results/ to share it')

    # Finish by printing the whole database, so each new run is read in the context of the others
    # rather than as an isolated number.
    print(f"\n{'machine':34s} {'power':5s} {'workload':20s} {'t_step':>9s} {'predicted':>10s} {'actual':>9s}")
    for r in pk.load_records():
        f, c, p = r['fingerprint'], r.get('calibration') or {}, r.get('prediction') or {}
        print(f"{(f.get('gpu') or f['device_type'])[:33]:34s} {pk.power_state_tag(f):5s} "
              f"{r['workload']['name'][:19]:20s} {c.get('t_step_s', float('nan'))*1e3:7.1f}ms "
              f"{p.get('total_human', '-'):>10s} "
              f"{pk.fmt_duration(r['actual_total_s']) if r.get('actual_total_s') else '-':>9s}")


if __name__ == '__main__':
    main()

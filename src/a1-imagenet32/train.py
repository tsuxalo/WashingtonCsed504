"""
Per-card trainer for ResNet-18 or a ViT on ImageNet-32 (1.28M images, 1000 classes).

One process, one GPU, one model. We run this headless for hours, so it checkpoints every
epoch and can --resume from the last one (see the checkpoint step in main()). The whole
dataset lives resident in GPU memory (see data.py), which means there is no DataLoader, no
worker process, and no CPU in the inner loop -- the data is already sitting where the math
happens.

Usage and examples:
    python train.py --model resnet18 --gpu 0            # ~40 min
    python train.py --model vit      --gpu 1            # run concurrently on the other card
    python train.py --model resnet18 --smoke-test       # 30-second sanity check, exits
    python train.py --model vit --resume                # pick up from the last checkpoint

Recipes (these are the defaults wired up below):
    resnet18: SGD + momentum, LR scaled linearly with batch size from the CIFAR baseline
              (0.1 @ 128), 5-epoch warmup, cosine decay, label smoothing 0.1.
    vit:      AdamW, LR 1e-3, weight decay 0.05, 5-epoch warmup, cosine.

The ViT gets a warmup it cannot skip. Warmup is not optional for a transformer. Without it the
attention softmax saturates on the first few noisy batches and the model never climbs back out.
A CNN tolerates skipping it; a ViT does not, so we warm up both and stop worrying about which
one needed it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

import data as D
import engine as E
import models as M

OUT_DIR = os.path.join(os.path.dirname(__file__), 'runs')


def build_optimizer(model, name, batch_size, args):
    # Pick the optimizer by model family: CNNs get SGD with momentum, transformers get AdamW.
    # We key off the family with name.startswith rather than the exact name. An earlier version
    # tested `name == 'resnet18'`, so it silently handed resnet50 the transformer's AdamW recipe:
    # no error, just a CNN quietly trained on the wrong optimizer, which would have wrecked that
    # run. Matching the family covers the whole resnet* line.
    if name.startswith('resnet'):
        # The SGD recipe uses linear LR scaling, following Goyal et al.: lr = 0.1 * batch/256.
        # This one has to be right. The first version of this line said batch/128, carrying over
        # CIFAR-10's baseline (0.1 @ batch 128) instead of ImageNet's (0.1 @ batch 256). That
        # doubled the LR to 0.4, and the result was not a slightly-worse model, it was two broken
        # runs: ResNet-18's validation accuracy peaked at epoch 2 and then fell for the next seven
        # as warmup kept pushing the LR up, and ResNet-50 -- deeper, less stable -- went to
        # loss=NaN on epoch 1 and never recovered. A 2x LR error is not a tuning detail; it is the
        # difference between training and not training. The /256 is load-bearing.
        lr = args.lr if args.lr else 0.1 * batch_size / 256
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=5e-4, nesterov=True, fused=True)
    else:
        # The AdamW recipe for the transformer. weight_decay=0.05 is not comparable to the CNN's
        # 5e-4. AdamW decouples the decay from the gradient, so the same-looking knob means
        # something different here; do not read one across to the other.
        lr = args.lr if args.lr else 1e-3 * batch_size / 512
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, fused=True)
    return opt, lr


def main():
    # Step 1: Parse the CLI. Every knob has a sane default, so a bare `--model X --gpu N` just
    # works; the flags below are only for when you want to deviate from the recipe.
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(M.BUILDERS), required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=512)
    p.add_argument('--lr', type=float, default=None, help='override the scaled default')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--data-parallel', action='store_true',
                   help='split each batch across BOTH GPUs (measure before trusting it)')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-test', action='store_true', help='tiny subset, 2 epochs, then exit')
    p.add_argument('--tag', default=None, help='name for this run (default: model name)')
    p.add_argument('--strong-aug', dest='strong_aug', action='store_true', default=None,
                   help='mixup + CutMix + random erasing (default: ON for ViTs, OFF for CNNs)')
    p.add_argument('--no-strong-aug', dest='strong_aug', action='store_false')
    p.add_argument('--clip', type=float, default=1.0, help='grad-norm clip (0 to disable)')
    args = p.parse_args()


    # The key locals this function threads through, with what each one holds:
    #
    # tag:           run name; also the checkpoint/log filename stem (runs/<tag>.pt, .jsonl)
    # device:        the one cuda:N card this process owns
    # amp_dtype:     autocast dtype for the whole run, bf16 on Ampere+ else fp16
    # channels_last: True only for a CNN under bf16 (NHWC memory format)
    # epochs/subset: collapse to (2, 50_000) under --smoke-test, else (args.epochs, full set)
    # ckpt_path:     runs/<tag>.pt      (resume state: model + opt + sched + scaler + history)
    # jsonl_path:    runs/<tag>.jsonl   (one line per epoch, appended as we go)


    # Step 2: Name the run. Smoke tests get their own tag, never the real one.
    #
    # We use a separate tag on purpose. Reusing the real tag once appended 2 bogus 50k-subset
    # epochs to the front of the real run's JSONL, which would have quietly corrupted every plot
    # downstream. A distinct smoke-* stem keeps the throwaway history in its own file.
    tag = args.tag or (f'smoke-{args.model}' if args.smoke_test else args.model)


    # Step 3: Decide strong augmentation by model family when the flag was left unset.
    #
    # It defaults on for ViTs and off for CNNs. This is not favoritism, it is what the first run
    # measured. With only crop+flip the ViTs memorized the training set (97% train / 33% val)
    # while the ResNet stayed healthy (+8.7% gap). The transformer has no locality prior to
    # restrain it, so the regularization has to come from the data instead. Passing --strong-aug
    # or --no-strong-aug overrides this.
    if args.strong_aug is None:
        args.strong_aug = args.model.startswith('vit')


    # Step 4: Pin this process to its one card and switch on the fast-math paths.
    # set_device is not optional: without it torch.cuda.synchronize() and the memory stats report
    # against device 0 even though our tensors live on cuda:N. TF32 is on for matmul and cudnn,
    # benchmark is on so cudnn autotunes the (fixed) conv shapes, and seed 42 keeps runs comparable.
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)


    # Step 5: Choose the autocast dtype, bf16 where the hardware has it and fp16 as the fallback.
    #
    # We prefer bf16 over fp16 when we can. bf16 keeps fp32's exponent range, so there is no
    # overflow to babysit and no GradScaler needed. Older cards fall back to fp16 plus a scaler.
    # Paired with channels_last below for the CNNs, this measured 1.34x on a power-capped sm_89
    # laptop and ~+2% on the Blackwell workstation; the hardware decides how much it is worth.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


    # Step 6: Wire up the output paths and shrink the job under --smoke-test.
    # Everything for a run keys off `tag`: the .pt we resume from and the .jsonl we log to. A smoke
    # test collapses to 2 epochs on a 50k-image subset so the whole thing finishes in seconds.
    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f'{tag}.pt')
    jsonl_path = os.path.join(OUT_DIR, f'{tag}.jsonl')

    epochs = 2 if args.smoke_test else args.epochs
    subset = 50_000 if args.smoke_test else None


    # Step 7: Load both splits straight onto the GPU and report the resident footprint.
    # This is the one-time cost that buys us a CPU-free inner loop (see data.py): once these
    # tensors are on the card, no batch ever touches the host again.
    print(f'[{tag}] device {device} ({torch.cuda.get_device_name(device)})')
    t_load = time.time()
    train_ds = D.GpuImageNet32(device, 'train', subset=subset)
    val_ds = D.GpuImageNet32(device, 'val', subset=10_000 if args.smoke_test else None)
    print(f'[{tag}] dataset resident on GPU: {train_ds.gb():.1f} GB train + {val_ds.gb():.1f} GB val '
          f'({train_ds.n:,} + {val_ds.n:,} images) in {time.time()-t_load:.0f}s')


    # Step 8: Build the model on-device, then flip CNNs to NHWC (channels_last) where it pays off.
    model = M.build(args.model, train_ds.n_classes).to(device)
    channels_last = args.model.startswith('resnet') and amp_dtype is torch.bfloat16
    if channels_last:
        # Convert to NHWC before the DataParallel wrap and the optimizer, so both see the final
        # memory format. This is bf16 only: fp16 with channels_last drops into a pathological cuDNN
        # path (3.5x slower on sm_89). ViTs are excluded on purpose, since NHWC is a no-op and
        # really just overhead for a transformer.
        model = model.to(memory_format=torch.channels_last)
    print(f'[{tag}] {args.model}: {M.n_params(model):,} parameters')


    # Step 9: Optionally split each batch across both cards with DataParallel.
    # This is opt-in (--data-parallel) and worth measuring before you trust it: when the dataset
    # already lives on one GPU, the scatter/gather can eat the speedup. Each card sees a half-batch
    # of args.batch//2.
    if args.data_parallel:
        model = nn.DataParallel(model, device_ids=[0, 1])
        print(f'[{tag}] DataParallel across GPUs 0 and 1 '
              f'(each card sees a half-batch of {args.batch//2})')


    # Step 10: Optimizer (see build_optimizer), loss, and the fp16-only grad scaler.
    # The scaler is a no-op under bf16 (enabled= is False there); it only earns its keep on the
    # fp16 path, where it keeps small gradients from underflowing to zero.
    optimizer, lr = build_optimizer(model, args.model, args.batch, args)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)


    # Step 11: The LR schedule is a linear warmup for the first args.warmup epochs, then cosine
    # decay. SequentialLR hands off from `warm` to `cos` at the warmup milestone. T_max uses
    # max(1, ...) so a warmup that eats every epoch (a tiny smoke run) cannot hand cosine a
    # zero-length span.
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - args.warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[args.warmup])


    # Step 12: Set up epoch bookkeeping, then either start clean or --resume the last checkpoint.
    # This is the whole reason we checkpoint every epoch: the job runs headless for hours, so a
    # crash or reboot should cost one epoch, not the run. Resuming restores model + optimizer +
    # scheduler + scaler + history, so the LR curve picks up exactly where it left off.
    start_epoch, best, history = 1, 0.0, []
    if not args.resume and os.path.exists(jsonl_path):
        # A fresh run starts from a fresh history; we never append to a previous run's log.
        os.remove(jsonl_path)
    if args.resume and os.path.exists(ckpt_path):
        e, best, history = E.load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
        start_epoch = e + 1
        print(f'[{tag}] resumed from epoch {e} (best top1 {best:.2%})')


    # Step 13: Echo the resolved config so the log header records exactly what this run did.
    print(f'[{tag}] strong aug (mixup/cutmix/erasing): {args.strong_aug} | grad clip {args.clip} | '
          f'amp {str(amp_dtype).replace("torch.", "")}{" + channels_last" if channels_last else ""}')
    print(f'[{tag}] {epochs} epochs, batch {args.batch}, lr {lr:.4f} '
          f'({args.warmup}-epoch warmup -> cosine), label smoothing {args.label_smoothing}')
    print(f'[{tag}] {train_ds.n_batches(args.batch):,} batches/epoch\n', flush=True)


    # Step 14: The epoch loop -- train, evaluate, step the schedule, log, then checkpoint.
    # The per-batch work lives in engine.py; this loop just sequences an epoch and, crucially,
    # saves a checkpoint at the end of every epoch so --resume always has a fresh-as-of-last-epoch
    # state to come back to.
    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr = E.train_one_epoch(model, train_ds, optimizer, criterion, scaler, device,
                               args.batch, epoch, epochs,
                               amp_dtype=amp_dtype, channels_last=channels_last,
                               strong_aug=args.strong_aug, clip=(args.clip or None))
        va = E.evaluate(model, val_ds, criterion, device,
                        amp_dtype=amp_dtype, channels_last=channels_last)
        scheduler.step()

        is_best = va['top1'] > best
        best = max(best, va['top1'])
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        E.log_epoch(tag, epoch, epochs, tr, va, scheduler.get_last_lr()[0],
                    time.time() - t_start, device, is_best, jsonl_path)
        E.save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, best, history)


    # Step 15: Print the final best and dump a one-shot result.json summary of the whole run.
    # n_params unwraps model.module first: under DataParallel the real model is one level down, and
    # counting the wrapper would report the wrong parameter total.
    total = time.time() - t_start
    print(f'\n[{tag}] DONE. best val top1 {best:.2%} in {E.fmt_time(total)}')
    json.dump({'tag': tag, 'model': args.model, 'params': M.n_params(
                   model.module if isinstance(model, nn.DataParallel) else model),
               'epochs': epochs, 'batch': args.batch, 'lr': lr,
               'best_top1': best, 'seconds': total, 'history': history},
              open(os.path.join(OUT_DIR, f'{tag}_result.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()

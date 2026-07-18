"""
engine.py -- the training loop, metrics, and checkpointing.

train_one_epoch/evaluate are the SAME functions as cifar10_train.ipynb -- we proved they are
architecture-agnostic when the ViT trained through them with zero changes.  What is added here is
what a multi-hour run needs and a 3-minute run does not:

  * top-5 accuracy      (mandatory on 1000 classes; top-1 alone is a misleading picture)
  * throughput (img/s)  (the single best early-warning signal that the input pipeline is starving)
  * checkpoint + resume (a crash at hour 3 should cost one epoch, not three hours)
  * JSONL history       (so the analysis notebook never has to retrain anything)
  * ETA                 (so you know whether to wait up or go to bed)
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def accuracy(logits: torch.Tensor, y: torch.Tensor, topk=(1, 5)) -> list[torch.Tensor]:
    """Top-k correct COUNTS (not fractions) -- returned as GPU tensors so we never sync mid-epoch."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)              # (B, maxk)
    correct = pred.eq(y.view(-1, 1))                # (B, maxk) bool
    return [correct[:, :k].any(dim=1).sum() for k in topk]


def train_one_epoch(model, ds, optimizer, criterion, scaler, device, batch_size, epoch, epochs,
                    use_amp=True, amp_dtype=torch.float16, channels_last=False,
                    strong_aug=False, clip=None):
    model.train()
    # Accumulate on the GPU.  Calling .item() every batch would force a GPU->CPU sync each step and
    # stall the pipeline -- on CIFAR that alone cost ~7%.  We sync ONCE, at the end of the epoch.
    loss_sum = torch.zeros((), device=device)
    c1 = torch.zeros((), device=device)
    c5 = torch.zeros((), device=device)
    n = torch.zeros((), device=device)

    n_batches = ds.n_batches(batch_size)
    t0 = time.time()
    bar = tqdm(ds.epoch(batch_size, train=True), total=n_batches,
               desc=f'epoch {epoch:3d}/{epochs} train', leave=False, ncols=110)

    for step, (x, y) in enumerate(bar):
        optimizer.zero_grad(set_to_none=True)

        y_a = y_b = y
        lam = 1.0
        if strong_aug:
            import data as _D
            _D.random_erasing_(x)
            x, y_a, y_b, lam = _D.mixup_cutmix(x, y)

        if channels_last:                    # AFTER mixup/erasing: their indexing re-contiguates
            x = x.contiguous(memory_format=torch.channels_last)

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            # Mixed targets: the model is never given a confident one-hot answer, so it cannot just
            # memorize image->label.  With lam == 1.0 this collapses to the plain loss.
            loss = (lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    if lam < 1.0 else criterion(logits, y_a))
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)                       # must unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()

        b = y.size(0)
        with torch.no_grad():
            k1, k5 = accuracy(logits.detach(), y_a)   # under mixup this is approximate
            loss_sum += loss.detach() * b
            c1 += k1
            c5 += k5
            n += b

        if step % 50 == 0:      # refresh the postfix rarely: each one is a GPU->CPU sync
            done = (step + 1) * batch_size
            bar.set_postfix_str(f'loss {loss.item():.3f} top1 {(c1/n).item():.1%} '
                                f'{done/(time.time()-t0)/1000:.1f}k img/s')

    dt = time.time() - t0
    n_f = n.item()
    mean_loss = (loss_sum / n).item()

    # A diverged run is worth nothing, so do not spend hours computing it.  ResNet-50 sat at
    # loss=NaN for an hour before anyone noticed; this turns that into an immediate, loud failure.
    if mean_loss != mean_loss:      # NaN
        raise RuntimeError(
            f'loss is NaN at epoch {epoch} -- the run has diverged. '
            f'Almost always the learning rate is too high; try --lr {"{:.3g}".format(0.5 * _lr(optimizer))}')

    return {'loss': mean_loss, 'top1': (c1 / n).item(), 'top5': (c5 / n).item(),
            'sec': dt, 'img_s': n_f / dt}


def _lr(optimizer):
    return optimizer.param_groups[0]['lr']


@torch.no_grad()
def evaluate(model, ds, criterion, device, batch_size=1024, use_amp=True,
             amp_dtype=torch.float16, channels_last=False):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    c1 = torch.zeros((), device=device)
    c5 = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    for x, y in ds.epoch(batch_size, train=False):
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        k1, k5 = accuracy(logits, y)
        b = y.size(0)
        loss_sum += loss * b
        c1 += k1
        c5 += k5
        n += b
    return {'loss': (loss_sum / n).item(), 'top1': (c1 / n).item(), 'top5': (c5 / n).item()}


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history):
    tmp = path + '.tmp'
    net = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'best': best, 'history': history}, tmp)
    os.replace(tmp, path)      # atomic: a crash mid-write can never corrupt the good checkpoint


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ck = torch.load(path, map_location=device)
    net = model.module if isinstance(model, nn.DataParallel) else model
    net.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    if scaler.is_enabled() and ck.get('scaler'):
        scaler.load_state_dict(ck['scaler'])   # bf16 runs use a disabled scaler: nothing to restore
    return ck['epoch'], ck['best'], ck['history']


def fmt_time(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec//60}m{sec%60:02d}s'
    return f'{sec//3600}h{(sec%3600)//60:02d}m'


def log_epoch(tag, epoch, epochs, tr, va, lr, elapsed, device, is_best, jsonl_path):
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    remaining = (epochs - epoch) * (tr['sec'] + 3)
    star = ' *' if is_best else '  '
    print(f'[{tag}] epoch {epoch:3d}/{epochs}{star}| '
          f'train loss {tr["loss"]:.3f} top1 {tr["top1"]:6.2%} | '
          f'val top1 {va["top1"]:6.2%} top5 {va["top5"]:6.2%} | '
          f'lr {lr:.4f} | {tr["sec"]:5.1f}s {tr["img_s"]/1000:5.1f}k img/s | '
          f'mem {mem:4.1f}/{total:.0f}GB | '
          f'elapsed {fmt_time(elapsed)} ETA {fmt_time(remaining)}', flush=True)

    with open(jsonl_path, 'a') as f:
        f.write(json.dumps({'epoch': epoch, 'lr': lr, 'elapsed': elapsed, 'is_best': is_best,
                            'train': tr, 'val': va}) + '\n')

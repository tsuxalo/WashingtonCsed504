"""
train_loop.py -- the training loop, metrics, and checkpointing for a multi-hour run.

train_one_epoch and evaluate are the same functions as cifar10_train.ipynb. We proved they are
architecture-agnostic when the ViT trained through them with zero changes. What we bolt on here is
everything a multi-hour run needs and a 3-minute run does not:

  - top-5 accuracy      (mandatory on 1000 classes; top-1 alone paints a misleading picture)
  - throughput (img/s)  (the single best early warning that the input pipeline is starving)
  - checkpoint + resume (a crash at hour 3 should cost you one epoch, not three hours)
  - JSONL history       (so the analysis notebook never has to retrain anything)
  - ETA                 (so you know whether to wait up or go to bed)

One lesson dominates everything below: every .item() call is a GPU to CPU sync that stalls the
pipeline. So we accumulate loss and top-k counts in on-device tensors and pay that host sync once
per epoch, not once per batch. On CIFAR that discipline alone was worth about 7%. Watch for the
host sync notes; they mark the handful of places we deliberately reach back to the host.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


# Metrics plus the training / eval loops.

def accuracy(logits: torch.Tensor, y: torch.Tensor, topk=(1, 5)) -> list[torch.Tensor]:
    """
    Top-k correct counts (not fractions), one scalar per k, kept as GPU tensors.

    We hand back raw counts rather than fractions on purpose: the caller sums them across the whole
    epoch and divides once at the end, so nothing in here ever syncs back to the host. A row is
    "correct at k" when its true label is among that row's k highest logits.

    Variables:
     - B: batch size (rows of `logits`)
     - K: columns we score against per row, equal to maxk
     - topk: the k's we care about; (1, 5) asks for top-1 and top-5
    """
    # Step 1: Pull the maxk highest-scoring class indices for every row.
    #
    # pred: (B, K)
    #
    # top-5 already contains top-1, so there is no reason to sort all 1000 classes; maxk columns
    # is all we need. Here K is maxk.

    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)


    # Step 2: Mark where a predicted index equals the true label.
    #
    # correct: (B, K) bool
    #
    # y is (B,) reshaped to (B, 1) so it broadcasts across all K columns.

    correct = pred.eq(y.view(-1, 1))


    # Step 3: For each k, the row is a hit if any of its first k columns matched.
    #
    # .sum() collapses the batch to a scalar count, still on the GPU and still no .item().

    return [correct[:, :k].any(dim=1).sum() for k in topk]


def train_one_epoch(model, ds, optimizer, criterion, scaler, device, batch_size, epoch, epochs,
                    use_amp=True, amp_dtype=torch.float16, channels_last=False,
                    strong_aug=False, clip=None):
    model.train()

    # Step 1: Set up the epoch accumulators on the GPU.
    #
    # We want a running loss and top-k counts, but calling .item() to read them every batch would
    # force a GPU to CPU sync each step and stall the whole pipeline. On CIFAR that one habit cost
    # about 7%. So we keep everything in 0-d device tensors and pay the host sync exactly once, at
    # the end of the epoch. B is rows per batch, N is images seen so far this epoch.

    loss_sum = torch.zeros((), device=device)
    c1 = torch.zeros((), device=device)
    c5 = torch.zeros((), device=device)
    n = torch.zeros((), device=device)


    # Step 2: Start the progress bar and the throughput clock.
    #
    # n_batches only feeds tqdm's ETA; t0 is what we divide images-seen by to report img/s.

    n_batches = ds.n_batches(batch_size)
    t0 = time.time()
    bar = tqdm(ds.epoch(batch_size, train=True), total=n_batches,
               desc=f'epoch {epoch:3d}/{epochs} train', leave=False, ncols=110)

    for step, (x, y) in enumerate(bar):
        optimizer.zero_grad(set_to_none=True)

        # Step 3: Optionally apply the strong augmentation the ViTs need (erasing plus mixup/CutMix).
        #
        # The default is the plain batch, where y_a equals y_b and lam is 1.0, which makes every
        # mixed-target formula below collapse back to the ordinary single-label case.

        y_a = y_b = y
        lam = 1.0
        if strong_aug:
            import imagenet_data as _D
            _D.random_erasing_(x)
            x, y_a, y_b, lam = _D.mixup_cutmix(x, y)


        # Switch to channels_last after augmentation, never before. mixup and erasing index into x
        # and hand back a re-contiguated tensor, which would silently undo the memory format.
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)


        # Step 4: Forward pass under autocast, then the mixed-target loss.
        #
        # The model is never handed a confident one-hot target. The loss is a lam-weighted blend of
        # the two labels, so it cannot just memorize image to label. When lam is 1.0 the whole thing
        # collapses to the plain criterion(logits, y_a).

        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            # Mixed targets are never a confident one-hot answer, hence no easy memorization.
            loss = (lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                    if lam < 1.0 else criterion(logits, y_a))


        # Step 5: The standard AMP dance: scale, backward, step, update.
        #
        # If we are clipping we must unscale first, because clip_grad_norm_ reads the true gradient
        # magnitudes and those are still multiplied by the loss scale until we undo it.

        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()


        # Step 6: Fold this batch into the running totals, all on the GPU, no sync.
        #
        # Under mixup the logits are scored against y_a only, so the training top-1/top-5 you see
        # here are approximate. That is also why the train/val gap is not comparable: val sees clean
        # labels, train sees blended ones, so do not read the gap as over- or underfitting.

        b = y.size(0)
        with torch.no_grad():
            k1, k5 = accuracy(logits.detach(), y_a)
            loss_sum += loss.detach() * b
            c1 += k1
            c5 += k5
            n += b


        # Step 7: Refresh the tqdm postfix only every 50 steps.
        #
        # Each refresh reads .item() off three device tensors, which is a host sync, so doing it
        # every step would reintroduce exactly the stall Step 1 went to such trouble to avoid.

        if step % 50 == 0:
            done = (step + 1) * batch_size
            bar.set_postfix_str(f'loss {loss.item():.3f} top1 {(c1/n).item():.1%} '
                                f'{done/(time.time()-t0)/1000:.1f}k img/s')

    # Step 8: Now pay the one sync we have been saving up, and pull the epoch totals back to the host.

    dt = time.time() - t0
    n_f = n.item()
    mean_loss = (loss_sum / n).item()


    # Step 9: Fail loudly on divergence.
    #
    # A NaN run is worthless, so do not burn hours finishing it. ResNet-50 once sat at loss=NaN for
    # an hour before anyone noticed. `mean_loss != mean_loss` is the cheap NaN test, and the message
    # hands you a concrete next move (half the current LR).

    if mean_loss != mean_loss:
        raise RuntimeError(
            f'loss is NaN at epoch {epoch} -- the run has diverged. '
            f'Almost always the learning rate is too high; try --lr {"{:.3g}".format(0.5 * _lr(optimizer))}')

    return {'loss': mean_loss, 'top1': (c1 / n).item(), 'top5': (c5 / n).item(),
            'sec': dt, 'img_s': n_f / dt}


def _lr(optimizer):
    # Current learning rate, straight off the first (and only) param group. Handy for logging and
    # for the "try half of this" hint in the divergence message above.
    return optimizer.param_groups[0]['lr']


@torch.no_grad()
def evaluate(model, ds, criterion, device, batch_size=1024, use_amp=True,
             amp_dtype=torch.float16, channels_last=False):
    # Same accumulate-on-GPU, sync-once discipline as training, just minus the augmentation and the
    # optimizer. Eval runs on clean labels, so these top-1/top-5 are the numbers you actually trust.
    # Do not line them up one-to-one against the training accuracy printed above: that one is scored
    # against mixed-up targets, so the train/val "gap" is not a like-for-like measurement.
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
    # One host sync, right at the end, exactly like train_one_epoch.
    return {'loss': (loss_sum / n).item(), 'top1': (c1 / n).item(), 'top5': (c5 / n).item()}


# Checkpointing, so a crash at hour 3 costs one epoch, not the whole run.

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history):
    # Everything needed to resume the run bit-for-bit: model weights, optimizer, LR schedule, AMP
    # scaler, plus the bookkeeping (which epoch, best-so-far, full history).
    #
    # Unwrap DataParallel first. A wrapped model prefixes every key with "module.", and a later
    # single-GPU (or CPU) reload would then fail to find its weights, so we save the bare net.
    tmp = path + '.tmp'
    net = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'best': best, 'history': history}, tmp)
    # Write to a temp file, then rename. os.replace is atomic, so a crash mid-write can trash the
    # .tmp file but can never corrupt the good checkpoint we are replacing.
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # The mirror image of save_checkpoint: restore each piece into the live objects and hand back the
    # bookkeeping so the training loop can pick up exactly where it left off.
    ck = torch.load(path, map_location=device)

    # Same DataParallel unwrap as the save side. Load into the bare net so the "module." prefixes
    # line up whether or not we happen to be wrapped this time.
    net = model.module if isinstance(model, nn.DataParallel) else model
    net.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])

    # bf16 runs use a disabled GradScaler: there is no scale state to restore, and the saved dict
    # may be empty, so we only load when the scaler is actually live.
    if scaler.is_enabled() and ck.get('scaler'):
        scaler.load_state_dict(ck['scaler'])
    return ck['epoch'], ck['best'], ck['history']


# Logging, one human line plus one JSONL row per epoch; the JSONL is what the analysis notebook reads.

def fmt_time(sec: float) -> str:
    # Seconds to a compact human string: "45s", "3m07s", "2h05m". Purely cosmetic for the log line;
    # we drop to the largest sensible unit instead of always spelling out h/m/s.
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec//60}m{sec%60:02d}s'
    return f'{sec//3600}h{(sec%3600)//60:02d}m'


def log_epoch(tag, epoch, epochs, tr, va, lr, elapsed, device, is_best, jsonl_path):
    # Two outputs from one call: a human-readable console line and one machine-readable JSONL row.
    # The JSONL is our contract with the analysis notebook. It reads history off disk and never has
    # to retrain anything, so every number we might want to plot later has to land in that row.

    # Step 1: Gather the stats that are not already inside tr/va: peak VRAM this epoch, the card's
    # total for context, and a dead-simple ETA (epochs left times this epoch's seconds plus about 3s
    # of eval/checkpoint overhead). `star` flags a new best so you can eyeball progress at a glance.
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    remaining = (epochs - epoch) * (tr['sec'] + 3)
    star = ' *' if is_best else '  '

    # Step 2: The console line: train loss/top1, val top1/top5, LR, timing plus throughput, memory,
    # and elapsed/ETA. flush=True so it lands immediately underneath a long-running tqdm bar.
    print(f'[{tag}] epoch {epoch:3d}/{epochs}{star}| '
          f'train loss {tr["loss"]:.3f} top1 {tr["top1"]:6.2%} | '
          f'val top1 {va["top1"]:6.2%} top5 {va["top5"]:6.2%} | '
          f'lr {lr:.4f} | {tr["sec"]:5.1f}s {tr["img_s"]/1000:5.1f}k img/s | '
          f'mem {mem:4.1f}/{total:.0f}GB | '
          f'elapsed {fmt_time(elapsed)} ETA {fmt_time(remaining)}', flush=True)

    # Step 3: Append this epoch as one JSON object per line. Append mode with one object per line
    # means a crash can at worst lose the last row, never the whole history.
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps({'epoch': epoch, 'lr': lr, 'elapsed': elapsed, 'is_best': is_best,
                            'train': tr, 'val': va}) + '\n')

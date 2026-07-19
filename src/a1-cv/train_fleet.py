"""Keep both RTX PRO 6000s busy, the way a data center actually trains.

A single training run uses one card and leaves the other idle, which on a two-card box throws away
half the hardware we paid for. No real training cluster works that way; it keeps every accelerator
fed. This script is the small scheduler that does that for us.

We give it a queue of models to train. It keeps one run alive on each GPU, and whenever a card
finishes its run it immediately pulls the next model off the queue and starts it there. Two cards
means two runs in flight at all times, until the queue drains. Each run is just `train_run.py --gpu N`
(the per-card trainer we already have), so this file only adds the "who runs where, and what is
next" part.

Usage (pick a preset batch of models to train, or a custom one):
    python train_fleet.py --queue cifar          # train the four CIFAR models on both cards (~30 min)
    python train_fleet.py --queue imagenet       # train the four ImageNet-32 models (several hours)
    python train_fleet.py --queue cifar --smoke  # prove the wiring first (~1 min), then it exits
    python train_fleet.py --dataset cifar100 --models resnet18 vit --epochs 40   # a custom batch
    python train_fleet.py --data-parallel --models vit_base   # one model split across both cards (see below)

Watch it from two other terminals:
    python dashboard.py                       # the live dashboard: both cards, every run's curves, ETA
    nvidia-smi dmon -s u                    # the raw truth; the sm column is the compute engine

Why a fleet and not DataParallel? There are two ways to spend a second card, and they are not the
same thing.

A fleet (the normal mode) runs a different model on each card at the same time. Both cards sit
at about 90% or more, and the whole study finishes in roughly half the wall-clock, because the
ResNet and the ViT train simultaneously instead of one after the other. This is what a
hyperparameter or architecture search does, and it is the honest win on this hardware.

DataParallel (`--data-parallel`) splits each batch of one model across both cards. We measured this
at about 1.0x on these 32x32 models: the per-step scatter/gather traffic between the cards cancels
the extra compute when the model is small. It only pays off for big models and big batches, and the
"real" version (DDP) needs NCCL, which the Windows PyTorch wheels do not ship. We keep the flag so
you can measure it yourself and see the 1.0x, not so you will rely on it. The journey notebook's
DataParallel section has the side-by-side img/s.

So to keep both cards pinned near 100%, run the fleet. To learn why one-model-two-cards is not free,
run `--data-parallel` and compare the throughput.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# HERE is this folder (src/a1-cv); train_run.py, dashboard.py, runs/ and logs/ all live here.
# PY is the exact Python running this script, so the child train_run.py runs in the same conda env
# (uw-csed504). We never hardcode "python", because a bare "python" is often the wrong interpreter.
# LOGS is where each run's console output is written.
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(HERE, 'logs')

# The two preset batches you can train. Each entry is (dataset, model, epochs); epochs=None means
# "use the --epochs default". These are exactly the eight models in the study's results table, so
# running a preset reproduces that half of the numbers.
#
#   cifar     the four CIFAR models: resnet18 and vit, on both CIFAR-10 and CIFAR-100. The ViTs get
#             200 epochs, because a Vision Transformer needs a long schedule to converge; the ResNets
#             converge in a fraction of that, so they get far fewer.
#   imagenet  the four ImageNet-32 models: resnet18, resnet50, vit, and vit_base, 40 epochs each. The
#             two extra sizes (resnet50, vit_base) are controls -- they check that a win is the
#             architecture, not just "the bigger model had more parameters".
#
# Within each list the first two entries land one per card, so both GPUs start immediately.
QUEUES = {
    'cifar':    [('cifar100', 'vit', 200), ('cifar10', 'vit', 200),
                 ('cifar100', 'resnet18', 40), ('cifar10', 'resnet18', 30)],
    'imagenet': [('imagenet32', 'resnet18', None), ('imagenet32', 'vit', None),
                 ('imagenet32', 'resnet50', None), ('imagenet32', 'vit_base', None)],
}


def tag_base(dataset, model):
    """The run's name stem, matching train_run.py: the bare model on ImageNet-32, dataset-prefixed on CIFAR."""
    return model if dataset == 'imagenet32' else f'{dataset}_{model}'


def resolve_queue(args):
    """Turn the CLI options into the list of (dataset, model, epochs) specs to run, or None when the
    user has not said what to train.

    A custom --models list wins (spread over the single --dataset at --epochs); otherwise we use the
    named preset from --queue, whose specs carry their own per-model epochs.
    """
    if args.models:
        return [(args.dataset, m, None) for m in args.models]
    if args.queue:
        return list(QUEUES[args.queue])
    return None


def launch(spec, gpu, args):
    """
    Start one train_run.py run on one card and return a live handle we can poll later.

    Inputs:
     - spec: a (dataset, model, epochs) tuple; epochs=None falls back to args.epochs
     - gpu: which CUDA card to pin this run to
     - args: the parsed command-line options (epochs, smoke, data_parallel)

    Returns:
     - a dict with the Popen process, the open log file, and some bookkeeping
    """
    dataset, model, epochs = spec
    epochs = args.epochs if epochs is None else epochs
    base = tag_base(dataset, model)

    # Step 1: Build the command. It is the trainer we already have, pinned to this card and dataset.
    #
    # The -u flag makes the child's stdout unbuffered, so the dashboard sees each tqdm line
    # immediately instead of in 4 KB gulps. Buffered child output is the classic "why is my log
    # empty?" bug.
    cmd = [PY, '-u', '-W', 'ignore', os.path.join(HERE, 'train_run.py'),
           '--dataset', dataset, '--model', model, '--gpu', str(gpu), '--epochs', str(epochs)]
    if args.smoke:
        # A tiny subset, 2 epochs, then it exits: a wiring check before we spend real time.
        cmd.append('--smoke-test')

    if args.data_parallel:
        # Only used by the single-job DataParallel demo below, never by the fleet path.
        cmd.append('--data-parallel')

    # Step 2: Send this run's console output to logs/<base>.log. There are two reasons for this.
    #
    # First, dashboard.py scrapes the current epoch's tqdm bar from the tail of this file, because the
    # JSONL only updates once per finished epoch; without the log the dashboard looks frozen for
    # minutes. Second, an hours-long run's output has to survive us closing this terminal.
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f'{base}.log'), 'w', encoding='utf-8')

    # Step 3: Spawn it. Popen returns immediately, which is the whole point: we launch on both cards
    # and then supervise, rather than waiting for one to finish first.
    #
    # On Windows, CREATE_NO_WINDOW stops a console window from popping up for each child.
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, creationflags=flags)

    # We flush every status line. An hours-long run is usually redirected to a file, and a buffered
    # print would leave that file empty for minutes, so you would have no idea the fleet was alive.
    label = 'SMOKE' if args.smoke else f'{epochs}ep'
    print(f'  [gpu {gpu}] start  {base:18s} {label:>6s}  (pid {proc.pid})  logging to logs/{base}.log', flush=True)
    return {'base': base, 'gpu': gpu, 'proc': proc, 'log': log, 't0': time.time()}


def run_fleet(args):
    """
    The scheduler: keep every card busy until the queue is empty.

    Inputs:
     - args: the parsed command-line options (queue/models, n_gpu, epochs, smoke)
    """
    queue = resolve_queue(args)
    names = [tag_base(d, m) for d, m, _ in queue]
    print(f'\nFleet: {len(queue)} runs {names} across {args.n_gpu} cards'
          f'{" (SMOKE)" if args.smoke else ""}\n')

    # slots is a list of length n_gpu. slots[g] is the run currently on card g, or None if that card
    # is idle.
    slots = [None] * args.n_gpu

    # Step 1: Prime the pump. Put the first job on each card so both start immediately.
    for g in range(args.n_gpu):
        if queue:
            slots[g] = launch(queue.pop(0), g, args)

    print('\n  both cards are now training. Watch:  python dashboard.py   (and Task Manager: '
          'set a GPU graph to "Compute_0"/"Cuda" for both cards)\n')

    # Step 2: The supervision loop. Poll each card about once a second, and when a run finishes,
    # start the next queued model on the same card. This is what keeps utilization pinned instead of
    # draining down to one card as runs finish at different speeds. The ViT takes about twice as long
    # as the ResNet, so they never finish together, and the faster card must not sit idle waiting.
    try:
        while any(slots) or queue:
            for g in range(args.n_gpu):
                job = slots[g]
                if job is None:
                    continue

                # poll() returns None while the run is still going; an int is its exit code.
                ret = job['proc'].poll()
                if ret is None:
                    continue

                # This card just freed up.
                job['log'].close()
                dt = time.time() - job['t0']

                # A nonzero exit code is a crash, not a clean finish, so the log is where to look.
                ok = 'done ' if ret == 0 else f'FAILED({ret})'
                print(f'  [gpu {g}] {ok} {job["base"]:18s}  in {dt/60:.1f} min', flush=True)
                slots[g] = launch(queue.pop(0), g, args) if queue else None

            # Poll cadence: cheap, and once a second is plenty for hour-long runs.
            time.sleep(1.0)
    except KeyboardInterrupt:
        # Step 3: Ctrl+C tears the whole fleet down cleanly. Without this, killing the scheduler would
        # orphan the training processes: they would keep both cards pinned with no easy way to see or
        # stop them short of Task Manager.
        print('\nCtrl+C received, stopping all runs...')
        for job in slots:
            if job:
                job['proc'].terminate()
                job['log'].close()

    print('\nFleet done. Results in runs/*_result.json ; curves in runs/*.jsonl ; '
          'analyze in the journey notebook.')


def run_data_parallel_demo(args):
    """
    The honest one-model-two-cards path. Runs a single train_run.py with --data-parallel so you can
    watch both cards light up in Task Manager, and then compare its img/s to the same model on one
    card (the fleet, or the notebook's DataParallel section) to see the roughly 1.0x for yourself.

    Inputs:
     - args: the parsed command-line options; the first spec of the resolved queue is the model to split
    """
    spec = resolve_queue(args)[0]
    dataset, model, _ = spec
    print(f'\nDataParallel demo: one {model} ({dataset}) split across cards 0 and 1.')
    print('Note: measured ~1.0x on these 32x32 models -- the point is to see why, not to go faster.\n')

    # --data-parallel makes train_run.py drive both cards itself, so we launch it on card 0 only.
    job = launch(spec, 0, args)
    try:
        job['proc'].wait()
    except KeyboardInterrupt:
        job['proc'].terminate()
    job['log'].close()


def main():
    p = argparse.ArgumentParser(
        description='Train a batch of models across both GPUs, one model per card, from a queue.')
    p.add_argument('--queue', choices=list(QUEUES),
                   help="which preset batch to train: 'cifar' (the four CIFAR models, ~30 min on two "
                        "GPUs) or 'imagenet' (the four ImageNet-32 models, several hours)")
    p.add_argument('--models', nargs='+', default=None,
                   help='train a custom list of models instead of a preset (uses --dataset and --epochs)')
    p.add_argument('--dataset', default='imagenet32',
                   help='dataset for a custom --models list (a preset --queue carries its own datasets)')
    p.add_argument('--epochs', type=int, default=40,
                   help='epochs per run when a spec sets none of its own (the imagenet preset uses this)')
    p.add_argument('--n-gpu', type=int, default=2, help='how many GPUs to spread across')
    p.add_argument('--smoke', action='store_true',
                   help='run each job as a quick --smoke-test to prove the wiring, then exit')
    p.add_argument('--data-parallel', action='store_true',
                   help='instead of a fleet, run one model split across both cards (the honest ~1x demo)')
    args = p.parse_args()

    if not args.queue and not args.models:
        p.error('nothing to train -- choose a preset (--queue cifar or --queue imagenet) or a custom '
                'batch (--models resnet18 vit --dataset cifar100).')

    if args.data_parallel:
        run_data_parallel_demo(args)
    else:
        run_fleet(args)


if __name__ == '__main__':
    main()

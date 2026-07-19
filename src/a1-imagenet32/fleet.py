"""Keep both RTX PRO 6000s busy, the way a data center actually trains.

A single training run uses one card and leaves the other idle, which on a two-card box throws away
half the hardware we paid for. No real training cluster works that way; it keeps every accelerator
fed. This script is the small scheduler that does that for us.

We give it a queue of models to train. It keeps one run alive on each GPU, and whenever a card
finishes its run it immediately pulls the next model off the queue and starts it there. Two cards
means two runs in flight at all times, until the queue drains. Each run is just `train.py --gpu N`
(the per-card trainer we already have), so this file only adds the "who runs where, and what is
next" part.

Usage:
    python fleet.py                         # the capstone queue on both cards (this is the 8-hour run)
    python fleet.py --smoke                 # same wiring, but --smoke-test each job (~30s) to prove it
    python fleet.py --models resnet18 vit   # a custom queue
    python fleet.py --epochs 40             # override the schedule length
    python fleet.py --data-parallel --models vit_base   # one model split across both cards (see below)

Watch it from two other terminals:
    python monitor.py                       # the live dashboard: both cards, every run's curves, ETA
    nvidia-smi dmon -s u                    # the raw truth; the sm column is the compute engine

Why a fleet and not DataParallel? There are two ways to spend a second card, and they are not the
same thing.

A fleet (this file's default) runs a different model on each card at the same time. Both cards sit
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

# HERE is this folder (src/a1-imagenet32); train.py, monitor.py, runs/ and logs/ all live here.
# PY is the exact Python running this script, so the child train.py runs in the same conda env
# (uw-csed504). We never hardcode "python", because a bare "python" is often the wrong interpreter.
# LOGS is where each run's console output is written.
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(HERE, 'logs')

# The capstone queue. ResNet-18 and the ViT are the head-to-head crossover pair; resnet50 and
# vit_base are the capacity controls, so a win cannot be dismissed as "the bigger model just had
# more parameters". The order starts the two families first, so one lands on each card and the
# dashboard immediately shows a CNN and a Transformer training side by side.
DEFAULT_QUEUE = ['resnet18', 'vit', 'resnet50', 'vit_base']


def launch(model, gpu, args):
    """
    Start one train.py run on one card and return a live handle we can poll later.

    Inputs:
     - model: the model name to train (e.g. 'resnet18')
     - gpu: which CUDA card to pin this run to
     - args: the parsed command-line options (epochs, smoke, data_parallel)

    Returns:
     - a dict with the Popen process, the open log file, and some bookkeeping
    """
    # Step 1: Build the command. It is the trainer we already have, pinned to this card.
    #
    # The -u flag makes the child's stdout unbuffered, so the dashboard sees each tqdm line
    # immediately instead of in 4 KB gulps. Buffered child output is the classic "why is my log
    # empty?" bug.
    cmd = [PY, '-u', '-W', 'ignore', os.path.join(HERE, 'train.py'),
           '--model', model, '--gpu', str(gpu), '--epochs', str(args.epochs)]
    if args.smoke:
        cmd.append('--smoke-test')                 # Tiny subset, 2 epochs, then it exits: a wiring check.
    if args.data_parallel:
        cmd.append('--data-parallel')              # Only used by the single-job DataParallel demo below.

    # Step 2: Send this run's console output to logs/<model>.log. There are two reasons for this.
    #
    # First, monitor.py scrapes the current epoch's tqdm bar from the tail of this file, because the
    # JSONL only updates once per finished epoch; without the log the dashboard looks frozen for
    # minutes. Second, an 8-hour run's output has to survive us closing this terminal.
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f'{model}.log'), 'w', encoding='utf-8')

    # Step 3: Spawn it. Popen returns immediately, which is the whole point: we launch on both cards
    # and then supervise, rather than waiting for one to finish first.
    #
    # On Windows, CREATE_NO_WINDOW stops a console window from popping up for each child.
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE, creationflags=flags)
    # We flush every status line. An 8-hour run is usually redirected to a file, and a buffered print
    # would leave that file empty for minutes, so you would have no idea the fleet was alive.
    print(f'  [gpu {gpu}] start  {model:9s}  (pid {proc.pid})  -> logs/{model}.log', flush=True)
    return {'model': model, 'gpu': gpu, 'proc': proc, 'log': log, 't0': time.time()}


def run_fleet(args):
    """
    The scheduler: keep every card busy until the queue is empty.

    Inputs:
     - args: the parsed command-line options (models, n_gpu, epochs, smoke)
    """
    queue = list(args.models)
    print(f'\nFleet: {len(queue)} runs {queue} across {args.n_gpu} cards '
          f'({"SMOKE" if args.smoke else str(args.epochs) + " epochs"} each)\n')

    # slots is a list of length n_gpu. slots[g] is the run currently on card g, or None if that card
    # is idle.
    slots = [None] * args.n_gpu

    # Step 1: Prime the pump. Put the first job on each card so both start immediately.
    for g in range(args.n_gpu):
        if queue:
            slots[g] = launch(queue.pop(0), g, args)

    print('\n  -> both cards are now training. Watch:  python monitor.py   (and Task Manager: '
          'set a GPU graph to "Compute_0"/"Cuda" for BOTH cards)\n')

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
                ret = job['proc'].poll()           # None means still running; an int means it exited.
                if ret is None:
                    continue

                # This card just freed up.
                job['log'].close()
                dt = time.time() - job['t0']
                ok = 'done ' if ret == 0 else f'FAILED({ret})'   # A nonzero exit is a crash; check the log.
                print(f'  [gpu {g}] {ok} {job["model"]:9s}  in {dt/60:.1f} min', flush=True)
                slots[g] = launch(queue.pop(0), g, args) if queue else None
            time.sleep(1.0)                          # Poll cadence: cheap, and 1s is plenty for hour-long runs.
    except KeyboardInterrupt:
        # Step 3: Ctrl+C tears the whole fleet down cleanly. Without this, killing the scheduler would
        # orphan the training processes: they would keep both cards pinned with no easy way to see or
        # stop them short of Task Manager.
        print('\nCtrl+C -> stopping all runs...')
        for job in slots:
            if job:
                job['proc'].terminate()
                job['log'].close()

    print('\nFleet done. Results in runs/*_result.json ; curves in runs/*.jsonl ; '
          'analyze in the journey notebook.')


def run_data_parallel_demo(args):
    """
    The honest one-model-two-cards path. Runs a single train.py with --data-parallel so you can
    watch both cards light up in Task Manager, and then compare its img/s to the same model on one
    card (the fleet, or the notebook's DataParallel section) to see the roughly 1.0x for yourself.

    Inputs:
     - args: the parsed command-line options; args.models[0] is the model to split
    """
    model = args.models[0]
    print(f'\nDataParallel demo: ONE {model} split across cards 0 and 1.')
    print('CAVEAT: measured ~1.0x on these 32x32 models -- the point is to SEE why, not to go faster.\n')
    job = launch(model, 0, args)                    # --data-parallel makes train.py use both cards itself.
    try:
        job['proc'].wait()
    except KeyboardInterrupt:
        job['proc'].terminate()
    job['log'].close()


def main():
    p = argparse.ArgumentParser(description='Fill both GPUs with training runs (the model-factory way).')
    p.add_argument('--models', nargs='+', default=DEFAULT_QUEUE, help='the queue of models to train')
    p.add_argument('--epochs', type=int, default=40, help='epochs per run (ignored under --smoke)')
    p.add_argument('--n-gpu', type=int, default=2, help='cards to spread across')
    p.add_argument('--smoke', action='store_true', help='--smoke-test each job: prove the wiring in ~30s')
    p.add_argument('--data-parallel', action='store_true',
                   help='instead of a fleet, run ONE model split across both cards (the honest ~1x demo)')
    args = p.parse_args()

    if args.data_parallel:
        run_data_parallel_demo(args)
    else:
        run_fleet(args)


if __name__ == '__main__':
    main()

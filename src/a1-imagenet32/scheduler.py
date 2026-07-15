"""
scheduler.py -- keep both GPUs busy with a queue of runs, unattended.

Each GPU gets a list of jobs.  When the GPU has no train.py running on it, the next job starts.
Nothing is killed and nothing is preempted, so it is safe to start this while runs are in flight --
it simply waits for the current job on each card to exit before launching the next.

    python scheduler.py            # run until the queues are empty
    python scheduler.py --plan     # print what it WOULD run, then exit

WHY THESE JOBS -- the experiment we are actually running
--------------------------------------------------------
The headline so far is that a parameter-matched ViT (42.76%) beat ResNet-18 (41.54%) on ImageNet-32.
But that ViT was given 60 epochs AND mixup/CutMix, while the ResNet got 40 epochs and neither.  Two
confounds.  A result with two confounds is not a result.  So:

    resnet18_aug60  : ResNet-18, 60 epochs, mixup     <- the DECISIVE control: matches the ViT on
                                                         BOTH axes.  If this beats 42.76%, the
                                                         "transformer wins" claim is dead.
    resnet18_60ep   : ResNet-18, 60 epochs, no mixup  <- isolates the EPOCH advantage alone
    vit_40ep        : ViT, 40 epochs, mixup           <- isolates it from the other side

Together with the two finished runs, that is a clean factorial over (architecture x epochs x mixup),
and it can tell us WHICH of the three actually explains the win.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\truer\.conda\envs\uw-csed504\python.exe'

# (gpu, args, tag) -- run in order, one at a time per GPU.
JOBS = {
    0: [
        (['--model', 'vit', '--epochs', '40', '--tag', 'vit_40ep'], 'vit_40ep'),
    ],
    1: [
        (['--model', 'resnet18', '--epochs', '60', '--strong-aug', '--tag', 'resnet18_aug60'],
         'resnet18_aug60'),
        (['--model', 'resnet18', '--epochs', '60', '--tag', 'resnet18_60ep'], 'resnet18_60ep'),
    ],
}


def running_on(gpu: int) -> str | None:
    """Return the tag/model of the train.py currently on this GPU, or None."""
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train.py' } | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return '?'          # if we cannot tell, assume busy -- never double-book a GPU
    for line in out.splitlines():
        m = re.search(r'--gpu\s+(\d)', line)
        if m and int(m.group(1)) == gpu:
            t = re.search(r'--model\s+(\S+)', line)
            return t.group(1) if t else 'unknown'
    return None


def launch(gpu: int, args: list[str], tag: str):
    log = os.path.join(HERE, 'logs', f'{tag}.log')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [PY, '-W', 'ignore', os.path.join(HERE, 'train.py'), *args, '--gpu', str(gpu),
           '--batch', '512']
    print(f'[sched] gpu{gpu}: launching {tag}  ->  {log}', flush=True)
    with open(log, 'w') as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()

    if a.plan:
        for gpu, jobs in JOBS.items():
            cur = running_on(gpu)
            print(f'gpu{gpu}: currently running {cur or "(idle)"}')
            for args, tag in jobs:
                print(f'   then: {tag:16s} {" ".join(args)}')
        return

    queues = {g: list(j) for g, j in JOBS.items()}
    print(f'[sched] started; {sum(len(q) for q in queues.values())} job(s) queued', flush=True)
    while any(queues.values()):
        for gpu, q in queues.items():
            if not q:
                continue
            cur = running_on(gpu)
            if cur is None:                      # GPU is free -> start the next job
                args, tag = q.pop(0)
                launch(gpu, args, tag)
                time.sleep(30)                   # let it claim the GPU before we poll again
        time.sleep(60)
    print('[sched] all queued jobs launched; exiting', flush=True)


if __name__ == '__main__':
    sys.exit(main())

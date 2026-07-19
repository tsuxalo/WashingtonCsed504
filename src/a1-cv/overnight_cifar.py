"""Retired. CIFAR training moved off the notebooks and onto the console.

This script used to run the CIFAR notebooks headless (nbconvert, MODE=perf) after the ImageNet-32
fleet finished, because the full CIFAR training lived in the notebooks. It no longer does:
train_run.py and train_fleet.py now train CIFAR directly -- same GPU-resident pipeline, same
dashboard, same per-epoch checkpoints as every other run -- so the notebooks stay fast-only and the
expensive training is never conflated with them.

Use the console instead:

    python train_fleet.py --queue cifar       # CIFAR-10 + CIFAR-100 to convergence, on both cards
    python dashboard.py                        # watch it live

The per-model schedules (the ViTs get the long run they need, the ResNets converge fast) live in the
'cifar' queue in train_fleet.py.
"""
import sys

# Kept as a stub so anyone who runs it out of habit gets the note above on stderr and a nonzero exit,
# instead of silence -- a stale reference to it in some pipeline then fails loudly rather than quietly
# doing nothing.
if __name__ == '__main__':
    print(__doc__, file=sys.stderr)
    sys.exit(1)

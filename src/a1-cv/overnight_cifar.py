"""Train the full (perf) CIFAR runs once the ImageNet-32 fleet has finished.

The fleet (train_fleet.py / train_run.py) has both GPUs the whole time it runs, and the full CIFAR
training lives in the notebooks (MODE=perf), so we cannot do both at once without them fighting for
the cards. This script waits for the fleet to finish, then runs the CIFAR perf notebooks on the freed
GPUs -- so all three datasets get their full-epoch results overnight, unattended.

    python overnight_cifar.py          # waits for train_run.py to be gone, then trains CIFAR (perf)

It runs cifar100_train twice (resnet18 and vit -- that is the notebook's whole "second data source"
comparison) and cifar10_train once (it does the CNN and both ViTs itself). Each executes with
MODE=perf and is written out with its results and a per-run log in overnight_cifar.log.
"""
import os
import subprocess
import sys
import time

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'overnight_cifar.log')


def say(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def fleet_alive():
    """True while any train_run.py (a fleet job) is still running. We ask Windows for python.exe
    processes whose command line mentions train_run.py, the same signal the dashboard uses."""
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train_run.py' }).Count"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(out or 0) > 0
    except Exception:
        return False


def run_nb(nb, out_name, extra_env):
    """Execute one notebook headless in MODE=perf, writing the run (with outputs) to out_name."""
    env = dict(os.environ, MODE='perf', **extra_env)
    say(f'START {out_name}  (MODE=perf {extra_env})')
    t = time.time()
    r = subprocess.run(
        [PY, '-m', 'nbconvert', '--to', 'notebook', '--execute',
         '--ExecutePreprocessor.timeout=10800', '--ExecutePreprocessor.kernel_name=python3',
         '--output', out_name, nb],
        cwd=HERE, env=env)
    say(f'DONE  {out_name}  exit={r.returncode}  in {(time.time()-t)/60:.0f} min')


def main():
    say('waiting for the ImageNet-32 fleet (train_run.py) to finish...')
    # Require several consecutive clear checks: the fleet is briefly down to one job between models,
    # so a single empty poll does not mean it is done.
    clear = 0
    while clear < 4:
        clear = clear + 1 if not fleet_alive() else 0
        time.sleep(30)
    say('fleet done -- both cards free. Starting CIFAR perf training.')

    run_nb('cifar100_train.ipynb', 'cifar100_train_resnet18.ipynb', {'MODEL': 'resnet18'})
    run_nb('cifar100_train.ipynb', 'cifar100_train_vit.ipynb',      {'MODEL': 'vit'})
    run_nb('cifar10_train.ipynb',  'cifar10_train.ipynb',           {})

    say('ALL CIFAR perf runs finished. Full results are in the notebooks above.')


if __name__ == '__main__':
    main()

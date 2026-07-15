"""
monitor.py -- a live dashboard for the ImageNet-32 training fleet.

Four training processes each printing their own tqdm bar into their own log file is unreadable.
This reads what they WRITE (runs/*.jsonl + the tqdm line at the tail of logs/*.log) plus nvidia-smi,
and renders one screen that answers the only questions you actually have:

    Is everything still alive?   Is it learning?   Is the GPU busy?   When will it be done?

It is READ-ONLY -- it never touches the training processes, so you can start and stop it freely,
run several copies, or close it mid-run without affecting anything.

    python monitor.py                 # refresh every 2s until Ctrl+C
    python monitor.py --once          # print one snapshot and exit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# A Windows console defaults to cp1252, which cannot encode a block character or a sparkline -- rich
# then dies with UnicodeEncodeError halfway through drawing the frame.  Force the stream to UTF-8;
# if that is refused (a genuinely legacy console), fall back to ASCII glyphs rather than crash.
UNICODE = True
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    UNICODE = False

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, LOGS = os.path.join(HERE, 'runs'), os.path.join(HERE, 'logs')

# Same palette as the notebook's scoreboard: colorblind-safe, and one hue per model.
COLOR = {'resnet18': 'dark_orange3', 'resnet50': 'orange1',
         'vit': 'spring_green4', 'vit_base': 'medium_purple3'}
SPARK = ' ▁▂▃▄▅▆▇█' if UNICODE else ' .:-=+*#%'
FULL, EMPTY = ('█', '░') if UNICODE else ('#', '.')


def sparkline(vals, width=28) -> str:
    """Unicode sparkline of the val-top1 curve -- the shape of learning, at a glance."""
    if not vals:
        return ''
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return SPARK[1] * len(v)
    return ''.join(SPARK[min(8, int((x - lo) / (hi - lo) * 8) + 1)] for x in v)


def read_jsonl(tag):
    p = os.path.join(RUNS, f'{tag}.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # a partially-flushed final line: ignore it, it'll be there next tick
    return out


def read_progress(tag):
    """Scrape the CURRENT epoch's progress out of the live tqdm bar at the tail of the log.

    The JSONL only gains a row when an epoch FINISHES, so without this the dashboard would look
    frozen for a whole minute at a time.
    """
    p = os.path.join(LOGS, f'{tag}.log')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:                       # tqdm rewrites its line with \r, so read raw
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        tail = f.read().decode('utf-8', 'replace')
    frames = tail.replace('\r', '\n').split('\n')
    for line in reversed(frames):
        m = re.search(r'(\d+)/(\d+)\s*\[', line)   # e.g. "1531/2503 ["
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            ips = re.search(r'([\d.]+)k img/s', line)
            return {'cur': cur, 'tot': tot, 'frac': cur / max(1, tot),
                    'img_s': float(ips.group(1)) * 1000 if ips else None}
    return None


def gpus():
    try:
        q = ('index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,'
             'temperature.gpu')
        out = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        rows = []
        for line in out.splitlines():
            i, u, mu, mt, pd, pl, t = [x.strip() for x in line.split(',')]
            rows.append({'i': int(i), 'util': float(u), 'mem': float(mu) / 1024,
                         'mem_tot': float(mt) / 1024, 'pw': float(pd), 'pw_max': float(pl),
                         'temp': float(t)})
        return rows
    except Exception:
        return []


def live_tags():
    """Which models have a train.py process actually running right now?"""
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train.py' } | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
        return {m.group(1) for m in re.finditer(r'--model\s+(\S+)', out)}
    except Exception:
        return set()


def bar(frac, width=18, color='white'):
    n = int(max(0.0, min(1.0, frac)) * width)
    return Text(FULL * n + EMPTY * (width - n), style=color)


def render(t0):
    """Vertical, one CARD per model.

    The first version was one wide table -- which needed ~135 columns and truncated every column to
    'val to...' / 'traini...' in a normal terminal.  A dashboard you cannot read is worse than no
    dashboard.  Stacked multi-line cards fit ~90 columns and never truncate.
    """
    alive = live_tags()
    tags = sorted({os.path.basename(p).split('.')[0] for p in glob.glob(os.path.join(RUNS, '*.jsonl'))}
                  | alive)

    blocks = []

    # ---- hardware ----
    g = Table.grid(padding=(0, 2))
    for d in gpus():
        us = 'bold green' if d['util'] > 85 else ('yellow' if d['util'] > 40 else 'red')
        hot = 'bold red' if d['temp'] >= 90 else ('yellow' if d['temp'] >= 87 else 'green')
        g.add_row(Text(f"cuda:{d['i']}", style='bold cyan'),
                  bar(d['util'] / 100, 12, us) + Text(f" {d['util']:3.0f}%", style=us),
                  Text(f"{d['mem']:5.1f}/{d['mem_tot']:.0f}GB"),
                  Text(f"{d['pw']:3.0f}/{d['pw_max']:.0f}W"),
                  Text(f"{d['temp']:.0f}C", style=hot))
    blocks.append(Panel(g, title='[bold]hardware', border_style='grey37', padding=(0, 1)))

    for tag in tags:
        rows = read_jsonl(tag)
        prog = read_progress(tag)
        col = COLOR.get(tag, 'white')
        running = tag in alive
        total = _total_epochs(tag)

        if not rows:
            blocks.append(Panel(Text('starting...', style='yellow'),
                                title=f'[bold {col}]{tag}', border_style='grey37', padding=(0, 1)))
            continue

        last = rows[-1]
        top1s = [r['val']['top1'] for r in rows]
        best = max(top1s)
        best_ep = top1s.index(best) + 1
        ep = last['epoch']
        tr1, va1 = last['train']['top1'], last['val']['top1']
        gap = tr1 - va1

        if not running:
            state = Text('DONE', style='bold green') if ep >= total else Text('STOPPED', style='bold red')
        elif len(top1s) >= 3 and top1s[-1] < top1s[-3] - 0.02:
            state = Text('REGRESSING', style='bold yellow')
        else:
            state = Text('training', style='green')

        # The overfitting verdict -- but ONLY when it is a valid comparison.  Under mixup/CutMix the
        # train accuracy is computed on blended images against the dominant label, so it is a
        # different (harder) exam than clean validation and the gap is not interpretable; it can even
        # go negative.  Judging it anyway is exactly the mistake we made reading the CIFAR curves.
        aug = _strong_aug(tag)
        if aug:
            health = Text('gap n/a (mixup)', style='dim')
        elif gap > 0.35:
            health = Text(f'gap {gap:+.0%} MEMORIZING', style='bold red')
        elif gap > 0.15:
            health = Text(f'gap {gap:+.0%} overfitting', style='yellow')
        else:
            health = Text(f'gap {gap:+.0%} healthy', style='green')

        frac = (ep + (prog['frac'] if prog else 0)) / max(1, total)
        ips = (prog or {}).get('img_s') or last['train']['img_s']
        recent = [r['train']['sec'] for r in rows[-5:]]
        per_ep = sorted(recent)[len(recent) // 2] + 4
        eta = _fmt((total - ep) * per_ep) if running else '-'

        t = Table.grid(padding=(0, 1))
        t.add_row(bar(frac, 30, col),
                  Text(f'{ep}/{total}', style='bold'),
                  state,
                  Text(f'{ips/1000:.1f}k img/s', style='dim'),
                  Text(f'ETA {eta}', style='dim'))
        t.add_row(Text(f'val  top1 {va1:6.2%}   top5 {last["val"]["top5"]:6.2%}', style=col),
                  Text(f'best {best:.2%} @ep{best_ep}', style='bold'),
                  Text(f'train{"(aug)" if aug else ""} {tr1:.2%}', style='dim'),
                  health)
        t.add_row(Text(sparkline(top1s, 60), style=col))
        blocks.append(Panel(t, title=f'[bold {col}]{tag}', border_style='grey37', padding=(0, 1)))

    blocks.append(Text('  ref: WRN-28-10 = 59.0% top-1 / 81.1% top-5 (Chrabaszcz 2017)  |  '
                       f'elapsed {_fmt(time.time()-t0)}  |  read-only, Ctrl+C safe', style='dim'))
    return Group(*blocks)


def _strong_aug(tag):
    """Did this run use mixup/CutMix?  If so its TRAIN accuracy is measured on blended images and is
    NOT comparable to clean validation accuracy -- the train/val 'gap' becomes meaningless (it even
    goes negative).  Same trap as the CIFAR notebook: augmented-train vs clean-test are two different
    exams.  So we label it instead of pretending to judge it."""
    p = os.path.join(LOGS, f'{tag}.log')
    try:
        with open(p, errors='replace') as f:
            for line in f:
                if 'strong aug' in line:
                    return 'True' in line
    except OSError:
        pass
    return False


def _total_epochs(tag):
    """Read the real epoch count from the run's own log header instead of hardcoding it.

    The hardcoded table said vit_base ran 60 epochs when it actually ran 40, so a COMPLETED run was
    displayed as 'STOPPED' in red.  Never hardcode what the process already tells you.
    """
    p = os.path.join(LOGS, f'{tag}.log')
    try:
        with open(p, errors='replace') as f:
            for line in f:
                m = re.search(r'(\d+) epochs, batch', line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return 40


def _fmt(s):
    s = int(s)
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s//60}m'
    return f'{s//3600}h{(s%3600)//60:02d}m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=2.0)
    a = ap.parse_args()

    t0 = min([os.path.getmtime(p) for p in glob.glob(os.path.join(LOGS, '*.log'))] or [time.time()])
    console = Console(legacy_windows=False)
    if a.once:
        console.print(render(t0))
        return
    with Live(render(t0), console=console, refresh_per_second=4, screen=False) as live:
        try:
            while True:
                time.sleep(a.interval)
                live.update(render(t0))
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()

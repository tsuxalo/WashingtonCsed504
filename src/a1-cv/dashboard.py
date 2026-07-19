"""
dashboard.py, one live screen for the whole ImageNet-32 training fleet.

Four training processes, each spraying its own tqdm bar into its own log file, is unreadable.
So we don't watch the processes at all. We read what they write instead: runs/*.jsonl plus the
tqdm line at the tail of logs/*.log. Fold in a quick nvidia-smi, and draw one frame that answers
the only questions you actually have while a run is cooking:

    Is everything still alive?   Is it learning?   Is the GPU busy?   When will it be done?

It is read-only, and we lean on that hard: we only open files and shell out to nvidia-smi, and we
never touch the training processes. So you can start us, stop us, run several copies, or close us
mid-run, and nothing downstream even notices.

Usage:
    python dashboard.py                 # redraw every 2s until you hit Ctrl+C
    python dashboard.py --once          # print one snapshot and exit (nice for logs/screenshots)

Why read the files instead of the processes? The trainers already persist everything worth
knowing: finished-epoch metrics to JSONL, and live per-batch progress to the tqdm tail. Tailing
those is decoupled and crash-proof. If a trainer dies, its card simply stops updating, and the
dashboard never falls over with it.
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

# Force our stdout to UTF-8, and remember in UNICODE whether we actually got it. A Windows
# console defaults to cp1252, which can't encode a block char or a sparkline, so rich then dies
# with UnicodeEncodeError halfway through drawing a frame. The reconfigure() call is refused on
# a genuinely legacy console; if so we don't crash, we flip UNICODE off and fall back to the
# ASCII glyphs chosen below.
UNICODE = True
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    UNICODE = False

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, LOGS = os.path.join(HERE, 'runs'), os.path.join(HERE, 'logs')

# Same palette as the notebook's scoreboard: colorblind-safe, one hue per model so a given run
# stays the same color everywhere you look.
COLOR = {'resnet18': 'dark_orange3', 'resnet50': 'orange1',
         'vit': 'spring_green4', 'vit_base': 'medium_purple3'}
SPARK = ' ▁▂▃▄▅▆▇█' if UNICODE else ' .:-=+*#%'
FULL, EMPTY = ('█', '░') if UNICODE else ('#', '.')


def sparkline(vals, width=28) -> str:
    """The val-top1 curve as a one-line Unicode sparkline, the shape of learning at a glance.

    We keep the last `width` points and rescale [lo, hi] onto the 8 ramp glyphs. A dead-flat curve
    (hi == lo) would divide by zero, so we short-circuit it to the lowest non-blank glyph.
    """
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
                # A half-written final line; skip it, it'll be complete on the next tick.
                pass
    return out


def read_progress(tag):
    """Scrape the current epoch's progress out of the live tqdm bar at the tail of the log.

    Why do this when we already parse the JSONL? Because JSONL only gains a row when an epoch
    finishes, so between rows (a whole minute at a time) the dashboard would sit frozen. The
    in-flight batch count lives only in the tqdm tail, so that's where we go to get it.
    """
    p = os.path.join(LOGS, f'{tag}.log')
    if not os.path.exists(p):
        return None
    # tqdm overwrites its line with \r and no newline, so we read raw bytes.
    with open(p, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        tail = f.read().decode('utf-8', 'replace')
    frames = tail.replace('\r', '\n').split('\n')
    for line in reversed(frames):
        # Capture "<cur>/<tot> [", for example "1531/2503 [".
        m = re.search(r'(\d+)/(\d+)\s*\[', line)
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
    """Which models have a train_run.py process actually running right now?

    We ask Windows for every python.exe whose command line mentions train_run.py, then pull the
    --model value out of each into a set of live tags. That's what tells a 'DONE' card from a
    'STOPPED' one, and lets a card exist before its first JSONL row is written.
    """
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train_run.py' } | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
        return {m.group(1) for m in re.finditer(r'--model\s+(\S+)', out)}
    except Exception:
        return set()


def bar(frac, width=18, color='white'):
    n = int(max(0.0, min(1.0, frac)) * width)
    return Text(FULL * n + EMPTY * (width - n), style=color)


def render(t0):
    """Build the frame: a hardware panel on top, then one card per model stacked vertically.

    Why stacked cards, not one wide table? The first cut was a single wide table, and it needed
    about 135 columns, so in a normal terminal every column truncated to 'val to...' and
    'traini...'. A dashboard you can't read is worse than no dashboard. Stacked multi-line cards
    fit in about 90 columns and never truncate.
    """
    alive = live_tags()
    tags = sorted({os.path.basename(p).split('.')[0] for p in glob.glob(os.path.join(RUNS, '*.jsonl'))}
                  | alive)

    blocks = []

    # Step 1: Build the hardware panel, one row per GPU (util bar, mem, power, temp), color-coded
    # so a busy card reads green, a coasting one yellow, an idle or overheating one red.
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

    # Step 2: Build one card per model. For every tag we know about, whether it has a JSONL on
    # disk or is live in `alive`, we build a self-contained multi-line card and stack it under the
    # hardware panel.
    for tag in tags:
        rows = read_jsonl(tag)
        prog = read_progress(tag)
        col = COLOR.get(tag, 'white')
        running = tag in alive
        total = _total_epochs(tag)

        # Part A: Nothing logged yet means the run just started, so drop a 'starting...' stub and
        # skip the rest; there are no metrics to draw for it this frame.
        if not rows:
            blocks.append(Panel(Text('starting...', style='yellow'),
                                title=f'[bold {col}]{tag}', border_style='grey37', padding=(0, 1)))
            continue

        # Part B: Read the latest finished epoch off rows[-1]: current top-1, best top-1 and the
        # epoch it peaked at, and the train-minus-val gap we'll grade down in Part D.
        last = rows[-1]
        top1s = [r['val']['top1'] for r in rows]
        best = max(top1s)
        best_ep = top1s.index(best) + 1
        ep = last['epoch']
        tr1, va1 = last['train']['top1'], last['val']['top1']
        gap = tr1 - va1

        # Part C: The run's headline state. If it's not alive, it's 'DONE' when it hit its epoch
        # budget, else 'STOPPED' (red). If it's still alive but top-1 slid more than 0.02 below a
        # couple epochs ago it's 'REGRESSING'; otherwise it's just training.
        if not running:
            state = Text('DONE', style='bold green') if ep >= total else Text('STOPPED', style='bold red')
        elif len(top1s) >= 3 and top1s[-1] < top1s[-3] - 0.02:
            state = Text('REGRESSING', style='bold yellow')
        else:
            state = Text('training', style='green')

        # Part D: The overfitting verdict, but only when train-vs-val is a fair comparison. Under
        # mixup/CutMix the train accuracy is scored on blended images against the dominant label,
        # so it's a different (harder) exam than clean validation; the gap is not interpretable
        # and can even go negative. Judging it anyway is exactly the mistake we made reading the
        # CIFAR curves, so we print 'gap n/a (mixup)' and leave it ungraded.
        aug = _strong_aug(tag)
        if aug:
            health = Text('gap n/a (mixup)', style='dim')
        elif gap > 0.35:
            health = Text(f'gap {gap:+.0%} MEMORIZING', style='bold red')
        elif gap > 0.15:
            health = Text(f'gap {gap:+.0%} overfitting', style='yellow')
        else:
            health = Text(f'gap {gap:+.0%} healthy', style='green')

        # Part E: Progress and ETA. `frac` blends finished epochs with the live tqdm fraction of
        # the current one; per-epoch time is the median of the last 5 (plus 4s slack), giving a
        # steady ETA that doesn't lurch when one epoch happens to run long.
        frac = (ep + (prog['frac'] if prog else 0)) / max(1, total)
        ips = (prog or {}).get('img_s') or last['train']['img_s']
        recent = [r['train']['sec'] for r in rows[-5:]]
        per_ep = sorted(recent)[len(recent) // 2] + 4
        eta = _fmt((total - ep) * per_ep) if running else '-'

        # Part F: Assemble the card. Row 1 is the progress bar, state, throughput and ETA, row 2
        # the val/best/train numbers and health verdict, row 3 the full-history top-1 sparkline.
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
    """Did this run use mixup/CutMix? We scan the log for the 'strong aug' line to find out.

    It matters because under mixup/CutMix the train accuracy is scored on blended images against
    the dominant label, a different, harder exam than clean validation. So the train/val 'gap'
    stops meaning anything (it can even go negative). Same trap the CIFAR notebook set for us:
    augmented-train vs clean-test are two different exams. So when this is True we label the run
    instead of pretending to judge its gap.
    """
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

    Why not just hardcode it? Because we did, and it bit us: the hardcoded table claimed vit_base
    ran 60 epochs when it actually ran 40, so a completed run got drawn as 'STOPPED' in red. Never
    hardcode what the process already prints for you; we parse "<N> epochs, batch" out of the
    header, else fall back to 40.
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

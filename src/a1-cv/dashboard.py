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

# The same scheme the notebooks use: one hue per dataset, the CNN its darker shade and the ViT its
# lighter one. Holding a CNN against a ViT on the same data is the comparison that matters, so those
# two share a hue; the datasets, which are not comparable to each other, are told apart by hue instead.
# The terminal names are the closest 256-color matches to the notebook's hexes, so a run looks the
# same here as it does in the charts (dark_orange3 is rgb 215,95,0 against #D55E00's 213,94,0).
PALETTE = {
    'cifar10':    ('dark_orange3',   'orange1'),
    'cifar100':   ('deep_sky_blue4', 'sky_blue2'),
    'imagenet32': ('green4',         'aquamarine3'),
}


def color_for(tag):
    """The run's color, matching the notebooks.

    Tags are <dataset>_<model>[_variant][_sN], so the dataset comes off the front and the family from
    the model that follows. Every seed and variant of one family therefore shares its shade, which is
    what we want on a dashboard showing three seeds of the same run side by side. Anything we do not
    recognise falls back to white rather than failing.
    """
    for dataset, (cnn, vit) in PALETTE.items():
        if tag.startswith(dataset + '_'):
            model = tag[len(dataset) + 1:]
            return vit if model.startswith('vit') else cnn

    return 'white'


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
    """Which runs have a train_run.py process actually running right now?

    We ask Windows for every python.exe whose command line mentions train_run.py and work out the tag
    each one is writing under, exactly the way train_run.py does: an explicit --tag if the process was
    given one, otherwise <dataset>_<model>. That is what tells a 'DONE' card from a 'STOPPED' one, and
    lets a card exist before its first JSONL row is written.

    We must rebuild the whole tag, not just read --model. Matching on --model alone drew a live
    cifar100_vit as STOPPED, because its process says '--model vit' while its tag is 'cifar100_vit'.
    """
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match 'train_run.py' } | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
        tags = set()
        for line in out.splitlines():
            # An explicit --tag wins, because that is the name train_run.py writes its files under.
            # The seed repeats rely on this: they run '--model resnet18 --tag resnet18_s1', so
            # rebuilding from the model alone would credit the live run to the older resnet18 card
            # and leave the repeat looking STOPPED for the whole night.
            mt = re.search(r'--tag\s+(\S+)', line)
            if mt:
                base = mt.group(1)
            else:
                mm = re.search(r'--model\s+(\S+)', line)
                if not mm:
                    continue

                model = mm.group(1)
                md = re.search(r'--dataset\s+(\S+)', line)
                dataset = md.group(1) if md else 'imagenet32'
                base = model if dataset == 'imagenet32' else f'{dataset}_{model}'

            # The smoke prefix goes on last, whichever way the stem was found, mirroring train_run.py.
            tags.add(f'smoke-{base}' if '--smoke-test' in line else base)
        return tags
    except Exception:
        return set()


def bar(frac, width=18, color='white'):
    n = int(max(0.0, min(1.0, frac)) * width)
    return Text(FULL * n + EMPTY * (width - n), style=color)


_REF_IPS_CACHE = {}


def _ref_ips(model):
    """This model's characteristic throughput (median img/s) from a prior completed run -- the
    'previous stats' the Predicted ETA is built on. Read once per model and cached. It transfers
    across datasets because the models are all 32x32: the ImageNet vit and a CIFAR vit push the same
    FLOPs per image, so the same card serves them at the same rate."""
    if model not in _REF_IPS_CACHE:
        ips = None
        try:
            d = json.load(open(os.path.join(RUNS, f'{model}_result.json')))
            vals = sorted(e['train']['img_s'] for e in d.get('history', [])
                          if 'train' in e and 'img_s' in e['train'])
            ips = vals[len(vals) // 2] if vals else None
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            ips = None
        _REF_IPS_CACHE[model] = ips
    return _REF_IPS_CACHE[model]


def _predict_total_s(dataset, model, total_epochs):
    """Predicted wall-clock for the whole run, from the model's characteristic throughput on a prior
    run rather than this run's live pace -- so it is a genuine prediction to hold the live estimate
    against, not a restatement of it. None when we have no reference throughput for the model yet.

    We can't probe this workstation for a peak-TFLOPS roofline while it is busy training (the probe
    would contend with the run and misread), so we use the measured previous-run throughput directly,
    which is the same quantity the roofline would have produced for a repetitive training loop."""
    ref = _ref_ips(model)
    if not ref or not dataset:
        return None

    # Predicted wall clock: images per epoch over throughput, times the epoch count, plus a one-time
    # startup cost (data upload and autotune) that isn't part of the steady-state per-epoch pace.
    n_train = 1_281_167 if dataset == 'imagenet32' else 50_000
    return total_epochs * n_train / ref + 15.0


def render(t0):
    """Build the frame: a hardware panel on top, then one card per model stacked vertically.

    Why stacked cards, not one wide table? The first cut was a single wide table, and it needed
    about 135 columns, so in a normal terminal every column truncated to 'val to...' and
    'traini...'. A dashboard you can't read is worse than no dashboard. Stacked multi-line cards
    fit in about 90 columns and never truncate.
    """
    alive = live_tags()
    # This session's fleet only: runs training now, or whose log was written in the last ~18 hours.
    # That keeps the board on the current overnight run and drops the old experiments in runs/ from
    # previous days -- and unlike tracking "seen since startup", it survives a dashboard restart, so a
    # run you kicked off before opening this still shows.
    recent = time.time() - 18 * 3600
    tags = set(alive)
    for p in glob.glob(os.path.join(LOGS, '*.log')):
        try:
            if os.path.getmtime(p) > recent:
                tags.add(os.path.basename(p)[:-4])
        except OSError:
            pass
    tags = sorted(tags)

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
    blocks.append(Panel(g, title='[bold]hardware', border_style='gray37', padding=(0, 1)))

    # Step 2: Build one card per model. For every tag we know about, whether it has a JSONL on
    # disk or is live in `alive`, we build a self-contained multi-line card and stack it under the
    # hardware panel.
    for tag in tags:
        rows = read_jsonl(tag)
        prog = read_progress(tag)
        col = color_for(tag)
        running = tag in alive
        total = _total_epochs(tag)
        gpu, params, dataset = _run_meta(tag)

        # The card title carries the run's dataset, size, and card, e.g. "vit   cifar100  11M param
        # cuda:0". The parameter count is the main reason throughput differs so much between models --
        # a bigger model does more math per image -- and the card says which GPU the run is pinned to.
        title = f'[bold {col}]{tag}[/]'
        meta = (([dataset] if dataset else [])
                + ([f'{params/1e6:.0f}M param'] if params else [])
                + ([f'cuda:{gpu}'] if gpu is not None else []))
        if meta:
            title += '   [dim]' + '  '.join(meta) + '[/]'

        # Part A: Nothing logged yet means the run just started, so drop a 'starting...' stub and
        # skip the rest; there are no metrics to draw for it this frame.
        if not rows:
            blocks.append(Panel(Text('starting...', style='yellow'),
                                title=title, border_style='gray37', padding=(0, 1)))
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

        # Part C: The run's headline state, as (text, style). If it's not alive, it's 'DONE' when it
        # hit its epoch budget, else 'STOPPED' (red). If it's still alive but top-1 slid more than
        # 0.02 below a couple epochs ago it's 'REGRESSING'; otherwise it's just training.
        if not running:
            state_txt, state_sty = ('DONE', 'bold green') if ep >= total else ('STOPPED', 'bold red')
        elif len(top1s) >= 3 and top1s[-1] < top1s[-3] - 0.02:
            state_txt, state_sty = 'REGRESSING', 'bold yellow'
        else:
            state_txt, state_sty = 'training', 'green'

        # Part D: The overfitting verdict, but only when train-vs-val is a fair comparison. Under
        # mixup/CutMix the train accuracy is scored on blended images against the dominant label, so
        # it's a different (harder) exam than clean validation; the gap is not interpretable and can
        # even go negative. Judging it anyway is exactly the mistake we made reading the CIFAR curves,
        # so we print 'gap n/a (mixup)' and leave it ungraded.
        aug = _strong_aug(tag)
        if aug:
            health_txt, health_sty = 'gap n/a (mixup)', 'dim'
        elif gap > 0.35:
            health_txt, health_sty = f'gap {gap:+.0%} MEMORIZING', 'bold red'
        elif gap > 0.15:
            health_txt, health_sty = f'gap {gap:+.0%} overfitting', 'yellow'
        else:
            health_txt, health_sty = f'gap {gap:+.0%} healthy', 'green'

        # Part E: Progress, throughput, and the two ETAs. `frac` blends finished epochs with the live
        # tqdm fraction of the current one. `per_ep` is this run's own recent wall time per epoch (from
        # the cumulative-elapsed deltas, so it counts eval), which drives the ESTIMATED time left. The
        # PREDICTED total is independent: the model's characteristic throughput on a prior run (see
        # _predict_total_s), a real prediction to hold the live estimate against.
        frac = (ep + (prog['frac'] if prog else 0)) / max(1, total)
        ips = (prog or {}).get('img_s') or last['train']['img_s']
        elapsed = last['elapsed']
        k = min(5, len(rows) - 1)
        per_ep = (elapsed - rows[-k - 1]['elapsed']) / k if k > 0 else elapsed
        remaining = _fmt((total - ep) * per_ep) if (running and ep < total) else '-'
        model = tag[len(dataset) + 1:] if (dataset and tag.startswith(dataset + '_')) else tag
        pred_s = _predict_total_s(dataset, model, total)
        predicted = _fmt(pred_s) if pred_s else '?'
        lr = last.get('lr')

        # Part F: Assemble the card as fixed-width lines, so the same field sits at the same column on
        # every card. The sparkline is its own line, NOT a grid cell: as a column-0 cell its width
        # (60 for a long run, 40 for a short one) stretched that column and shoved every field right by
        # a different amount per card -- exactly the misalignment this fixes.
        L1 = bar(frac, 30, col) + Text.assemble(
            '  ', (f'ep {ep:>3}/{total:<4}', 'bold'), '   ',
            (f'{state_txt:<11}', state_sty), (f'{ips/1000:>6.1f}k img/s', 'dim'))
        L2 = Text.assemble(
            ('val   ', 'dim'), (f'top1 {va1:6.2%}  top5 {last["val"]["top5"]:6.2%}', col),
            ('    best ', 'dim'), (f'{best:6.2%} @ep{best_ep:<4}', 'bold'), (health_txt, health_sty))
        L3 = Text.assemble(
            ('train ', 'dim'), (f'{"(aug) " if aug else "      "}{tr1:6.2%}', 'dim'),
            ('    lr ', 'dim'), (f'{lr:8.5f}' if lr is not None else '   -    ', 'dim'),
            ('    loss ', 'dim'), (f'{last["train"]["loss"]:6.3f}', 'dim'))
        L4 = Text.assemble(
            ('time  ', 'dim'), ('elapsed ', 'dim'), (f'{_fmt(elapsed):<6}', ''),
            ('  remaining ', 'dim'), (f'{remaining:<6}', 'cyan'),
            ('  predicted ', 'dim'), (f'{predicted:<6}', 'magenta'),
            ('  ', 'dim'), (f'{per_ep:.1f}s/ep', 'dim'))
        L5 = Text(sparkline(top1s, 60), style=col)
        blocks.append(Panel(Group(L1, L2, L3, L4, L5), title=title, border_style='gray37', padding=(0, 1)))

    blocks.append(Text('  ref: WRN-28-10 = 59.0% top-1 / 81.1% top-5 (Chrabaszcz 2017)  |  '
                       f'watching {_fmt(time.time()-t0)}  |  read-only, Ctrl+C safe', style='dim'))
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
    aug = False
    try:
        with open(p, errors='replace') as f:
            for line in f:
                if 'strong aug' in line:
                    # Keep scanning instead of breaking: if a reused log holds several runs, the last
                    # 'strong aug' line is the current run's, so the newest one wins.
                    aug = 'True' in line
    except OSError:
        pass
    return aug


def _total_epochs(tag):
    """Read the real epoch count for the current run, robust to stale content in a reused log.

    We prefer the live tqdm "epoch <cur>/<tot>" at the tail, because that is written by the run
    happening right now. If an earlier run (say a smoke test) left a header with a different count
    at the top of the same log file, a first-match header read would report that stale number -- it
    once drew a 40-epoch run as 2/2. The tqdm tail cannot be stale; it is the process's current
    line. We fall back to the newest "<N> epochs, batch" header, then to 40.
    """
    p = os.path.join(LOGS, f'{tag}.log')
    total = None
    try:
        # The live tqdm tail first -- "epoch   6/40 train: ...".
        with open(p, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8000))
            tail = f.read().decode('utf-8', 'replace').replace('\r', '\n')
        for line in reversed(tail.split('\n')):
            m = re.search(r'epoch\s+\d+/(\d+)', line)
            if m:
                return int(m.group(1))
        # No tqdm yet (run just starting): take the last header, so the newest run wins.
        with open(p, errors='replace') as f:
            for line in f:
                m = re.search(r'(\d+) epochs, batch', line)
                if m:
                    total = int(m.group(1))
    except OSError:
        pass
    return total or 40


def _run_meta(tag):
    """Parse which GPU a run is on and its parameter count, from its log header.

    The trainer prints "[tag] device cuda:N (...)" and "[tag] <model>: <N> parameters" at startup.
    We take the last match of each, so a stale header left by an earlier run in a reused log does not
    win over the current run's.
    """
    p = os.path.join(LOGS, f'{tag}.log')
    gpu, params, dataset = None, None, None
    try:
        with open(p, errors='replace') as f:
            for line in f:
                m = re.search(r'device cuda:(\d+)', line)
                if m:
                    gpu = int(m.group(1))
                m = re.search(r'([\d,]+) parameters', line)
                if m:
                    params = int(m.group(1).replace(',', ''))
                # Anchor on the " |" that follows the dataset in the config line ("dataset cifar100 |
                # strong aug ..."), so we don't match the words in "dataset resident on GPU: ..." that
                # older ImageNet logs print, which would set dataset to 'resident'.
                m = re.search(r'dataset (\w+)\s+\|', line)
                if m:
                    dataset = m.group(1)
    except OSError:
        pass
    return gpu, params, dataset


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

    # Elapsed is measured from when this dashboard started, not from log mtimes: old logs from
    # earlier runs (days ago) would otherwise drag "watching" back to a bogus 100+ hours.
    t0 = time.time()
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

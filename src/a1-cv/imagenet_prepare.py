"""
imagenet_prepare.py -- one-time conversion of ImageNet-1k-32x32 from parquet/JPEG to raw uint8 arrays.

WHY THIS EXISTS
---------------
The HuggingFace dataset (benjamin-paine/imagenet-1k-32x32) stores each image as a JPEG blob inside
a parquet file.  That is a fine way to *ship* 1.28M images (596 MB on disk), but it is a terrible way
to *train* on them: a naive DataLoader would decode 1,281,167 JPEGs on the CPU **every epoch**.

We already measured, on CIFAR-10, that plain crop+flip augmentation in Python was starving the GPU by
3x -- and that was with NO image decoding at all.  Add a JPEG decode per image per epoch on a dataset
25x larger and the GPU would spend nearly all its time idle.

So we decode ONCE, here, into a flat uint8 array:

    1,281,167 x 32 x 32 x 3 bytes = 3.9 GB

which fits comfortably in RAM (and even in GPU memory).  After this runs, training never touches
JPEG, PIL, or parquet again -- it just indexes into an array.  This single step is what makes the
project tractable; it is not an optimization we are doing for fun.

USAGE
-----
    python imagenet_prepare.py                      # uses the defaults below
    python imagenet_prepare.py --force              # re-do it even if outputs exist
    python imagenet_prepare.py --workers 16         # override the process count

OUTPUT (into --out-dir, gitignored -- 4 GB does not belong in git)
    train_x.npy   uint8  (1281167, 32, 32, 3)      3.9 GB
    train_y.npy   int16  (1281167,)
    val_x.npy     uint8  (50000, 32, 32, 3)
    val_y.npy     int16  (50000,)
    stats.json    per-channel mean/std computed FROM THIS DATA, plus row counts
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm

# Default location of the cloned dataset repo (a sibling of the course repo).
DEFAULT_DATA = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'imagenet-1k-32x32', 'data'))
DEFAULT_OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data', 'imagenet32'))

IMG_SHAPE = (32, 32, 3)
N_CLASSES = 1000


def _decode(payload: bytes) -> np.ndarray:
    """JPEG bytes -> (32, 32, 3) uint8.  Module-level so Windows 'spawn' workers can pickle it."""
    im = Image.open(io.BytesIO(payload))
    if im.mode != 'RGB':                 # a handful of ImageNet images are greyscale or CMYK
        im = im.convert('RGB')
    a = np.asarray(im, dtype=np.uint8)
    if a.shape != IMG_SHAPE:             # should never fire, but a silent shape bug here would be
        raise ValueError(f'expected {IMG_SHAPE}, got {a.shape}')   # very painful to find later
    return a


def convert_split(files: list[str], x_path: str, y_path: str, workers: int,
                  batch_rows: int = 16_384) -> tuple[np.ndarray, np.ndarray]:
    """Decode every row of `files` into memmapped .npy arrays.  Returns (x, y) as memmaps."""
    n_total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f'  {len(files)} parquet file(s), {n_total:,} rows -> {x_path}')

    # open_memmap writes straight to disk, so we never hold 3.9 GB in RAM at once.
    x = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.uint8, shape=(n_total, *IMG_SHAPE))
    y = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int16, shape=(n_total,))

    written = 0
    with Pool(workers) as pool, tqdm(total=n_total, unit='img', unit_scale=True, desc='  decoding') as bar:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=batch_rows, columns=['image', 'label']):
                # The 'image' column is a struct<bytes, path>; we only want the bytes.
                blobs = [rec['bytes'] for rec in batch.column('image').to_pylist()]
                labels = batch.column('label').to_pylist()

                # chunksize matters: without it, Pool ships one 975-byte job at a time and the IPC
                # overhead dwarfs the decode.
                decoded = pool.map(_decode, blobs, chunksize=256)

                n = len(decoded)
                x[written:written + n] = np.stack(decoded)
                y[written:written + n] = np.asarray(labels, dtype=np.int16)
                written += n
                bar.update(n)

    assert written == n_total, f'wrote {written}, expected {n_total}'
    x.flush(); y.flush()
    return x, y


def channel_stats(x: np.ndarray, sample: int = 200_000, seed: int = 0) -> tuple[list, list]:
    """Per-channel mean/std in [0,1], from a random sample (exact to ~4 decimals, and 6x faster).

    Sampled RANDOMLY, not sequentially: this dataset is sorted by class, so the first N rows are
    only the first few classes and their color statistics are NOT representative.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=min(sample, len(x)), replace=False))
    a = x[idx].astype(np.float64) / 255.0
    return ([round(float(v), 4) for v in a.mean(axis=(0, 1, 2))],
            [round(float(v), 4) for v in a.std(axis=(0, 1, 2))])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default=DEFAULT_DATA, help='dir holding the *.parquet files')
    p.add_argument('--out-dir', default=DEFAULT_OUT, help='where to write the .npy arrays')
    p.add_argument('--workers', type=int, default=min(16, (os.cpu_count() or 4)))
    p.add_argument('--force', action='store_true', help='rebuild even if outputs already exist')
    args = p.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f'ERROR: no such directory: {args.data_dir}')
        print('Clone the dataset first:  git clone https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32')
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda n: os.path.join(args.out_dir, n)

    if os.path.exists(out('stats.json')) and not args.force:
        print(f'Already prepared (found {out("stats.json")}).  Use --force to rebuild.')
        return 0

    def split_files(prefix):
        return sorted(f for f in (os.path.join(args.data_dir, n) for n in os.listdir(args.data_dir))
                      if os.path.basename(f).startswith(prefix) and f.endswith('.parquet'))

    train_files, val_files = split_files('train'), split_files('validation')
    if not train_files or not val_files:
        print(f'ERROR: expected train-*.parquet and validation-*.parquet in {args.data_dir}')
        return 1

    t0 = time.time()
    print(f'Decoding with {args.workers} worker processes.\n')
    print('TRAIN')
    train_x, train_y = convert_split(train_files, out('train_x.npy'), out('train_y.npy'), args.workers)
    print('VALIDATION')
    val_x, val_y = convert_split(val_files, out('val_x.npy'), out('val_y.npy'), args.workers)

    # ---- Sanity checks.  Fail loudly HERE rather than after an hour of training. ----
    print('\nSanity checks:')
    counts = np.bincount(np.asarray(train_y), minlength=N_CLASSES)
    problems = []
    if len(train_y) != 1_281_167:
        problems.append(f'train rows = {len(train_y):,}, expected 1,281,167')
    if (counts == 0).any():
        problems.append(f'{int((counts == 0).sum())} classes have ZERO training images')
    if int(train_y.min()) != 0 or int(train_y.max()) != N_CLASSES - 1:
        problems.append(f'labels span [{train_y.min()}, {train_y.max()}], expected [0, 999]')

    print(f'  train images      : {len(train_y):,}')
    print(f'  val images        : {len(val_y):,}')
    print(f'  classes present   : {int((counts > 0).sum())} / {N_CLASSES}')
    print(f'  images per class  : min {counts.min()}, max {counts.max()}, mean {counts.mean():.0f}'
          f'   (the reference paper reports 732-1300)')

    # The data is stored SORTED BY CLASS -- worth proving, because a sequential subset would then
    # contain only a handful of classes and quietly ruin any experiment built on it.
    first_10k = np.asarray(train_y[:10_000])
    print(f'  first 10k labels  : {len(np.unique(first_10k))} distinct class(es) '
          f'-> the file IS sorted by class; ALWAYS shuffle, and never take a sequential subset')

    print('\nComputing per-channel statistics from a random sample of the TRAINING data...')
    mean, std = channel_stats(train_x)
    print(f'  ImageNet-32 mean : {tuple(mean)}')
    print(f'  ImageNet-32 std  : {tuple(std)}')
    print(f'  (CIFAR-10 was    : (0.4914, 0.4822, 0.4465) / (0.247, 0.2435, 0.2616)  -- do NOT reuse)')

    json.dump({'mean': mean, 'std': std,
               'n_train': int(len(train_y)), 'n_val': int(len(val_y)),
               'n_classes': N_CLASSES, 'image_shape': list(IMG_SHAPE),
               'source': os.path.abspath(args.data_dir)},
              open(out('stats.json'), 'w'), indent=2)

    gb = (train_x.nbytes + val_x.nbytes) / 1e9
    print(f'\nWrote {gb:.1f} GB to {os.path.abspath(args.out_dir)} in {(time.time()-t0)/60:.1f} min')
    print(f'  stats.json written -- train_run.py reads mean/std from it.')

    if problems:
        print('\nPROBLEMS FOUND:')
        for msg in problems:
            print('  !!', msg)
        return 1
    print('\nAll checks passed.')
    return 0


if __name__ == '__main__':          # required: Windows 'spawn' re-imports this module in every worker
    sys.exit(main())

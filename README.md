# UW CSED 504 — Summer 2026

Shared repository for the University of Washington **CSED 504** course, Summer 2026.

## Repository Structure

```
WashingtonCsed504/
├── assignments/    # Homework and assignment starter code
├── labs/           # Lab exercises and in-class activities
├── resources/      # Supplementary reading materials and references
└── projects/       # Course project templates and guidelines
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
   cd WashingtonCsed504
   ```

2. **Set up a Python virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Stay up to date**
   ```bash
   git pull origin main
   ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting any code or assignments.

## License

This repository is licensed under the [MIT License](LICENSE).
# WashingtonCsed504

Shared environment and starter code for **UW CSED 504** (Computer Vision + NLP).

Each platform has a one-shot setup script that creates a conda environment named
**`uw-csed504`** (Python 3.12) with a matching package set, so everyone in the group
runs the same stack whether they're on Windows, macOS, Linux, or Google Colab.

---

## What's in here

| Path | Purpose |
|------|---------|
| `setup_windows.ps1` | Windows setup (NVIDIA CUDA 12.8) |
| `setup_mac.sh` | macOS setup (Apple MPS / CPU) |
| `setup_linux.sh` | Linux / WSL2 setup (NVIDIA CUDA 12.8, or CPU) |
| `cuda_check.ps1` | Windows: (re)configure GPU visibility any time |
| `src/common/gpu_check.py` | Shared device detection + multi-GPU helpers |
| `src/a1-cv/hello_image.ipynb` | A1 sanity notebook (builds a Vision Transformer) |
| `src/a2-nlp/hello_text.ipynb` | A2 sanity notebook |

If a hello notebook runs top-to-bottom without errors, your environment is ready.

---

## Prerequisites: install Anaconda / Miniconda

The local setups (Windows / macOS / Linux) use **conda**. If you don't already have it,
grab an installer from the official download page — it offers both the full **Anaconda
Distribution** and the smaller **Miniconda** (either works; Miniconda is recommended):

- **Download page (Windows / macOS / Linux):** https://www.anaconda.com/download
- **Direct installer archive (all OSes/versions):** https://repo.anaconda.com/miniconda/

Pick the installer for your OS (Windows `.exe`, macOS `.pkg` / Apple-Silicon, Linux `.sh`).

<details>
<summary>Linux / WSL2 one-liner Miniconda install</summary>

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all      # restart your shell afterward
```
</details>

> **Google Colab needs none of this** — conda and PyTorch are already there. Jump to
> [Google Colab](#google-colab).

---

## Get the code

```bash
git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
cd WashingtonCsed504
```

---

## Setup by platform

### Windows (NVIDIA GPU)

Uses CUDA 12.8 wheels (Blackwell `sm_120`-compatible) and auto-detects all GPUs.

1. Open the **Anaconda Prompt** (Start menu → "Anaconda Prompt"), **not** plain PowerShell —
   the script needs `conda` on the path.
2. `cd` to the repo, then run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
   ```

The script installs everything with pip (so PyTorch is the only OpenMP provider — this is
what avoids **OMP Error #15**), registers the Jupyter kernel, and pins every same-architecture
GPU. Re-run GPU detection any time with `.\cuda_check.ps1`.

### macOS (Apple Silicon or Intel)

Uses Apple **MPS** (Metal) acceleration on Apple-Silicon Macs; CPU otherwise.

```bash
conda activate base
bash setup_mac.sh
```

### Linux / WSL2 (NVIDIA GPU or CPU)

Uses CUDA 12.8 wheels like Windows. CPU-only machines work too (PyTorch falls back to CPU).

```bash
conda activate base
bash setup_linux.sh
```

> Run scripts with `bash setup_*.sh` (no `chmod` needed). WSL2 counts as Linux — use this script.

### Google Colab

No local setup. The hello notebooks self-install their packages and include the clone step.

1. Open the notebook in Colab (e.g. from GitHub: **File → Open notebook → GitHub**, paste the
   repo URL, pick `src/a1-cv/hello_image.ipynb`).
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is fine).
3. Run all cells. The first cells clone the repo, `%cd` into the notebook's folder, and
   `%pip install` the needed packages.

---

## Using the environment

```bash
conda activate uw-csed504
```

- In **VS Code** or **Jupyter**, select the kernel **"Python (uw-csed504)"**.
- Verify the install and see your device:

  ```bash
  python src/common/gpu_check.py
  ```

  Expected device line by platform:

  | Platform | Output |
  |----------|--------|
  | Windows / Linux + NVIDIA | `Device : cuda [N GPUs visible ...]` |
  | macOS (Apple Silicon) | `Device : MPS - Apple Silicon GPU` |
  | CPU-only / Colab CPU | `Device : CPU (...)` |

Then open `src/a1-cv/hello_image.ipynb` or `src/a2-nlp/hello_text.ipynb`, choose the
`uw-csed504` kernel, and **Run All**. Each ends with "All checks passed."

---

## GPU notes (`src/common/gpu_check.py`)

`get_device()` picks the best device (CUDA → MPS → CPU) and, on multi-GPU NVIDIA machines,
makes all same-architecture GPUs visible. Helpers you can import:

- `get_device()` / `set_seed(42)` — device + reproducibility (used by the notebooks).
- `enable_fast_matmul()` — TF32 + cuDNN autotune; pair with **bf16 autocast** for the biggest
  single-GPU training speedup.
- `get_data_parallel_model(model, DEVICE)` — `nn.DataParallel` across GPUs (helps only when a
  step's compute is large enough to outweigh cross-GPU communication).
- `get_max_memory()` — budget dict for HuggingFace `device_map="auto"` to split a model that's
  too big for one card across multiple GPUs.

---

## Troubleshooting

- **`OMP Error #15` / duplicate `libiomp5md`** — you have a mixed conda+pip install. Re-run the
  setup script for your platform; it installs an all-pip stack so PyTorch is the sole OpenMP
  provider.
- **`conda: command not found`** — on Windows use the **Anaconda Prompt**; on macOS/Linux run
  `conda activate base` first (or `source ~/miniconda3/bin/activate`).
- **`torch.cuda.is_available()` is False** on an NVIDIA box — check the driver with `nvidia-smi`,
  and make sure you're in the `uw-csed504` env (`conda activate uw-csed504`).
- **Permission denied running a `.sh`** — invoke it as `bash setup_linux.sh` (or `setup_mac.sh`);
  no execute bit required.

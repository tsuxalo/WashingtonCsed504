# Hardware and Parallelism

## Single-GPU Colab

Default: one active training trial. Concurrent trials on one GPU are not enabled without a measured throughput and memory benefit. A trial uses native BF16 when available, otherwise FP16 with a scaler, CNN channels-last, TF32 where supported, the repository's resident data path, and zero DataLoader workers.

`torch.cuda.is_bf16_supported()` permits emulation by default in current PyTorch, so the hardware profile separately checks native BF16 support with `including_emulation=False` when available.

## Multiple GPUs

The scheduler assigns one independent spawned worker process per GPU. Each worker records its device, retains one resident dataset, and continues accepting queued trials after a recoverable failure. Heterogeneous GPUs are conservatively limited by visible device count and user memory reserves. DDP is not the HPO default; it belongs to separately measured single-trial scaling.

## CPU

Physical cores are divided among trial processes, with a conservative eight-thread cap unless explicitly overridden. Inter-op threads default to one. This prevents nested OpenMP/MKL oversubscription. RAM headroom limits trial concurrency.

## Apple MPS

MPS uses one active trial and FP32 by default. The original repository loop has CUDA-specific AMP behavior, so MPS mixed precision is not claimed without a measured adapter.

## User overrides

`device`, `gpu_id`, `concurrent_trials`, `intraop_threads`, `interop_threads`, `workers`, `precision`, `tf32`, `channels_last`, `compile`, and `memory_reserve_gb` are configurable.

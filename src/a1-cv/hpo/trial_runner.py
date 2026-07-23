from __future__ import annotations

import gc
import inspect
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import psutil
import torch
import torch.nn as nn

from .adapters import (
    DatasetBundle,
    RepoModules,
    build_trial_dataset,
    build_trial_model,
    build_trial_optimizer,
    build_trial_scheduler,
    subset_view,
)
from .constraints import validate_candidate
from .exceptions import InvalidTrialError, TrialExecutionError
from .hardware import HardwareProfile, resolve_device, resolve_precision


@dataclass
class TrialResource:
    stage: str
    target_epochs: int
    max_steps: int | None = None
    data_fraction: float = 1.0
    validation_fraction: float = 1.0
    seed: int = 42
    evaluate_test: bool = False
    continue_checkpoint: bool = True


@dataclass
class TrialResult:
    candidate_id: str
    trial_number: int | None
    stage: str
    status: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    resource: dict[str, Any]
    checkpoint_path: str | None
    failure_reason: str | None = None
    invalid_reason: str | None = None
    prune_reason: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


Reporter = Callable[[int, dict[str, Any]], bool]

# FLOP counting through TorchDispatch/FlopCounterMode is architecture-specific
# and can be expensive or unstable when repeatedly initialized in one notebook
# process. Share measurements across TrialRunner instances.
_GLOBAL_FLOPS_CACHE: dict[str, dict[str, Any]] = {}


class TrialRunner:
    def __init__(
        self,
        repo_root: str | Path,
        *,
        dataset_config: dict[str, Any],
        model_config: dict[str, Any],
        runtime_config: dict[str, Any],
        output_dir: str | Path,
        hardware: HardwareProfile | None = None,
    ):
        self.modules = RepoModules(repo_root)
        self.dataset_config = dict(dataset_config)
        self.model_config = dict(model_config)
        self.runtime_config = dict(runtime_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hardware = hardware
        self.device = resolve_device(
            str(self.runtime_config.get("device", "auto")),
            self.runtime_config.get("gpu_id"),
        )
        self.precision = resolve_precision(
            str(self.runtime_config.get("precision", "auto")),
            self.device,
            hardware,
        )
        # The repository training loop runs in-process. Explicitly apply the
        # scheduler's CPU-thread allocation here; otherwise every serial trial
        # inherits all logical cores, which can make tiny CPU trials hundreds
        # of times slower through nested OpenMP/MKL oversubscription.
        intraop_threads = max(1, int(self.runtime_config.get("intraop_threads") or 1))
        interop_threads = max(1, int(self.runtime_config.get("interop_threads") or 1))
        torch.set_num_threads(intraop_threads)
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError:
            # PyTorch permits setting inter-op threads only before parallel
            # work begins. Reused notebook processes may already be initialized.
            pass
        os.environ["OMP_NUM_THREADS"] = str(intraop_threads)
        os.environ["MKL_NUM_THREADS"] = str(intraop_threads)
        self._dataset: DatasetBundle | None = None
        self._flops_cache = _GLOBAL_FLOPS_CACHE

    @property
    def dataset(self) -> DatasetBundle:
        if self._dataset is None:
            self._dataset = build_trial_dataset(
                self.modules,
                self.dataset_config,
                self.runtime_config,
                self.device,
            )
        return self._dataset

    def _checkpoint_path(self, candidate_id: str) -> Path:
        directory = self.output_dir / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{candidate_id}.pt"

    def _model_config(self, params: dict[str, Any]) -> dict[str, Any]:
        result = dict(self.model_config)
        architecture_keys = {
            "model",
            "hidden_dim",
            "layers",
            "heads",
            "mlp_dim",
            "patch_size",
        }
        for key in architecture_keys:
            if key in params:
                result["name" if key == "model" else key] = params[key]
        return result

    def _optimizer_config(self, params: dict[str, Any], model_name: str) -> dict[str, Any]:
        is_cnn = model_name.startswith("resnet")
        batch_size = int(params.get("batch_size", 128))
        default_lr = 0.1 * batch_size / 256 if is_cnn else 1e-3 * batch_size / 512
        result = {
            "optimizer": params.get("optimizer", "sgd" if is_cnn else "adamw"),
            "learning_rate": params.get("learning_rate", default_lr),
            "weight_decay": params.get("weight_decay", 5e-4 if is_cnn else 0.05),
            "momentum": params.get("momentum", 0.9),
            "nesterov": params.get("nesterov", is_cnn),
            "dampening": params.get("dampening", 0.0),
            "beta1": params.get("beta1", 0.9),
            "beta2": params.get("beta2", 0.999),
            "epsilon": params.get("epsilon", 1e-8),
            "amsgrad": params.get("amsgrad", False),
        }
        return result

    def _scheduler_config(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "scheduler": params.get("scheduler", "cosine"),
            "warmup_epochs": params.get("warmup_epochs", params.get("warmup", 5)),
            "minimum_lr": params.get("minimum_lr", 0.0),
            "step_size": params.get("step_size"),
            "gamma": params.get("gamma", 0.1),
            "milestones": params.get("milestones"),
        }

    def _flops(self, model_config: dict[str, Any], num_classes: int) -> dict[str, Any]:
        key = json.dumps({"model": model_config, "classes": num_classes}, sort_keys=True)
        if key in self._flops_cache:
            return self._flops_cache[key]
        try:
            value = self.modules.perfkit.count_flops_per_image(
                lambda: build_trial_model(
                    self.modules,
                    model_config,
                    num_classes,
                    torch.device("cpu"),
                ),
                input_shape=(3, 32, 32),
                bs=2,
            )
        except Exception as exc:
            value = {
                "fwd_flops_per_img": None,
                "train_flops_per_img": None,
                "params": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        self._flops_cache[key] = value
        return value

    def run(
        self,
        *,
        candidate_id: str,
        params: dict[str, Any],
        resource: TrialResource,
        trial_number: int | None = None,
        maximum_total_epochs: int | None = None,
        reporter: Reporter | None = None,
    ) -> TrialResult:
        started = time.time()
        checkpoint = self._checkpoint_path(candidate_id)
        runtime_for_validation = dict(self.runtime_config)
        runtime_for_validation["device"] = self.device.type
        try:
            validate_candidate(params, self.model_config, runtime_for_validation)
        except InvalidTrialError as exc:
            return TrialResult(
                candidate_id,
                trial_number,
                resource.stage,
                "invalid",
                dict(params),
                {},
                asdict(resource),
                None,
                invalid_reason=str(exc),
                started_at=started,
                finished_at=time.time(),
            )

        model_config = self._model_config(params)
        model_name = str(model_config.get("name", "resnet18"))
        batch_size = int(params.get("batch_size", 128))
        accumulation = int(params.get("gradient_accumulation", 1))
        train_loop_parameters = inspect.signature(
            self.modules.train_loop.train_one_epoch
        ).parameters
        accumulation_supported = "gradient_accumulation" in train_loop_parameters
        if accumulation != 1 and not accumulation_supported:
            return TrialResult(
                candidate_id,
                trial_number,
                resource.stage,
                "invalid",
                dict(params),
                {},
                asdict(resource),
                None,
                invalid_reason=(
                    "gradient_accumulation > 1 requires the documented "
                    "train_loop.py compatibility patch"
                ),
                started_at=started,
                finished_at=time.time(),
            )

        dataset = self.dataset
        maximum_examples = None
        if resource.max_steps is not None:
            steps_per_epoch = max(1, math.ceil(resource.max_steps / max(1, resource.target_epochs)))
            maximum_examples = steps_per_epoch * batch_size
        train_ds = subset_view(
            dataset.train,
            resource.data_fraction,
            seed=resource.seed,
            maximum_examples=maximum_examples,
        )
        validation_ds = subset_view(
            dataset.validation,
            resource.validation_fraction,
            seed=resource.seed + 10_000,
        )

        torch.manual_seed(resource.seed)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed_all(resource.seed)
            torch.backends.cuda.matmul.allow_tf32 = bool(self.runtime_config.get("tf32", True))
            torch.backends.cudnn.allow_tf32 = bool(self.runtime_config.get("tf32", True))
            torch.backends.cudnn.benchmark = bool(self.runtime_config.get("cudnn_benchmark", True))
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        model = build_trial_model(self.modules, model_config, dataset.num_classes, self.device)
        is_cnn = model_name.startswith("resnet")
        channels_last = bool(
            params.get(
                "channels_last",
                self.runtime_config.get("channels_last", is_cnn and self.device.type == "cuda"),
            )
        )
        if channels_last and not is_cnn:
            raise InvalidTrialError("channels_last is restricted to CNNs")
        if channels_last:
            model = model.to(memory_format=torch.channels_last)

        compile_enabled = bool(params.get("compile", self.runtime_config.get("compile", False)))
        if compile_enabled and hasattr(torch, "compile"):
            model = torch.compile(model, mode=self.runtime_config.get("compile_mode"))

        optimizer = build_trial_optimizer(model, self._optimizer_config(params, model_name))
        total_epochs = int(maximum_total_epochs or resource.target_epochs)
        scheduler = build_trial_scheduler(
            optimizer,
            self._scheduler_config(params),
            max(1, total_epochs),
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=float(params.get("label_smoothing", 0.1 if is_cnn else 0.0))
        )
        use_amp = self.device.type == "cuda" and self.precision in {"fp16", "bf16"}
        amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.device.type == "cuda" and self.precision == "fp16",
        )
        start_epoch, best, history = 1, 0.0, []
        if resource.continue_checkpoint and checkpoint.exists():
            epoch, best, history = self.modules.train_loop.load_checkpoint(
                str(checkpoint),
                model,
                optimizer,
                scheduler,
                scaler,
                self.device,
            )
            start_epoch = int(epoch) + 1

        strong_aug = bool(params.get("strong_augmentation", params.get("augmentation", "strong") == "strong"))
        if "strong_augmentation" not in params and "augmentation" not in params:
            strong_aug = model_name.startswith("vit")
        clip = float(params.get("gradient_clip", 0.0 if model_name == "resnet18" else 1.0))
        epoch_metrics: list[dict[str, Any]] = list(history)
        history_length_at_start = len(epoch_metrics)
        process = psutil.Process()
        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        examples_processed = 0
        optimization_steps = 0
        status = "completed"
        prune_reason = None

        try:
            for epoch in range(start_epoch, resource.target_epochs + 1):
                train_metrics = self.modules.train_loop.train_one_epoch(
                    model,
                    train_ds,
                    optimizer,
                    criterion,
                    scaler,
                    self.device,
                    batch_size,
                    epoch,
                    total_epochs,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                    strong_aug=strong_aug,
                    clip=clip or None,
                    **(
                        {"gradient_accumulation": accumulation}
                        if accumulation_supported
                        else {}
                    ),
                )
                validation_metrics = self.modules.train_loop.evaluate(
                    model,
                    validation_ds,
                    criterion,
                    self.device,
                    batch_size=max(batch_size, min(1024, validation_ds.n)),
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    channels_last=channels_last,
                )
                scheduler.step()
                best = max(best, float(validation_metrics["top1"]))
                examples_processed += train_ds.n
                optimization_steps += math.ceil(
                    math.ceil(train_ds.n / batch_size) / accumulation
                )
                record = {
                    "epoch": epoch,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "best_validation_top1": best,
                }
                epoch_metrics.append(record)
                self.modules.train_loop.save_checkpoint(
                    str(checkpoint),
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best,
                    epoch_metrics,
                )
                if reporter and reporter(epoch, record):
                    status = "pruned"
                    prune_reason = f"reporter pruned at epoch {epoch}"
                    break
        except torch.cuda.OutOfMemoryError as exc:
            status = "oom"
            failure = str(exc)
        except (RuntimeError, ValueError) as exc:
            status = "divergent" if "nan" in str(exc).lower() else "failed"
            failure = f"{type(exc).__name__}: {exc}"
        else:
            failure = None

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        final_validation = epoch_metrics[-1].get("validation", {}) if epoch_metrics else {}
        test_metrics = None
        if status == "completed" and resource.evaluate_test and dataset.test is not None:
            test_metrics = self.modules.train_loop.evaluate(
                model,
                dataset.test,
                criterion,
                self.device,
                batch_size=max(batch_size, min(1024, dataset.test.n)),
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                channels_last=channels_last,
            )

        flops = self._flops(model_config, dataset.num_classes)
        peak_gpu = (
            torch.cuda.max_memory_allocated(self.device) / 2**20
            if self.device.type == "cuda"
            else 0.0
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved(self.device) / 2**20
            if self.device.type == "cuda"
            else 0.0
        )
        checkpoint_mb = checkpoint.stat().st_size / 2**20 if checkpoint.exists() else 0.0
        train_flops = flops.get("train_flops_per_img")
        metrics = {
            "validation_top1": float(final_validation.get("top1", 0.0)),
            "validation_top5": float(final_validation.get("top5", 0.0)),
            "validation_loss": float(final_validation.get("loss", float("inf"))),
            "best_validation_top1": float(best),
            "test_top1": None if test_metrics is None else float(test_metrics["top1"]),
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "gpu_hours": wall_seconds / 3600 if self.device.type == "cuda" else 0.0,
            "cpu_hours": cpu_seconds / 3600,
            "epochs_completed": len(epoch_metrics),
            "optimization_steps": optimization_steps,
            "training_examples": examples_processed,
            "validation_examples": validation_ds.n * max(0, len(epoch_metrics) - history_length_at_start),
            "total_examples": examples_processed + validation_ds.n * max(0, len(epoch_metrics) - history_length_at_start),
            "parameter_count": flops.get("params") or sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_flops_per_image": train_flops,
            "approximate_training_flops": None if train_flops is None else train_flops * examples_processed,
            "peak_gpu_memory_mb": peak_gpu,
            "peak_gpu_reserved_mb": peak_reserved,
            "process_rss_mb": process.memory_info().rss / 2**20,
            "checkpoint_size_mb": checkpoint_mb,
            "throughput_examples_per_second": examples_processed / wall_seconds if wall_seconds else 0.0,
            "device": str(self.device),
            "precision": self.precision,
            "channels_last": channels_last,
            "data_strategy": dataset.strategy,
            "history": epoch_metrics,
            "flop_measurement": "measured by repository perfkit FlopCounterMode" if train_flops else "unavailable",
        }

        result = TrialResult(
            candidate_id=candidate_id,
            trial_number=trial_number,
            stage=resource.stage,
            status=status,
            params=dict(params),
            metrics=metrics,
            resource=asdict(resource),
            checkpoint_path=str(checkpoint) if checkpoint.exists() else None,
            failure_reason=failure,
            prune_reason=prune_reason,
            started_at=started,
            finished_at=time.time(),
        )
        del model, optimizer, scheduler, scaler
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

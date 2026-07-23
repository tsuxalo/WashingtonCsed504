from __future__ import annotations

import dataclasses
import json
import math
import os
import signal
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import optuna
from optuna.trial import TrialState

from .baselines import Baseline, enqueue_reference
from .config import load_study_config
from .constraints import check_hard_constraints
from .costing import estimate_cost
from .hardware import detect_hardware
from .modes import full_resources, halving_rungs, promote, proxy_resource, rung_resource
from .objectives import objective_values, primary_score
from .persistence import StudyFiles, atomic_write_json, read_jsonl, stable_hash
from .parallel import ParallelTrialExecutor
from .reporting import export_reports
from .scheduler import plan_resources
from .schemas import StudyConfig
from .search_space import enumerate_combinations, suggest
from .selection import pareto_front
from .trial_runner import TrialResource, TrialResult, TrialRunner


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class HpoStudy:
    def __init__(self, config: StudyConfig | str | Path | dict[str, Any], *, repo_root: str | Path):
        self.config = config if isinstance(config, StudyConfig) else load_study_config(config)
        self.repo_root = Path(repo_root).resolve()
        self.study_dir = self.config.output_dir / self.config.name
        self.files = StudyFiles(self.study_dir)
        self.hardware = detect_hardware()
        self.resource_plan = plan_resources(
            self.hardware,
            device=self.config.runtime.device,
            requested_concurrency=self.config.runtime.concurrent_trials,
            requested_intraop_threads=self.config.runtime.intraop_threads,
            requested_interop_threads=self.config.runtime.interop_threads,
            requested_workers=self.config.runtime.workers,
            memory_reserve_gb=self.config.runtime.memory_reserve_gb,
        )
        runtime = _jsonable(self.config.runtime)
        runtime.update({
            "device": self.resource_plan.device,
            "intraop_threads": self.resource_plan.intraop_threads,
            "interop_threads": self.resource_plan.interop_threads,
            "workers": self.resource_plan.workers_per_trial,
            "seed": self.config.seed,
        })
        self.runner = TrialRunner(
            self.repo_root,
            dataset_config=self.config.dataset,
            model_config=self.config.model,
            runtime_config=runtime,
            output_dir=self.study_dir,
            hardware=self.hardware,
        )
        self.state = self.files.load_state()
        self.state.setdefault("candidates", {})
        self.state.setdefault("completed_param_hashes", [])
        self.state.setdefault("run_count", 0)
        self.state.setdefault("completed_modes", {})
        self._stop_requested = False
        self._parallel_executor: ParallelTrialExecutor | None = None
        self.optuna_study = self._create_optuna_study()
        self._initialize_files()

    def _sampler(self):
        restored = self.files.load_sampler() if self.config.resume else None
        if restored is not None:
            return restored
        name = self.config.sampler.lower()
        multi = len(self.config.objectives) > 1
        if name == "random":
            return optuna.samplers.RandomSampler(seed=self.config.seed)
        if name in {"qmc", "sobol"}:
            return optuna.samplers.QMCSampler(seed=self.config.seed, qmc_type="sobol")
        if name in {"nsga2", "nsga-ii"} or multi:
            return optuna.samplers.NSGAIISampler(seed=self.config.seed)
        return optuna.samplers.TPESampler(seed=self.config.seed, multivariate=True)

    def _create_optuna_study(self):
        self.config.storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{self.config.storage_path}",
            engine_kwargs={"connect_args": {"timeout": 60}},
        )
        directions = [objective.direction for objective in self.config.objectives]
        kwargs: dict[str, Any] = {
            "study_name": self.config.name,
            "storage": storage,
            "sampler": self._sampler(),
            "load_if_exists": self.config.resume,
        }
        if len(directions) == 1:
            kwargs["direction"] = directions[0]
        else:
            kwargs["directions"] = directions
        return optuna.create_study(**kwargs)

    def _initialize_files(self):
        atomic_write_json(self.files.environment, {
            "hardware": self.hardware.to_dict(),
            "resource_plan": self.resource_plan.to_dict(),
            "repository": str(self.repo_root),
            "git_revision": self._git_revision(),
            "created_at": time.time(),
        })
        atomic_write_json(self.files.resolved_config, _jsonable(self.config))
        self._validate_resume_hashes()
        self.files.record_event("study_initialized", mode=self.config.mode)

    def _git_revision(self) -> str | None:
        import subprocess
        try:
            return subprocess.check_output(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return None

    def _validate_resume_hashes(self):
        config_hash = stable_hash(_jsonable(self.config))
        space_hash = stable_hash([spec.to_dict() for spec in self.config.search_space])
        previous_config = self.state.get("configuration_hash")
        previous_space = self.state.get("search_space_hash")
        if self.config.resume and previous_config and previous_config != config_hash:
            raise RuntimeError("persisted study configuration hash does not match requested configuration")
        if self.config.resume and previous_space and previous_space != space_hash:
            raise RuntimeError("persisted search-space hash does not match requested search space")
        self.state["configuration_hash"] = config_hash
        self.state["search_space_hash"] = space_hash
        self._save_state()

    def _save_state(self):
        self.files.save_state(self.state)
        self.files.save_sampler(self.optuna_study.sampler)

    def enqueue_reference(self, baseline: Baseline) -> None:
        enqueue_reference(self.optuna_study, baseline)

    def _candidate_from_trial(self, trial: optuna.Trial, params: dict[str, Any]) -> dict[str, Any]:
        param_hash = stable_hash(params)
        return {
            "candidate_id": f"trial-{trial.number:05d}",
            "trial_number": trial.number,
            "params": params,
            "param_hash": param_hash,
            "status": "pending",
            "last_stage": None,
            "last_result": None,
            "created_at": time.time(),
        }

    def _new_candidates(self, count: int) -> list[dict[str, Any]]:
        existing_hashes = set(self.state.get("completed_param_hashes", [])) | {
            item.get("param_hash") for item in self.state["candidates"].values()
        }
        result = []
        attempts = 0
        while len(result) < count:
            attempts += 1
            if attempts > max(100, count * 50):
                raise RuntimeError("sampler repeatedly generated duplicate configurations")
            trial = self.optuna_study.ask()
            params = suggest(trial, self.config.search_space)
            param_hash = stable_hash(params)
            if param_hash in existing_hashes:
                self.optuna_study.tell(trial, state=TrialState.FAIL)
                continue
            candidate = self._candidate_from_trial(trial, params)
            self.state["candidates"][candidate["candidate_id"]] = candidate
            existing_hashes.add(param_hash)
            result.append(candidate)
            self._save_state()
        return result

    def _pending_running_candidates(self) -> list[dict[str, Any]]:
        return [
            item for item in self.state["candidates"].values()
            if item.get("status") in {"pending", "running", "promoted"}
        ]

    def _record(self, candidate: dict[str, Any], result: TrialResult) -> dict[str, Any]:
        row = result.to_dict()
        cost = estimate_cost(row["metrics"], self.config.cost_rates)
        row["metrics"].update(cost)
        violations = check_hard_constraints(row["metrics"], self.config.constraints)
        row["constraint_violations"] = violations
        if violations and row["status"] == "completed":
            row["status"] = "constraint_violated"
        self.files.record_result(row)
        candidate["last_stage"] = row["stage"]
        candidate["last_result"] = row
        candidate["status"] = row["status"]
        candidate.setdefault("results", []).append(row)
        self.files.record_event(
            "trial_result",
            candidate_id=candidate["candidate_id"],
            trial_number=candidate["trial_number"],
            stage=row["stage"],
            status=row["status"],
            metrics={key: value for key, value in row["metrics"].items() if key != "history"},
        )
        self._save_state()
        return row

    def _record_seed_result(self, candidate: dict[str, Any], result: TrialResult) -> dict[str, Any]:
        """Persist an individual seed run without treating it as a final candidate."""
        row = result.to_dict()
        row["metrics"].update(estimate_cost(row["metrics"], self.config.cost_rates))
        violations = check_hard_constraints(row["metrics"], self.config.constraints)
        row["constraint_violations"] = violations
        if row["status"] == "completed":
            row["status"] = "seed_constraint_violated" if violations else "seed_completed"
        self.files.record_result(row)
        self.files.record_event(
            "seed_result",
            candidate_id=candidate["candidate_id"],
            run_id=row["candidate_id"],
            trial_number=candidate["trial_number"],
            stage=row["stage"],
            status=row["status"],
            seed=row.get("resource", {}).get("seed"),
        )
        return row

    def _aggregate_seed_rows(
        self,
        candidate: dict[str, Any],
        seed_rows: list[dict[str, Any]],
        *,
        stage: str,
    ) -> dict[str, Any] | None:
        completed = [row for row in seed_rows if row.get("status") == "seed_completed"]
        if not completed:
            return None
        aggregate = dict(completed[-1])
        aggregate["candidate_id"] = candidate["candidate_id"]
        aggregate["stage"] = stage
        aggregate["status"] = "completed"
        aggregate["failure_reason"] = None
        aggregate["invalid_reason"] = None
        aggregate["prune_reason"] = None
        aggregate["metrics"] = self._aggregate_metrics([row["metrics"] for row in completed])
        aggregate["metrics"]["seed_repetitions"] = len(completed)
        aggregate["metrics"]["accounting_scope"] = "all_seed_runs"
        self.files.record_result(aggregate)
        self.files.record_event(
            "seed_aggregate",
            candidate_id=candidate["candidate_id"],
            stage=stage,
            seed_repetitions=len(completed),
        )
        candidate["last_result"] = aggregate
        candidate["status"] = "completed"
        self._save_state()
        return aggregate

    def _tell_completed(self, candidate: dict[str, Any], row: Mapping[str, Any]) -> None:
        values = objective_values(row["metrics"], self.config.objectives)
        value: float | list[float] = values[0] if len(values) == 1 else values
        self.optuna_study.tell(candidate["trial_number"], value)
        candidate["status"] = "completed"
        self.state["completed_param_hashes"].append(candidate["param_hash"])
        self._save_state()

    def _tell_pruned(self, candidate: dict[str, Any], reason: str) -> None:
        try:
            self.optuna_study.tell(candidate["trial_number"], state=TrialState.PRUNED, skip_if_finished=True)
        except TypeError:
            self.optuna_study.tell(candidate["trial_number"], state=TrialState.PRUNED)
        candidate["status"] = "pruned"
        candidate["prune_reason"] = reason
        self.state["completed_param_hashes"].append(candidate["param_hash"])
        self.files.record_event("candidate_pruned", candidate_id=candidate["candidate_id"], reason=reason)
        self._save_state()

    def _tell_failed(self, candidate: dict[str, Any], reason: str) -> None:
        try:
            self.optuna_study.tell(candidate["trial_number"], state=TrialState.FAIL, skip_if_finished=True)
        except TypeError:
            self.optuna_study.tell(candidate["trial_number"], state=TrialState.FAIL)
        candidate["status"] = "failed"
        candidate["failure_reason"] = reason
        self.state["completed_param_hashes"].append(candidate["param_hash"])
        self._save_state()

    def _get_parallel_executor(self) -> ParallelTrialExecutor | None:
        concurrency = self.resource_plan.concurrent_trials
        if concurrency <= 1:
            return None
        if self._parallel_executor is None:
            if self.resource_plan.device == "cuda":
                indices: list[int | None] = list(range(min(concurrency, self.hardware.gpu_count)))
            elif self.resource_plan.device == "cpu":
                indices = [None] * concurrency
            else:
                return None
            runtime = _jsonable(self.config.runtime)
            runtime.update({
                "intraop_threads": self.resource_plan.intraop_threads,
                "interop_threads": self.resource_plan.interop_threads,
                "workers": self.resource_plan.workers_per_trial,
                "seed": self.config.seed,
            })
            self._parallel_executor = ParallelTrialExecutor(
                repo_root=self.repo_root,
                dataset_config=self.config.dataset,
                model_config=self.config.model,
                runtime_config=runtime,
                output_dir=self.study_dir,
                device_type=self.resource_plan.device,
                device_indices=indices,
            )
        return self._parallel_executor

    def _run_many(
        self,
        items: list[tuple[dict[str, Any], TrialResource, str]],
        max_epochs: int,
    ) -> list[dict[str, Any]]:
        executor = self._get_parallel_executor()
        if executor is None or len(items) <= 1:
            rows = []
            for candidate, resource, run_id in items:
                if run_id == candidate["candidate_id"]:
                    rows.append(self._run_candidate(candidate, resource, max_epochs))
                else:
                    result = self.runner.run(
                        candidate_id=run_id,
                        trial_number=candidate["trial_number"],
                        params=candidate["params"],
                        resource=resource,
                        maximum_total_epochs=max_epochs,
                    )
                    rows.append(self._record(candidate, result))
            return rows

        tasks = []
        candidate_by_task = {}
        for index, (candidate, resource, run_id) in enumerate(items):
            task_id = f"batch-{time.time_ns()}-{index}"
            candidate["status"] = "running"
            candidate_by_task[task_id] = candidate
            tasks.append({
                "task_id": task_id,
                "candidate_id": run_id,
                "trial_number": candidate["trial_number"],
                "params": candidate["params"],
                "resource": dataclasses.asdict(resource),
                "maximum_total_epochs": max_epochs,
            })
        self._save_state()
        messages = executor.run(tasks)
        rows = []
        for message in messages:
            candidate = candidate_by_task[message["task_id"]]
            if message.get("error"):
                result = TrialResult(
                    candidate_id=candidate["candidate_id"],
                    trial_number=candidate["trial_number"],
                    stage="worker",
                    status="failed",
                    params=candidate["params"],
                    metrics={},
                    resource={},
                    checkpoint_path=None,
                    failure_reason=message["error"],
                    started_at=time.time(),
                    finished_at=time.time(),
                )
            else:
                result = TrialResult(**message["result"])
            rows.append(self._record(candidate, result))
        return rows

    def _run_candidate(self, candidate: dict[str, Any], resource: TrialResource, max_epochs: int) -> dict[str, Any]:
        candidate["status"] = "running"
        self._save_state()
        result = self.runner.run(
            candidate_id=candidate["candidate_id"],
            trial_number=candidate["trial_number"],
            params=candidate["params"],
            resource=resource,
            maximum_total_epochs=max_epochs,
        )
        return self._record(candidate, result)

    def run_proxy(self, *, trials: int | None = None) -> dict[str, Any]:
        count = trials or self.config.proxy.trials
        candidates = self._pending_running_candidates()
        if len(candidates) < count:
            candidates += self._new_candidates(count - len(candidates))
        rows = []
        max_epochs = max(
            [int(self.config.proxy.budget.epochs or 1)]
            + [int(item) for item in self.config.successive_halving.rung_budgets]
            + [int(self.config.full.budget.epochs or 1)]
        )
        selected = candidates[:count]
        rows = self._run_many(
            [(candidate, proxy_resource(self.config), candidate["candidate_id"]) for candidate in selected],
            max_epochs,
        )
        for candidate, row in zip(selected, rows, strict=True):
            if row["status"] == "completed" and not row.get("constraint_violations"):
                self._tell_completed(candidate, row)
            elif row["status"] in {"invalid", "failed", "oom", "divergent", "constraint_violated"}:
                self._tell_failed(candidate, row.get("failure_reason") or row.get("invalid_reason") or "constraint violation")
            else:
                self._tell_pruned(candidate, row.get("prune_reason") or "proxy pruning")
        return self._finalize("proxy", rows)

    def run_successive_halving(self, *, trials: int | None = None) -> dict[str, Any]:
        count = trials or self.config.proxy.trials
        candidates = self._pending_running_candidates()
        if len(candidates) < count:
            candidates += self._new_candidates(count - len(candidates))
        candidates = candidates[:count]
        max_epochs = max(
            [int(self.config.proxy.budget.epochs or 1)]
            + [int(item) for item in self.config.successive_halving.rung_budgets if self.config.successive_halving.resource_type == "epochs"]
            + [int(self.config.full.budget.epochs or 1)]
        )

        proxy_rows = []
        to_run = []
        for candidate in candidates:
            if candidate.get("last_stage") == "proxy" and candidate.get("last_result"):
                proxy_rows.append(candidate["last_result"])
            else:
                to_run.append(candidate)
        new_proxy_rows = self._run_many(
            [(candidate, proxy_resource(self.config), candidate["candidate_id"]) for candidate in to_run],
            max_epochs,
        )
        proxy_rows.extend(new_proxy_rows)
        for candidate, row in zip(to_run, new_proxy_rows, strict=True):
            if row["status"] != "completed" or row.get("constraint_violations"):
                self._tell_failed(candidate, row.get("failure_reason") or row.get("invalid_reason") or "proxy failed")

        ranked = [row for row in proxy_rows if row.get("status") == "completed" and not row.get("constraint_violations")]
        ranked.sort(key=lambda row: primary_score(row["metrics"], self.config.primary_objective), reverse=True)
        active_ids = {row["candidate_id"] for row in ranked[: self.config.proxy.promote_top_k]}
        active = [candidate for candidate in candidates if candidate["candidate_id"] in active_ids]
        for candidate in candidates:
            if candidate not in active and candidate.get("status") not in {"failed", "pruned"}:
                self._tell_pruned(candidate, "not selected after proxy screening")
        all_rows = list(proxy_rows)

        for rung in halving_rungs(self.config, len(active)):
            if self._stop_requested or not active:
                break
            rung_rows = []
            if self.config.successive_halving.resource_type != "seeds":
                rung_rows = self._run_many(
                    [
                        (
                            candidate,
                            rung_resource(self.config, rung, self.config.seed),
                            candidate["candidate_id"],
                        )
                        for candidate in active
                    ],
                    max_epochs,
                )
                all_rows.extend(rung_rows)
            else:
                for candidate in active:
                    seed_count = max(1, int(rung.budget))
                    seed_rows = []
                    for seed_index in range(seed_count):
                        resource = rung_resource(
                            self.config,
                            rung,
                            self.config.seed + seed_index,
                        )
                        result = self.runner.run(
                            candidate_id=f"{candidate['candidate_id']}-seed{seed_index}",
                            trial_number=candidate["trial_number"],
                            params=candidate["params"],
                            resource=resource,
                            maximum_total_epochs=max_epochs,
                        )
                        seed_rows.append(self._record_seed_result(candidate, result))
                    row = self._aggregate_seed_rows(candidate, seed_rows, stage="halving")
                    if row is None:
                        row = seed_rows[-1]
                    rung_rows.append(row)
                    all_rows.append(row)

            promoted_rows, eliminated_rows = promote(rung_rows, rung.survivors, self.config.primary_objective)
            promoted_ids = {row["candidate_id"] for row in promoted_rows}
            next_active = []
            for candidate in active:
                if candidate["candidate_id"] in promoted_ids:
                    candidate["status"] = "promoted"
                    next_active.append(candidate)
                    self.files.record_event(
                        "candidate_promoted",
                        candidate_id=candidate["candidate_id"],
                        rung=rung.index,
                        budget=rung.budget,
                    )
                else:
                    self._tell_pruned(candidate, f"eliminated at rung {rung.index} budget {rung.budget}")
            active = next_active
            self._save_state()

        if self.config.full.enabled and active:
            full_rows = []
            for candidate in active[: self.config.full.trials]:
                seed_results = []
                for resource in full_resources(self.config):
                    # Confirmation runs restart independently for each seed.
                    resource.continue_checkpoint = False
                    seed_id = f"{candidate['candidate_id']}-full-seed{resource.seed}"
                    result = self.runner.run(
                        candidate_id=seed_id,
                        trial_number=candidate["trial_number"],
                        params=candidate["params"],
                        resource=resource,
                        maximum_total_epochs=resource.target_epochs,
                    )
                    seed_results.append(self._record_seed_result(candidate, result))
                aggregate = self._aggregate_seed_rows(candidate, seed_results, stage="full")
                if aggregate is not None:
                    self._tell_completed(candidate, aggregate)
                    full_rows.append(aggregate)
                    all_rows.append(aggregate)
                else:
                    self._tell_failed(candidate, "all full-confirmation seed runs failed")
            active = []
        else:
            for candidate in active:
                row = candidate.get("last_result")
                if row and row.get("status") == "completed":
                    self._tell_completed(candidate, row)

        return self._finalize("successive_halving", all_rows)

    def _aggregate_metrics(self, metrics: list[Mapping[str, Any]]) -> dict[str, Any]:
        numeric_keys = {
            key
            for row in metrics
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        additive = {
            "wall_seconds", "cpu_seconds", "gpu_hours", "cpu_hours",
            "epochs_completed", "optimization_steps", "training_examples",
            "validation_examples", "total_examples", "approximate_training_flops",
            "checkpoint_size_mb", "known_component_total_usd", "estimated_cost_usd",
        }
        maxima = {
            "peak_gpu_memory_mb", "peak_gpu_reserved_mb", "process_rss_mb",
            "parameter_count",
        }
        result: dict[str, Any] = {}
        for key in numeric_keys:
            values = [float(row[key]) for row in metrics if row.get(key) is not None]
            if not values:
                continue
            if key in additive:
                result[key] = sum(values)
            elif key in maxima:
                result[key] = max(values)
            else:
                result[key] = sum(values) / len(values)
            result[f"{key}_min"] = min(values)
            result[f"{key}_max"] = max(values)
        result["seed_metrics"] = metrics
        return result

    def run_full(self, *, trials: int | None = None) -> dict[str, Any]:
        count = trials or self.config.full.trials
        if self.config.full.exhaustive:
            combinations = enumerate_combinations(
                self.config.search_space,
                limit=self.config.full.maximum_combinations,
            )
            for params in combinations:
                self.optuna_study.enqueue_trial(params)
            count = len(combinations)
        candidates = self._pending_running_candidates()
        if len(candidates) < count:
            candidates += self._new_candidates(count - len(candidates))
        rows = []
        for candidate in candidates[:count]:
            seed_results = []
            for resource in full_resources(self.config):
                seed_id = f"{candidate['candidate_id']}-seed{resource.seed}"
                result = self.runner.run(
                    candidate_id=seed_id,
                    trial_number=candidate["trial_number"],
                    params=candidate["params"],
                    resource=resource,
                    maximum_total_epochs=resource.target_epochs,
                )
                seed_results.append(self._record_seed_result(candidate, result))
            aggregate = self._aggregate_seed_rows(candidate, seed_results, stage="full")
            if aggregate is not None:
                rows.append(aggregate)
                self._tell_completed(candidate, aggregate)
            else:
                self._tell_failed(candidate, "all full-fidelity seed runs failed")
        return self._finalize("full", rows)

    def run(self) -> dict[str, Any]:
        if (
            self.config.resume
            and not self.config.continuous.enabled
            and self.state.get("completed_modes", {}).get(self.config.mode)
            and self.files.summary.exists()
        ):
            return json.loads(self.files.summary.read_text(encoding="utf-8"))
        self.state["run_count"] += 1
        self._save_state()
        old_handler = signal.getsignal(signal.SIGINT)

        def stop(_sig, _frame):
            self._stop_requested = True
            self.files.record_event("interrupt_requested")

        signal.signal(signal.SIGINT, stop)
        try:
            if self.config.continuous.enabled:
                return self.run_continuous()
            if self.config.mode == "proxy":
                return self.run_proxy()
            if self.config.mode == "full":
                return self.run_full()
            return self.run_successive_halving()
        finally:
            signal.signal(signal.SIGINT, old_handler)
            if self._parallel_executor is not None:
                self._parallel_executor.close()
                self._parallel_executor = None
            self._save_state()

    def run_continuous(self) -> dict[str, Any]:
        started = time.time()
        best = -math.inf
        no_improvement = 0
        previous_front_hash = None
        pareto_stagnation = 0
        sessions = []
        while not self._stop_requested:
            if self._continuous_stop(started, best, no_improvement, pareto_stagnation):
                break
            before_finalized = len(set(self.state.get("completed_param_hashes", [])))
            remaining = None
            if self.config.continuous.maximum_trials is not None:
                remaining = max(0, self.config.continuous.maximum_trials - before_finalized)
                if remaining == 0:
                    break
            if self.config.mode == "proxy":
                session_trials = 1
                summary = self.run_proxy(trials=session_trials)
            elif self.config.mode == "full":
                session_trials = 1
                summary = self.run_full(trials=session_trials)
            else:
                session_trials = self.config.proxy.trials
                if remaining is not None:
                    session_trials = max(1, min(session_trials, remaining))
                summary = self.run_successive_halving(trials=session_trials)
            sessions.append(summary)
            after_finalized = len(set(self.state.get("completed_param_hashes", [])))
            newly_finalized = max(1, after_finalized - before_finalized)
            rows = [
                row for row in read_jsonl(self.files.results)
                if row.get("status") == "completed" and row.get("metrics")
            ]
            scores = [primary_score(row["metrics"], self.config.primary_objective) for row in rows]
            current_best = max(scores, default=-math.inf)
            if current_best >= best + self.config.continuous.minimum_improvement:
                best = current_best
                no_improvement = 0
            else:
                no_improvement += newly_finalized
            front = pareto_front([row["metrics"] for row in rows], self.config.objectives)
            front_hash = stable_hash(front)
            if front_hash == previous_front_hash:
                pareto_stagnation += newly_finalized
            else:
                pareto_stagnation = 0
                previous_front_hash = front_hash
        return self._finalize("continuous", sessions)

    def _continuous_stop(self, started: float, best: float, no_improvement: int, pareto_stagnation: int) -> bool:
        cfg = self.config.continuous
        records = read_jsonl(self.files.results)
        completed = [row for row in records if row.get("status") == "completed"]
        wall_hours = (time.time() - started) / 3600
        gpu_hours = sum(float(row.get("metrics", {}).get("gpu_hours", 0) or 0) for row in completed)
        cpu_hours = sum(float(row.get("metrics", {}).get("cpu_hours", 0) or 0) for row in completed)
        cost = sum(float(row.get("metrics", {}).get("known_component_total_usd", 0) or 0) for row in completed)
        conditions = {
            "maximum_trials": cfg.maximum_trials is not None and len(set(self.state.get("completed_param_hashes", []))) >= cfg.maximum_trials,
            "maximum_wall_time_hours": cfg.maximum_wall_time_hours is not None and wall_hours >= cfg.maximum_wall_time_hours,
            "maximum_gpu_hours": cfg.maximum_gpu_hours is not None and gpu_hours >= cfg.maximum_gpu_hours,
            "maximum_cpu_hours": cfg.maximum_cpu_hours is not None and cpu_hours >= cfg.maximum_cpu_hours,
            "maximum_cost_usd": cfg.maximum_cost_usd is not None and cost >= cfg.maximum_cost_usd,
            "target_validation_metric": cfg.target_validation_metric is not None and best >= cfg.target_validation_metric,
            "no_improvement": cfg.stop_after_no_improvement_trials is not None and no_improvement >= cfg.stop_after_no_improvement_trials,
            "pareto_stagnation": cfg.pareto_stagnation_trials is not None and pareto_stagnation >= cfg.pareto_stagnation_trials,
        }
        hit = [name for name, value in conditions.items() if value]
        if hit:
            self.files.record_event("continuous_stop", conditions=hit)
            return True
        return False

    def _export_optuna_analysis(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        try:
            dataframe = self.optuna_study.trials_dataframe()
            dataframe.to_csv(self.study_dir / "optuna_trials.csv", index=False)
            output["optuna_trials_csv"] = str(self.study_dir / "optuna_trials.csv")
        except Exception as exc:
            output["trials_dataframe_error"] = f"{type(exc).__name__}: {exc}"
        completed = [trial for trial in self.optuna_study.trials if trial.state == TrialState.COMPLETE]
        # Parameter importance from one or two trials is not meaningful and
        # some fANOVA backends can spend disproportionate time fitting tiny,
        # degenerate studies. Require a minimally informative sample instead.
        if len(completed) >= 5:
            try:
                from optuna.importance import get_param_importances
                if len(self.config.objectives) == 1:
                    importance = get_param_importances(self.optuna_study)
                else:
                    primary_index = next(
                        (index for index, objective in enumerate(self.config.objectives) if objective.primary),
                        0,
                    )
                    importance = get_param_importances(
                        self.optuna_study,
                        target=lambda trial: trial.values[primary_index] if trial.values else float("nan"),
                    )
                atomic_write_json(self.study_dir / "parameter_importance.json", importance)
                output["parameter_importance"] = importance
            except Exception as exc:
                output["parameter_importance_error"] = f"{type(exc).__name__}: {exc}"
        else:
            output["parameter_importance_status"] = (
                "skipped: at least 5 completed Optuna trials are required"
            )
        history = []
        for trial in completed:
            history.append({
                "trial_number": trial.number,
                "values": trial.values,
                "params": trial.params,
                "datetime_start": None if trial.datetime_start is None else trial.datetime_start.isoformat(),
                "datetime_complete": None if trial.datetime_complete is None else trial.datetime_complete.isoformat(),
            })
        atomic_write_json(self.study_dir / "optimization_history.json", history)
        output["completed_optuna_trials"] = len(completed)
        return output

    def _finalize(self, mode: str, rows: Any) -> dict[str, Any]:
        report = export_reports(self.study_dir, self.config.objectives)
        optuna_analysis = self._export_optuna_analysis()
        records = read_jsonl(self.files.results)
        summary = {
            "study": self.config.name,
            "mode": mode,
            "study_dir": str(self.study_dir),
            "storage_path": str(self.config.storage_path),
            "records": len(records),
            "status_counts": dict(Counter(str(row.get("status")) for row in records)),
            "candidate_status_counts": dict(
                Counter(
                    str(candidate.get("status"))
                    for candidate in self.state.get("candidates", {}).values()
                )
            ),
            "report": report,
            "optuna_analysis": optuna_analysis,
            "interrupted": self._stop_requested,
            "updated_at": time.time(),
        }
        atomic_write_json(self.files.summary, summary)
        self.files.record_event("study_finalized", summary=summary)
        if mode != "continuous" and not self._stop_requested:
            self.state.setdefault("completed_modes", {})[mode] = True
            self._save_state()
        return summary

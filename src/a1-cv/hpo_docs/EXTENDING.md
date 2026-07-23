# Extending Models, Datasets, and Metrics

## Models

The adapter first calls the existing `models.build(name, num_classes)`. The existing `make_vit` builder is used for supported custom ViT dimensions. To add a model:

1. Add or register it in the repository's existing model zoo.
2. Add architecture validation in `hpo/constraints.py` only for real builder constraints.
3. Add a configuration and adapter test.

Do not duplicate the full builder in the HPO package.

## Datasets

`build_trial_dataset` reuses the existing resident-device CIFAR and ImageNet-32 APIs. For a new dataset, return a `DatasetBundle` with train, validation, optional test, class count, example counts, strategy, and metadata. Tuning data must be distinct from final test data.

## Metrics and objectives

Add measured fields to `TrialResult.metrics`, then register their interpretation in `objectives.py` when aliases are required. Objectives remain separate; weighted sums are optional post-search preferences, not the default.

## Constraints

Reject static invalid combinations in `validate_candidate` before model/data allocation. Use hard metric constraints after trial measurement. Record the reason as invalid, OOM, failed, divergent, pruned, or constraint-violated rather than collapsing all outcomes into one status.

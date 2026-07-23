from hpo.search_space import combination_count, normalize_space, preview_rows

SEARCH_SPACE = {
    "optimizer": {"type": "categorical", "choices": ["sgd", "adamw"]},
    "batch_size": {"type": "categorical", "choices": [64, 128]},
    "momentum": {"type": "categorical", "choices": [0.8, 0.9], "condition": "optimizer == 'sgd'"},
    "beta1": {"type": "categorical", "choices": [0.85, 0.9], "condition": "optimizer == 'adamw'"},
}

specs = normalize_space(SEARCH_SPACE, source_name="dictionary example")
print(preview_rows(specs))
print("valid finite combinations:", combination_count(specs))

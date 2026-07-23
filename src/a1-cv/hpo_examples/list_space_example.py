from hpo.search_space import normalize_space, preview_rows

SEARCH_SPACE = [
    {"name": "learning_rate", "type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    {"name": "batch_size", "type": "categorical", "choices": [64, 128, 256]},
    {"name": "optimizer", "type": "categorical", "choices": ["sgd", "adamw"]},
    {"name": "momentum", "type": "float", "low": 0.8, "high": 0.99, "step": 0.01, "condition": "optimizer == 'sgd'"},
]

print(preview_rows(normalize_space(SEARCH_SPACE, source_name="list example")))

# Search-space Inputs and Validation

## Python dictionary

```python
space = {
    "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    "batch_size": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "optimizer": {"type": "categorical", "choices": ["sgd", "adamw"]},
    "momentum": {
        "type": "float", "low": 0.8, "high": 0.99, "step": 0.01,
        "condition": "optimizer == 'sgd'",
    },
}
```

## Python list

```python
space = [
    {"name": "learning_rate", "type": "float", "low": 1e-5, "high": 1e-1, "log": True},
    {"name": "batch_size", "type": "categorical", "choices": [64, 128, 256]},
]
```

## CSV schema

```csv
name,type,low,high,choices,step,log,default,condition,enabled
learning_rate,float,0.00001,0.1,,,true,0.001,,true
batch_size,categorical,,,64|128|256|512,,false,256,,true
optimizer,categorical,,,sgd|adamw,,false,sgd,,true
momentum,float,0.8,0.99,,0.01,false,0.9,optimizer == "sgd",true
```

Supported types: `float`, `int`, `categorical`, `bool`, and `fixed`.

Conditions allow names, constants, lists/tuples/sets, `and`, `or`, `not`, comparisons, `in`, and `not in`. Function calls, attributes, indexing, arithmetic, imports, and unrestricted `eval` are rejected.

Validation reports the input source, item or CSV row, parameter name, invalid field, and correction. It detects duplicate names, empty choices, invalid bounds, nonpositive logarithmic ranges, bad steps/defaults, unknown condition references, cycles, and continuous spaces incorrectly requested for exhaustive enumeration.

Conditional finite combination counting follows active branches. It does not multiply inactive optimizer-specific parameters into a misleading Cartesian count.

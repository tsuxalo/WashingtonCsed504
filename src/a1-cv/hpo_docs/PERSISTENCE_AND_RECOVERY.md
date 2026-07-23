# Persistence and Recovery

Each study stores:

- Optuna SQLite history;
- a pickled sampler because Optuna RDB does not persist sampler instance state;
- configuration and search-space hashes;
- hardware/environment metadata;
- candidate/rung state;
- atomic repository checkpoints;
- JSONL trial and event logs;
- reports and selected configurations.

To resume, use the same study name, configuration, search-space hash, SQLite path, and output directory, then run with `resume: true`. Completed parameter hashes are not sampled again. A completed non-continuous study reopens by returning its existing summary.

For Google Colab, place the SQLite database, study directory, and checkpoints on Google Drive or copy them there after every bounded session. SQLite is appropriate for a single local controller; Optuna explicitly warns against SQLite for heavily parallel distributed optimization. Multi-GPU workers in this framework report to one parent controller rather than writing independent optimization transactions.

`KeyboardInterrupt` records an interruption request, preserves partial results, closes workers, and leaves checkpoints resumable.

# Limitations

- GPU and multi-GPU training were not measurable in the packaging environment; corresponding tests skip cleanly.
- Colab device model, availability, session duration, and pricing vary and are detected or supplied at runtime.
- Search estimates are projections with uncertainty, not guarantees.
- The base repository's historical CIFAR result files used the test set as validation; reference reruns are required for fair HPO comparison.
- Gradient accumulation above one requires the supplied minimal `train_loop.py` patch.
- MPS defaults to FP32 because the repository AMP path is CUDA specific.
- SQLite is used by one parent controller. It is not presented as a high-concurrency distributed database.
- FLOP values come from repository `FlopCounterMode` and are approximate operator-accounting measurements, not hardware instruction counts.
- No discovery benchmark accuracy is fabricated. The additions package includes methodology/configurations; actual CIFAR discovery results require data and accelerator execution.

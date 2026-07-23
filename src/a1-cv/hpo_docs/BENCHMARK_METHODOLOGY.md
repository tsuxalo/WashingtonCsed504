# CIFAR / ResNet Benchmark Methodology

## Reference track

The repository's measured CIFAR-10 ResNet-18 result is registered as a reference recipe: SGD, batch 512, learning rate 0.2, weight decay 5e-4, momentum 0.9, Nesterov, cosine schedule, five warm-up epochs, label smoothing 0.1, and 30 epochs. The stored result reports 92.73% top-1.

Because that historical run used the test set as validation, the HPO reference configuration must be rerun using the new deterministic train/validation split before fair comparison. The test set is only optionally evaluated during final confirmation.

## Discovery track

The broad search space contains the reference region but also SGD/AdamW, learning-rate and regularization ranges, several batch sizes, schedules, warm-up, smoothing, clipping, and augmentation. Success is declared before execution using:

- validation accuracy within a configured margin of the rerun reference; or
- lower time/compute at comparable validation accuracy;
- recovery of a similar optimizer, learning-rate, batch, schedule, and regularization region.

The framework exports best validation, budget-constrained selections, Pareto knee, examples, epochs, trial statuses, time-to-threshold data from history, compute, memory, estimate error, and parameter distance. It does not call any observed configuration a universal global optimum.

## External reference

The original ResNet paper is: He, Zhang, Ren, and Sun, “Deep Residual Learning for Image Recognition,” CVPR 2016, https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html (accessed 2026-07-20).

The paper's CIFAR architectures and schedules are reference context, not directly interchangeable with this repository's torchvision ResNet-18 stem, augmentation, validation split, precision, hardware, or training budget.

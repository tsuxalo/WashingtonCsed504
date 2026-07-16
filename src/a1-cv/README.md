# cifar100_hf_train.ipynb — a second data source through the same models

(the full CNN-vs-ViT study is in [`../README.md`](../README.md) — this notebook adds one
experiment on top of it, in one file)

## what i added

one notebook that trains our existing models on **CIFAR-100 pulled from the HuggingFace Hub**
instead of torchvision. it imports `../a1-imagenet32/models.py` completely unchanged — the only
thing that moves is `num_classes=100`, which is exactly the porting claim the main study makes,
now proven on a third dataset from a totally different source.

## why it matters for part 1

- **it checks off the proposal's data goal.** our proposal literally says *"adapt to HuggingFace
  data, evaluating different data sources"* — that was the one part-1 item nobody had done yet.
  turns out the entire cost of a new data source is a ~15-line wrapper class (HF hands us PIL
  images + labels, torchvision transforms do the rest, the training loop never notices).
  whatever datasets part 2 ends up using, this is the acquisition path.
- **it adds a third point to the crossover story.** the study showed CNN ahead by 19.5 at 50k
  images and ViT ahead by 1.2 at 1.28M — but *total images* and *images per class* were tangled
  together in that comparison. CIFAR-100 splits them: same 50k images as CIFAR-10, but 100
  classes, so **500 per class instead of 5,000**. run the notebook once with `MODEL='resnet18'`
  and once with `MODEL='vit'` and the gap tells you which kind of scale the transformer is
  actually starved of. that's a design decision we control in part 2 (class counts), so it's
  worth knowing.
- **it applies the study's lessons instead of just citing them**: the recipe follows the
  architecture (AdamW + mixup for the ViT, SGD without mixup for the CNN — the study measured
  mixup makes the CNN 7.6 points WORSE), lr scales with batch size, NaN guard in the loop, and
  workers per the engineering notes (with one new catch documented: spawned dataloader workers
  can't see notebook-defined classes, so multi-worker loading is Linux/Colab only).

## how to run it

open it in Colab from GitHub (Runtime → GPU → Run all — first cell clones the repo). full 40
epochs is ~30–40 min on a free T4. on a weak laptop set `SUBSET_PER_CLASS = 10` and
`EPOCHS = 1` — that's a pipeline check, not a result, and it still takes ~15–20 min on an old
4-core Mac (i verified it runs end-to-end there), so Colab is honestly the better idea. the
test set stays the full 10k either way so the accuracy is honest. no tokens or accounts
needed, the dataset is public.

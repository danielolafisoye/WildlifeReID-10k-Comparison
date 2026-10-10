# MiewID on WildlifeReID-10k

Fine-tunes and evaluates MiewID-msv3, the wildlife-specific candidate of the comparison. Everything that
defines the experimental protocol lives in `src/models/common/` so that ConvNeXt can be run through the
identical split, retrieval and metrics; only `backbone.py` is MiewID-specific.

```
src/models/
  common/data.py        metadata, validation split, transforms, dataset
  common/arcface.py     ArcFace identity head
  common/engine.py      embedding extraction, staged fine-tuning, query evaluation
  common/prepare.py     one-off resized copy of the dataset
  common/retrieval.py   gallery search and the four open-set scores
  common/metrics.py     Top-1/5, mAP, BAKS, BAUS, normalised accuracy, AUROC, TNR@95TPR
  miewid/backbone.py    loads MiewID-msv3 and selects the layers to unfreeze
  miewid/train.py       fine-tuning entry point
  miewid/evaluate.py    test-split evaluation, optional gallery export for the API
```

## Protocol

| Step | What happens |
|---|---|
| Split | The official `split` column is kept as is: 109,927 training images, 30,561 test images, 523 test-only (unknown) identities. |
| Validation | Carved from the training split only. 5% of the training identities per source dataset become pseudo-unknown (510 identities, 5,670 images). For the others, the latest 20% of each identity's images become known validation queries (19,351 images). 84,906 images of 9,739 identities remain for training. |
| Stage 1 | Backbone frozen. The ArcFace head starts from the identity mean embeddings and is trained alone on cached embeddings. |
| Stage 2 | The last EfficientNetV2 stage, the head convolution, GeM pooling and the BatchNorm neck are trained with a small learning rate (`2e-5`) beside the head (`1e-3`). BatchNorm statistics stay frozen. |
| Checkpoint | Chosen by validation normalised accuracy. The pretrained model is validated as epoch 0, so fine-tuning is only kept if it improves on it. |
| Threshold | One per open-set score, chosen on the full validation partition. The test split is used once, at the end. |
| Test | Gallery = whole training split. Queries = test split. |

Augmentation is training-only: rotation up to 10 degrees, 5% translation, 0.9 to 1.1 scale, brightness / contrast /
saturation jitter. No horizontal flip, because flank patterns are not symmetric.

## Open-set scores

The proposal thresholds the similarity to the nearest gallery image. The marker asked for more from the
open-set / out-of-distribution literature, so `evaluate.py` reports four scores side by side. All of them come
from the same embeddings, so comparing them costs no extra training.

| Score | Idea | Source |
|---|---|---|
| `max_sim` | similarity to the nearest gallery image (the proposal's rule) | retrieval analogue of maximum softmax probability, Hendrycks & Gimpel 2017 |
| `knn` | mean similarity to the 5 nearest gallery images | deep nearest neighbours, Sun et al. 2022 |
| `prototype` | similarity to the nearest identity centroid | nearest class mean; the "NMS" score in the WildlifeReID-10k paper; related to Mahalanobis scoring, Lee et al. 2018 |
| `margin` | best identity score minus second-best identity score | ratio test |

Check each citation before it goes into the report.

## Metrics

BAKS and BAUS follow the WildlifeReID-10k paper: averaged over images within an identity, then identities within
a source dataset, then source datasets. Normalised accuracy is their geometric mean, `sqrt(BAKS * BAUS)`.
AUROC and TNR@95TPR measure known-versus-unknown detection without a threshold. Top-1, Top-5 and mAP are computed
over the test queries whose identity is in the gallery.

## Running it on Google Colab

Training needs a GPU. In Colab choose Runtime > Change runtime type > T4 GPU, then run these cells.

**Cell 1: storage, code, libraries.** Run in every new session.

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone -b feature/miewid https://github.com/danielolafisoye/WildlifeReID-10k-Comparison.git
%cd /content/WildlifeReID-10k-Comparison
!pip install -q -r src/models/requirements.txt

DRIVE = "/content/drive/MyDrive/wildlife-reid"
RUN = f"{DRIVE}/miewid"
DATA = "/content/wr10k_440"
!mkdir -p {RUN}
!nvidia-smi
```

**Cell 2: dataset, first time only.** Colab has 2 CPU cores, and resizing 140,488 images to 440 x 440 on every
pass is what makes training slow. This cell resizes them once and stores the result in Drive (about 7 GB).

```python
import kagglehub
RAW = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k")
!python src/models/common/prepare.py --data-root {RAW} --out {DATA} --size 440
!tar -cf {DRIVE}/wr10k_440.tar -C /content wr10k_440
```

If the session drops while it is resizing, run Cell 1 and this cell again.

**Cell 2, every later session.** Restores the resized copy from Drive instead of rebuilding it.

```python
!tar -xf {DRIVE}/wr10k_440.tar -C /content
!ls {DATA}
```

**Cells 3 to 5: the experiment.**

```python
!python src/models/miewid/evaluate.py --data-root {DATA} --out {RUN}/eval_pretrained
```

```python
!python src/models/miewid/train.py --data-root {DATA} --out {RUN} --epochs 3 --resume
```

```python
!python src/models/miewid/evaluate.py --data-root {DATA} --out {RUN}/eval_finetuned --checkpoint {RUN}/best_backbone.pt
```

All three continue from where they stopped: after a disconnect, run Cell 1, the restore cell, and the same
command again. If the GPU runs out of memory during training, add `--batch-size 16`.

Outputs in `RUN`:

- `eval_pretrained/results.json` and `eval_finetuned/results.json`: every number for the report, including
  model time per image and parameter count.
- `best_backbone.pt`: fine-tuned weights. Serve them with `REID_MODEL=miewid REID_CHECKPOINT=<path>`.
- `history.json`: loss and validation metrics per epoch.
- `last.pt` and `eval_*/embeddings_*.npz`: resume points (about 3 GB together; delete them when finished).

Adding `--export-gallery <folder>` to `evaluate.py` writes the gallery index the Flask API loads.

The scripts need `transformers` 4.x, which the `pip install` line takes care of. MiewID does not load with 5.x.

To check a change on a laptop before pushing, run the same scripts on a few identities with
`--limit-identities 12 --epochs 1 --batch-size 4 --workers 0`.

## Validity notes for the report

- MiewID's own training data overlaps with WildlifeReID-10k (stated by the benchmark authors), so its numbers
  are not a leakage-free measurement.
- The validation thresholds are chosen against a gallery of 84,906 images; at test time the gallery is the full
  109,927-image training split.
- Training and evaluation read a copy of the dataset resized once to 440 x 440 and stored as JPEG (quality 92),
  so pixels differ very slightly from resizing the original files at query time.
- The no-date part of the validation holdout orders images by `cluster_id`, which only approximates the
  benchmark's similarity-aware split.

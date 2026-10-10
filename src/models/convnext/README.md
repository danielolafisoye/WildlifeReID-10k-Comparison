# ConvNeXt Base on WildlifeReID-10k (re-identification pipeline)

Runs ConvNeXt Base, the general-purpose candidate, through exactly the same protocol as MiewID: same validation
split, ArcFace fine-tuning, gallery search, open-set scores and metrics, all from `src/models/common/`. Only
`backbone.py` differs. The protocol itself is described in `src/models/miewid/README.md`.

| | MiewID | ConvNeXt Base |
|---|---|---|
| Pretraining | wildlife re-identification (MiewID-msv3) | ImageNet (timm `convnext_base.fb_in22k_ft_in1k`) |
| Input | 440 x 440 | 224 x 224 |
| Embedding | 2152-d | 1024-d (pooled features after the final LayerNorm) |
| Parameters | 51.1M | 87.6M |
| Unfrozen by default | last of 6 stages + neck (39M) | last of 4 stages + final norm (27M) |

`train_reid.py` and `evaluate_reid.py` are the entry points. `train.py`, `test.py` and `prepare_data.py` in this
folder belong to the separate closed-set classifier and are not used by this pipeline.

## Running it on Google Colab

Runtime > Change runtime type > T4 GPU, then:

**Cell 1: storage, code, libraries.** Run in every new session.

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!git clone -b feature/convnext-reid https://github.com/danielolafisoye/WildlifeReID-10k-Comparison.git
%cd /content/WildlifeReID-10k-Comparison
!pip install -q -r src/models/requirements.txt

DRIVE = "/content/drive/MyDrive/wildlife-reid"
RUN = f"{DRIVE}/convnext"
DATA = "/content/wr10k_224"
!mkdir -p {RUN}
!nvidia-smi
```

**Cell 2: dataset, first time only.** Makes the 224 x 224 copy and stores it in Drive.

```python
import kagglehub
RAW = kagglehub.dataset_download("wildlifedatasets/wildlifereid-10k")
!python src/models/common/prepare.py --data-root {RAW} --out {DATA} --size 224
!tar -cf {DRIVE}/wr10k_224.tar -C /content wr10k_224
!ls -lh {DRIVE}/wr10k_224.tar
```

**Cell 2, every later session.**

```python
!tar -xf {DRIVE}/wr10k_224.tar -C /content
!ls {DATA}
```

**Cells 3 to 5: the experiment.**

```python
!python src/models/convnext/evaluate_reid.py --data-root {DATA} --out {RUN}/eval_pretrained
```

```python
!python src/models/convnext/train_reid.py --data-root {DATA} --out {RUN} --epochs 3 --resume
```

```python
!python src/models/convnext/evaluate_reid.py --data-root {DATA} --out {RUN}/eval_finetuned --checkpoint {RUN}/best_backbone.pt
```

All three continue from where they stopped after a disconnect. Use the same `--epochs` as the MiewID run so the
two models get the same training budget.

## Notes for the report

- The pretrained baseline is expected to be weak: ImageNet features were never trained to tell individuals apart.
  That gap is part of the research question.
- ConvNeXt sees 224 x 224 images and MiewID 440 x 440, as each model's pretrained weights require. The models
  therefore differ in input resolution as well as in pretraining.
- timm's default ConvNeXt Base weights are pretrained on ImageNet-22k and fine-tuned on ImageNet-1k.

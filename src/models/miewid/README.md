# MiewID

This folder fine-tunes the official `conservationxlabs/miewid-msv3` embedding
model on the same 100 WildlifeReID-10k identities used by ConvNeXt.

## Files

- `prepare_data.py`: identical known/unknown split selection (`random_state=42`)
- `train.py`: official two-stage MiewID + sub-center ArcFace fine-tuning
- `test.py`: Top-1, Top-5, open-set AUROC, and optional mAP
- `inference.py`: standalone image-to-gallery retrieval with unknown rejection
- `selected_identities.json`: permanent identity record shared by every MiewID run

## Run

Set `WILDLIFE_DATASET_ROOT` if the dataset is not in the KaggleHub cache, then:

```bash
pip install -r requirements.txt
python train.py
python test.py
python inference.py path/to/query.jpg --threshold 0.50
```

If `metadata.csv` or a selected image is missing, `prepare_data.py` downloads
only that file from the public Kaggle dataset and reuses KaggleHub's cache.
Selected files are prefetched sequentially before training or testing, avoiding
Colab's unreliable dataset-mount resolver and DataLoader download races.

MiewID's official custom model currently requires Transformers 4. If Colab has
already installed Transformers 5, run:

```bash
pip install --force-reinstall "transformers>=4.45,<5.0"
```

Training saves `miewid_weights.pth`, `label_map.json`, and
`known_gallery.pth` in this directory. The gallery contains normalized
embeddings from known-identity training images; test images are only queries.

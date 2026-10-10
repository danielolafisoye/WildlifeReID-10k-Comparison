# Wildlife Re-Identification: models, Flask API and React front end

Open-set animal re-identification on WildlifeReID-10k. MiewID and ConvNeXt Base are fine-tuned and evaluated
under one protocol, and the better one is served behind a REST API with a web app.

```
User image -> React app -> Flask REST API -> preprocessing -> selected ReID model
           -> cosine similarity vs gallery -> open-set rule -> JSON -> React
```

## What is where

| Part | Location | State |
|---|---|---|
| Shared protocol: split, ArcFace training, retrieval, metrics | `src/models/common/` | Working |
| MiewID | `src/models/miewid/` | Trained and evaluated on the full dataset |
| ConvNeXt Base | `src/models/convnext/` | Code ready, smoke-tested; full run not done yet |
| Flask API | `src/backend/` | Working, 9 tests |
| React UI | `src/frontend/` | Working |

How to train and evaluate is in `src/models/miewid/README.md` and `src/models/convnext/README.md`.

## Run the app

Backend (port 5000):

```bash
cd src/backend
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python app.py
```

Frontend (port 5173, proxies `/api` to the backend):

```bash
cd src/frontend
npm install
npm run dev
```

Open http://localhost:5173. Without a model and gallery the API runs a weight-free `dummy` embedder and every
query returns "unknown", which is enough to check that the two halves talk to each other.

## Serve a trained model

The API needs three things from a finished training run: the fine-tuned weights, the gallery of known
individuals, and the open-set rule with its threshold. The evaluation script exports the last two together.

1. In Colab, after the fine-tuned evaluation has finished, run it once more with `--export-gallery`. The
   embeddings are already cached, so this takes a few minutes:

   ```python
   !python src/models/miewid/evaluate.py --data-root {DATA} --out {RUN}/eval_finetuned --checkpoint {RUN}/best_backbone.pt --export-gallery {RUN}/deploy
   ```

2. Download `deploy/embeddings.npy`, `deploy/meta.json` and `best_backbone.pt` from Google Drive. Put the first
   two in `src/backend/data/gallery/` and the weights anywhere, for example `src/backend/data/best_backbone.pt`.

3. Install the model libraries and start the API with the model selected:

   ```bat
   pip install -r ..\models\requirements.txt
   set REID_MODEL=miewid
   set REID_CHECKPOINT=data\best_backbone.pt
   python app.py
   ```

For ConvNeXt use `src/models/convnext/evaluate_reid.py` and `REID_MODEL=convnext`. The gallery must come from the
same model and checkpoint that the API loads.

## Open-set rule

`meta.json` carries the rule and threshold that scored best on validation data, so the API applies what the
experiment found instead of a hand-picked number.

| Score | Accept the best match when | 
|---|---|
| `max_sim` | its similarity is at least the threshold (the rule in the proposal) |
| `margin` | it is ahead of the second-best individual by at least the threshold |

## API

`POST /api/predict`: multipart form, field `image` (jpg/jpeg/png/webp, up to 10 MB), optional `threshold`.

```json
{
  "status": "known",
  "prediction": { "identity": "zebra_001", "score": 0.83 },
  "reason": null,
  "top_k": [ { "identity": "zebra_001", "score": 0.83, "image_id": "images/.../a.jpg" } ],
  "score": "margin",
  "decision_score": 0.21,
  "threshold": 0.091,
  "model": "miewid",
  "inference_ms": 41.2
}
```

When the rule rejects the match: `"status": "unknown"`, `"reason": "below_threshold"` (or `"gallery_empty"`).
Errors return `{ "error": "..." }` with 400 (bad/missing image), 413 (too large) or 415 (wrong file type).

`GET /api/health` reports the model and gallery size. `GET /api/config` reports the rule, its threshold and the
upload limits for the UI.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `REID_MODEL` | `dummy` | `dummy`, `convnext` or `miewid` |
| `REID_DEVICE` | `cpu` | `cpu` or `cuda` |
| `REID_CHECKPOINT` | empty | fine-tuned backbone weights (`best_backbone.pt`); empty = pretrained weights |
| `REID_GALLERY_DIR` | `src/backend/data/gallery` | folder holding `embeddings.npy` and `meta.json` |
| `REID_SCORE` | empty | `max_sim` or `margin`; empty = the rule stored in the gallery |
| `REID_THRESHOLD` | empty | threshold; empty = the validated threshold stored in the gallery |
| `REID_TOP_K` | `5` | number of ranked identities returned |
| `REID_MAX_UPLOAD_MB` | `10` | upload limit |
| `CORS_ORIGINS` | `http://localhost:5173` | allowed browser origins |
| `VITE_API_URL` (frontend) | empty | API base URL for a deployed build |

## A small gallery without training

`python scripts/build_gallery.py --images gallery_images --model dummy` (from `src/backend/`) builds a gallery
from a folder with one sub-folder of images per identity. It is useful for demos and tests only.

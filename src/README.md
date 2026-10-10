# Wildlife Re-Identification: Flask API + React front end

Scaffold for the Big Data Analytics project: open-set animal re-identification on WildlifeReID-10k,
comparing MiewID and ConvNeXt Base, then serving the better model behind a REST API and a web app.

```
User image -> React app -> Flask REST API -> preprocessing -> selected ReID model
           -> cosine similarity vs gallery -> open-set threshold -> JSON -> React
```

## What works now, and what is a placeholder

| Part | State |
|---|---|
| Flask API (`/api/predict`, `/api/health`, `/api/config`) | Working, tested (7 tests) |
| Upload validation, preprocessing, gallery search, open-set threshold | Working |
| React UI (drop zone, threshold slider, Top-5 results, unknown message) | Working |
| `dummy` embedder | Weight-free stand-in (compares coarse colour layout). Not a real re-id model |
| `convnext` / `miewid` embedders | Wired in but **untested**: need `torch`, `timm`, `transformers` and weights |
| Training / fine-tuning, ArcFace head, metrics (Top-1/5, mAP, BAKS, BAUS, AUROC) | **Not included**. This is the experimental part of the project |

## Run it

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

Open http://localhost:5173.

## Build a gallery

The gallery is the set of known individuals the query is compared against. Put images in one folder per identity:

```
gallery_images/
    zebra_001/img_a.jpg
    zebra_001/img_b.jpg
    leopard_017/img_c.jpg
```

then, from `src/backend/`:

```bash
python scripts/build_gallery.py --images gallery_images --model dummy
```

This writes `src/backend/data/gallery/embeddings.npy` and `meta.json`. Restart the API to load it. With an empty gallery
every query returns "unknown" (reason `gallery_empty`).

## API

`POST /api/predict`: multipart form, field `image` (jpg/jpeg/png/webp, up to 10 MB), optional `threshold`.

```json
{
  "status": "known",
  "prediction": { "identity": "zebra_001", "score": 0.83 },
  "reason": null,
  "top_k": [ { "identity": "zebra_001", "score": 0.83, "image_id": "zebra_001/img_a.jpg" } ],
  "threshold": 0.65,
  "model": "miewid",
  "inference_ms": 41.2
}
```

When no match reaches the threshold: `"status": "unknown"`, `"reason": "below_threshold"` (or `"gallery_empty"`).
Errors return `{ "error": "..." }` with 400 (bad/missing image), 413 (too large) or 415 (wrong file type).

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `REID_MODEL` | `dummy` | `dummy`, `convnext` or `miewid` |
| `REID_DEVICE` | `cpu` | `cpu` or `cuda` |
| `REID_CHECKPOINT` | empty | fine-tuned backbone weights |
| `REID_THRESHOLD` | `0.65` | open-set threshold tau. The paper's 0.6-0.7 region for MiewID is only a starting point; tune per model on validation data |
| `REID_TOP_K` | `5` | number of ranked identities returned |
| `REID_GALLERY_DIR` | `src/backend/data/gallery` | where the gallery index is read from |
| `REID_MAX_UPLOAD_MB` | `10` | upload limit |
| `CORS_ORIGINS` | `http://localhost:5173` | allowed browser origins |
| `VITE_API_URL` (frontend) | empty | API base URL for a deployed build |

## Layout

```
src/backend/
  app.py            Flask app + routes
  config.py         settings
  reid/             preprocessing, model (embedders), gallery, openset
  scripts/          build_gallery.py
  tests/            pytest suite
src/frontend/
  src/              App.jsx, api.js, components/UploadDropzone.jsx, components/ResultPanel.jsx
```

## Next steps for the project

1. Download WildlifeReID-10k and keep its official open-set / time-aware / similarity-aware splits.
2. Fine-tune each backbone with an ArcFace head, then save the backbone weights (staged unfreezing as in the proposal).
3. Build one gallery per model with `build_gallery.py` and choose each model's threshold on a validation split.
4. Compute Top-1, Top-5, mAP, BAKS, BAUS and AUROC, then set `REID_MODEL` to the winner.
5. Check the MiewID model id and input size in `reid/model.py` against its official model card.

The repo's `src/models/convnext/` and `src/models/miewid/` folders are for the training code. The embedders used
for serving live in `src/backend/reid/model.py`; load the fine-tuned backbone weights there via `REID_CHECKPOINT`.

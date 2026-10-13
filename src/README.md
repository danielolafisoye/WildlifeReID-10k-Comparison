# Application guide

The working application serves the saved ConvNeXt Base and MiewID models behind
a Flask API and lets the React frontend choose either model. ConvNeXt returns
softmax-ranked identities. MiewID creates an embedding and performs cosine
similarity search against its saved known-animal gallery.

```text
Upload or test image -> React -> Flask -> selected model -> open-set threshold
                     -> predicted identity + confidence + Top-5 -> React
```

## Current implementation

| Part | State |
|---|---|
| `/api/health`, `/api/config`, `/api/predict` | Working and tested |
| `/api/test-images` and image endpoint | Local, archive, or Kaggle on demand |
| ConvNeXt saved checkpoint and label map | Connected |
| MiewID saved checkpoint and embedding gallery | Connected |
| Upload and test-image frontend flows | Connected |
| Top-5 results and 50% unknown threshold | Connected |
| Backend automated tests | 8 tests |

## Run

From the repository root, install `src/backend/requirements.txt` in `.venv`.
Then use two terminals.

```bash
# terminal 1
source .venv/bin/activate
cd src/backend
python app.py

# terminal 2
cd src/frontend
npm install
npm run dev
```

Open <http://localhost:4173>. Flask uses port 5000.

## Prediction API

Send `POST /api/predict` as multipart form data with:

- `model`: `convnext` or `miewid`
- exactly one of `image` or `test_image_id`

Example response:

```json
{
  "predicted_identity": "ATRW_99",
  "confidence": 0.82,
  "is_unknown": false,
  "top_5": ["ATRW_99", "ATRW_12", "ATRW_45", "ATRW_08", "ATRW_31"],
  "top_5_details": [
    {"identity": "ATRW_99", "confidence": 0.82}
  ],
  "model": "miewid",
  "threshold": 0.5,
  "source": "upload",
  "inference_ms": 91.2
}
```

## Main configuration

| Variable | Default | Meaning |
|---|---|---|
| `REID_DEFAULT_MODEL` | `convnext` | Initial model selection |
| `REID_DEVICE` | `auto` | Automatically selects CUDA, MPS, then CPU |
| `REID_THRESHOLD` | `0.50` | Shared frontend unknown cutoff |
| `REID_TOP_K` | `5` | Number of candidates returned |
| `WILDLIFE_KAGGLE_ON_DEMAND` | `true` | Fetch only requested dataset files |
| `WILDLIFE_DATASET_ROOT` | `data/wildlifereid-10k` | Optional extracted dataset |
| `WILDLIFE_DATASET_ARCHIVE` | `data/archive.zip` | Optional full archive |

All checkpoint, gallery, upload and CORS variables are documented in
`src/backend/.env.example`.

## Tests and production build

```bash
source .venv/bin/activate
python -m pytest src/backend/tests -q
cd src/frontend
npm run build
```

The final saved-checkpoint comparison is implemented separately in
`src/models/evaluate_comparison.py`; it does not change the serving code or
retrain either model.

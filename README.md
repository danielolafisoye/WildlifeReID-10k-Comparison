# WildlifeReID-10k model comparison

Open-set wildlife re-identification with a React interface, Flask API, and two
trained model options: ConvNeXt Base classification and MiewID embedding
retrieval. Users can upload an image or choose a WildlifeReID-10k test image,
select a model, and receive the predicted identity, unknown decision, confidence,
and Top-5 candidates.

## Final comparison

Both saved checkpoints were evaluated without retraining on the same ordered
100 identities and the same 606 test queries. The test set contains 238 new
images of known identities and 368 images from 20 identities never seen during
training.

| Model | Top-1 | Top-5 | Open-set AUROC | Mean inference time |
|---|---:|---:|---:|---:|
| ConvNeXt Base | 82.77% | **97.90%** | 90.81% | **34.1 ms** |
| MiewID | **88.24%** | 95.80% | **95.86%** | 91.2 ms |

MiewID is the recommended primary re-identification model because it leads on
Top-1 accuracy and unknown-animal detection. ConvNeXt remains the faster
baseline and has the stronger Top-5 result. Inference times were measured on
Apple MPS and should only be compared within this run.

Detailed outputs:

- [Written report](docs/results_report.md)
- [Final report PDF](docs/WildlifeReID_Final_Report.pdf)
- [Presentation](ss/WildlifeReID_Results.pptx)
- [Aggregate results](src/models/comparison_results.json)
- [Per-query predictions](src/models/comparison_predictions.csv)

## Project structure

```text
src/
  frontend/       React user interface
  backend/        Flask inference API
  models/
    convnext/     ConvNeXt training, testing, weights and label map
    miewid/       MiewID training, testing, inference, gallery and weights
data/             Dataset access notes; full dataset is ignored
dist/frontend/    Production frontend build
ss/               Final presentation
docs/             Report, figures and reproducibility scripts
```

## Quick start with Docker

The easiest way to run the app. You need
[Docker Desktop](https://www.docker.com/products/docker-desktop/) and
[Git LFS](https://git-lfs.com/).

```bash
git clone <repository-url>
cd WildlifeReID-10k-Comparison
git lfs install
git lfs pull
docker compose up -d
```

Run these commands in the project folder (the one containing
`compose.yaml`). Open <http://localhost:4173>. The first start builds the
image and takes several minutes (PyTorch is large); later starts are quick.
Use `docker compose up` instead to see the logs in the terminal (stop with
`Ctrl+C`), and `docker compose down` to stop it. After changing the code,
rebuild with `docker compose up -d --build`.

Everything runs in one container: Flask serves both the API and the built
React app, on CPU, with both models included. Test images are downloaded
from Kaggle on demand and kept in the `kaggle-cache` volume.

If you already have the dataset, put it in `data/wildlifereid-10k/` (or
`data/archive.zip`) and it is used instead of Kaggle.

Troubleshooting:

- **Model fails to load**: the weights are Git LFS files. Run
  `git lfs install` and `git lfs pull`, then `docker compose up -d --build`.
- **"Test dataset is unavailable"**: the container could not reach Kaggle.
  Uploading your own image still works.
- **First prediction is slow**: the model loads on first use.
- **Out of memory**: give Docker Desktop at least 4 GB of RAM
  (Settings > Resources).

## Manual setup

The model checkpoints use Git LFS:

```bash
git lfs install
git lfs pull
```

Create one Python environment from the repository root and install the backend
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/backend/requirements.txt
```

Install the frontend packages:

```bash
cd src/frontend
npm install
```

## Run the application

Use two terminals from the repository root.

Backend:

```bash
source .venv/bin/activate
cd src/backend
python app.py
```

Frontend:

```bash
cd src/frontend
npm run dev
```

Open <http://localhost:4173>. The frontend proxies `/api` to Flask on port
5000. Model files are loaded lazily on first use, so the first prediction can
take longer.

## Dataset access

The 26 GB dataset is not committed. By default, the backend uses KaggleHub to
download `metadata.csv` and only the selected test image when requested. This
requires internet access and Kaggle access where applicable. Alternatively set:

```bash
export WILDLIFE_DATASET_ROOT=/absolute/path/to/wildlifereid-10k
# or
export WILDLIFE_DATASET_ARCHIVE=/absolute/path/to/archive.zip
```

See [data/dataset_url.txt](data/dataset_url.txt) for the public dataset URL.

## Verify the project

```bash
source .venv/bin/activate
python -m pytest src/backend/tests -q
cd src/frontend && npm run build
```

Re-run the fair saved-checkpoint comparison from the repository root with:

```bash
source .venv/bin/activate
pip install -r docs/requirements.txt
python src/models/evaluate_comparison.py --device auto
python docs/generate_results_assets.py
python docs/build_results_report.py
```

The evaluator first confirms that both label maps contain the same identities
in the same order. Top-1 and Top-5 are computed on known queries; open-set
AUROC uses known and unknown queries across all possible thresholds. It does
not retrain or modify either checkpoint.

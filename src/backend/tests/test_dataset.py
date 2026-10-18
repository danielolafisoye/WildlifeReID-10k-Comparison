import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reid.dataset import TestImageCatalog as ImageCatalog  # noqa: E402


def test_kaggle_mode_downloads_only_metadata_and_requested_image(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    image_path = tmp_path / "query.jpg"
    label_map_path = tmp_path / "label_map.json"

    pd.DataFrame(
        [
            {"split": "train", "identity": "known_1", "path": "train/known.jpg"},
            {"split": "test", "identity": "known_1", "path": "test/known.jpg"},
            {"split": "test", "identity": "unknown_1", "path": "test/unknown.jpg"},
        ]
    ).to_csv(metadata_path, index=False)
    image_path.write_bytes(b"image-bytes")
    label_map_path.write_text(
        json.dumps({"index_to_identity": ["known_1"]}),
        encoding="utf-8",
    )

    requested = []

    def download_file(handle, path):
        requested.append((handle, path))
        return str(metadata_path if path == "metadata.csv" else image_path)

    catalog = ImageCatalog(
        dataset_root=tmp_path / "missing-directory",
        archive_path=tmp_path / "missing.zip",
        label_map_path=label_map_path,
        unknown_identity_count=1,
        kaggle_dataset_handle="owner/dataset/versions/1",
        download_file=download_file,
    )

    assert requested == [("owner/dataset/versions/1", "metadata.csv")]
    listing = catalog.list_images(page=1, page_size=10)
    assert listing["total"] == 2

    selected_id = listing["items"][0]["id"]
    data, _, _ = catalog.image_bytes(selected_id)
    assert data == b"image-bytes"
    assert requested[-1][1] in {"test/known.jpg", "test/unknown.jpg"}

    reference_data, _, _ = catalog.identity_image_bytes("known_1")
    assert reference_data == b"image-bytes"
    assert requested[-1][1] == "train/known.jpg"

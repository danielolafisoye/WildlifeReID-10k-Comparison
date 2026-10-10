"""Read test images locally or download individual Kaggle files on demand."""

import json
import mimetypes
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

import pandas as pd


class DatasetUnavailableError(RuntimeError):
    """Raised when the local dataset or a requested image is unavailable."""


class TestImageCatalog:
    def __init__(
        self,
        dataset_root: Path,
        archive_path: Path,
        label_map_path: Path,
        unknown_identity_count: int = 20,
        kaggle_dataset_handle: str | None = None,
        allow_remote: bool = True,
        download_file: Callable[..., str] | None = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.archive_path = Path(archive_path)
        self.label_map_path = Path(label_map_path)
        self.kaggle_dataset_handle = kaggle_dataset_handle
        self.allow_remote = allow_remote
        self.download_file = download_file
        self.source = self._find_source()
        metadata = self._read_metadata()
        labels = json.loads(self.label_map_path.read_text(encoding="utf-8"))["index_to_identity"]

        train = metadata[metadata["split"].eq("train")]
        test = metadata[metadata["split"].eq("test")]
        known = test[test["identity"].isin(labels)].copy()

        unknown_pool = test[~test["identity"].isin(train["identity"])]
        unknown_identities = unknown_pool["identity"].drop_duplicates()
        sample_size = min(unknown_identity_count, len(unknown_identities))
        selected_unknown = unknown_identities.sample(sample_size, random_state=42)
        unknown = unknown_pool[unknown_pool["identity"].isin(selected_unknown)].copy()

        known["query_type"] = "known"
        unknown["query_type"] = "unknown"
        self.frame = pd.concat([known, unknown]).sort_index().copy()
        self.frame["query_id"] = self.frame.index.map(str)
        self.rows = {row["query_id"]: row for _, row in self.frame.iterrows()}

    def _find_source(self) -> str:
        if (self.dataset_root / "metadata.csv").exists():
            return "directory"
        if self.archive_path.exists():
            return "archive"
        if self.allow_remote and self.kaggle_dataset_handle:
            if self.download_file is None:
                try:
                    import kagglehub
                except ImportError as exc:
                    raise DatasetUnavailableError(
                        "Install kagglehub to load WildlifeReID-10k images on demand."
                    ) from exc
                self.download_file = kagglehub.dataset_download
            return "kaggle"
        raise DatasetUnavailableError(
            "WildlifeReID-10k is not available. Set WILDLIFE_DATASET_ROOT or "
            "WILDLIFE_DATASET_ARCHIVE, or enable WILDLIFE_KAGGLE_ON_DEMAND."
        )

    def _download_from_kaggle(self, relative_path: str) -> Path:
        try:
            downloaded = self.download_file(
                self.kaggle_dataset_handle,
                path=relative_path,
            )
        except Exception as exc:
            raise DatasetUnavailableError(
                f"Could not download '{relative_path}' from Kaggle: {exc}"
            ) from exc
        downloaded_path = Path(downloaded)
        if not downloaded_path.is_file():
            raise DatasetUnavailableError(
                f"Kaggle did not return the requested file: {relative_path}"
            )
        return downloaded_path

    def _read_metadata(self):
        if self.source == "directory":
            return pd.read_csv(self.dataset_root / "metadata.csv", low_memory=False)
        if self.source == "kaggle":
            return pd.read_csv(
                self._download_from_kaggle("metadata.csv"),
                low_memory=False,
            )
        with zipfile.ZipFile(self.archive_path) as archive:
            try:
                with archive.open("metadata.csv") as metadata:
                    return pd.read_csv(metadata, low_memory=False)
            except KeyError as exc:
                raise DatasetUnavailableError("metadata.csv is missing from the dataset archive.") from exc

    def list_images(self, page: int, page_size: int, query_type: str = "all") -> dict:
        frame = self.frame
        if query_type in {"known", "unknown"}:
            frame = frame[frame["query_type"].eq(query_type)]
        total = len(frame)
        start = (page - 1) * page_size
        items = [
            {
                "id": row["query_id"],
                "query_type": row["query_type"],
                "image_url": f"/api/test-images/{row['query_id']}/image",
            }
            for _, row in frame.iloc[start : start + page_size].iterrows()
        ]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": max(1, (total + page_size - 1) // page_size),
        }

    def image_bytes(self, query_id: str) -> tuple[bytes, str, str]:
        row = self.rows.get(str(query_id))
        if row is None:
            raise KeyError(query_id)
        relative = str(PurePosixPath(str(row["path"])))
        filename = PurePosixPath(relative).name
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        if self.source == "archive":
            with zipfile.ZipFile(self.archive_path) as archive:
                try:
                    return archive.read(relative), filename, mime_type
                except KeyError as exc:
                    raise DatasetUnavailableError(f"Image is missing from archive: {relative}") from exc

        if self.source == "kaggle":
            image_path = self._download_from_kaggle(relative)
            return image_path.read_bytes(), filename, mime_type

        root = self.dataset_root.resolve()
        image_path = (root / relative).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise DatasetUnavailableError(f"Dataset image is unavailable: {relative}")
        return image_path.read_bytes(), filename, mime_type

import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def resize_one(job: tuple[str, str, int]) -> None:
    src, dst, size = job
    if os.path.exists(dst):
        return
    with Image.open(src) as img:
        img.draft("RGB", (size, size))
        out = img.convert("RGB").resize((size, size), Image.BICUBIC)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    out.save(tmp, format="JPEG", quality=92)
    os.replace(tmp, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a resized copy of WildlifeReID-10k so training does not resize every image every epoch.")
    parser.add_argument("--data-root", required=True, type=Path, help="original dataset folder")
    parser.add_argument("--out", required=True, type=Path, help="folder for the resized copy; use it as --data-root afterwards")
    parser.add_argument("--size", type=int, required=True, help="square side in pixels (440 for MiewID)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    paths = pd.read_csv(args.data_root / "metadata.csv", low_memory=False)["path"]
    jobs = [(str(args.data_root / p), str(args.out / p), args.size) for p in paths]
    args.out.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(resize_one, jobs, chunksize=64), total=len(jobs), desc="resizing"):
            pass
    shutil.copy(args.data_root / "metadata.csv", args.out / "metadata.csv")
    print(f"resized copy written to {args.out}")


if __name__ == "__main__":
    main()

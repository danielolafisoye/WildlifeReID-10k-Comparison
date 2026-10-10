"""Build the gallery index from a folder of known individuals.

Expected layout (one sub-folder per identity):

    gallery_images/
        zebra_001/img_a.jpg
        zebra_001/img_b.jpg
        leopard_017/img_c.jpg

Usage (run from the backend folder):

    python scripts/build_gallery.py --images gallery_images --model dummy
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402
from reid.gallery import Gallery  # noqa: E402
from reid.model import load_embedder  # noqa: E402
from reid.preprocessing import InvalidImageError, load_image  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path, help="folder with one sub-folder per identity")
    parser.add_argument("--model", default=Config.MODEL_NAME, help="dummy | miewid | convnext")
    parser.add_argument("--device", default=Config.DEVICE)
    parser.add_argument("--checkpoint", default=Config.CHECKPOINT)
    parser.add_argument("--out", type=Path, default=Config.GALLERY_DIR)
    args = parser.parse_args()

    embedder = load_embedder(args.model, args.device, args.checkpoint)
    vectors, identities, image_ids = [], [], []

    for identity_dir in sorted(p for p in args.images.iterdir() if p.is_dir()):
        for path in sorted(identity_dir.iterdir()):
            if path.suffix.lower() not in EXTS:
                continue
            try:
                image = load_image(path.read_bytes())
            except InvalidImageError:
                print(f"skipped unreadable image: {path}")
                continue
            vectors.append(embedder.embed(image))
            identities.append(identity_dir.name)
            image_ids.append(f"{identity_dir.name}/{path.name}")

    if not vectors:
        sys.exit("No images found. Check the folder layout in the script docstring.")

    gallery = Gallery(np.stack(vectors), identities, image_ids, model_name=embedder.name)
    gallery.save(args.out)
    print(f"saved {len(gallery)} embeddings for {gallery.num_identities} identities to {args.out}")


if __name__ == "__main__":
    main()

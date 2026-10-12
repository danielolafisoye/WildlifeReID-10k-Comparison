"""Evaluate MiewID as embedding retrieval on the shared open-set test split."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from inference import DEFAULT_GALLERY, DEFAULT_WEIGHTS, identity_matches, load_finetuned_model
from model import extract_embeddings
from prepare_data import identities, test_loader


def average_precision(similarities, true_identity, gallery_identities):
    """Image-gallery AP hook; enabled with --compute-map."""
    order = similarities.argsort(descending=True)
    relevant = torch.tensor(
        [gallery_identities[index] == true_identity for index in order.tolist()],
        dtype=torch.float32,
    )
    if relevant.sum() == 0:
        return 0.0
    precision = relevant.cumsum(0) / torch.arange(1, len(relevant) + 1)
    return float((precision * relevant).sum() / relevant.sum())


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MiewID retrieval.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-map", action="store_true")
    args = parser.parse_args()

    model, device = load_finetuned_model(args.weights, args.device)
    gallery = torch.load(args.gallery, map_location="cpu", weights_only=True)
    gallery_embeddings = F.normalize(gallery["embeddings"].float(), dim=1)
    gallery_identities = gallery["identities"]

    print("Testing on:", device, flush=True)
    top1_correct = 0
    top5_correct = 0
    known_total = 0
    unknown_labels = []
    unknown_scores = []
    average_precisions = []

    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing MiewID"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                query_embeddings = extract_embeddings(model, images).cpu()

            image_similarities = query_embeddings @ gallery_embeddings.T

            for row, label in enumerate(labels.tolist()):
                matches = identity_matches(
                    query_embeddings[row : row + 1],
                    gallery,
                    top_k=5,
                )
                best_similarity = matches[0]["similarity"]
                is_unknown = label == -1
                unknown_labels.append(int(is_unknown))
                unknown_scores.append(1.0 - best_similarity)

                if not is_unknown:
                    true_identity = identities[label]
                    predicted = [match["identity"] for match in matches]
                    top1_correct += int(predicted[0] == true_identity)
                    top5_correct += int(true_identity in predicted)
                    known_total += 1

                    if args.compute_map:
                        average_precisions.append(
                            average_precision(
                                image_similarities[row],
                                true_identity,
                                gallery_identities,
                            )
                        )

    top1_accuracy = top1_correct / known_total
    top5_accuracy = top5_correct / known_total
    auroc = roc_auc_score(unknown_labels, unknown_scores)

    print("\nFINAL RESULTS")
    print(f"Top 1 Accuracy: {top1_accuracy * 100:.2f}%")
    print(f"Top 5 Accuracy: {top5_accuracy * 100:.2f}%")
    print(f"Open Set AUROC: {auroc * 100:.2f}%")
    if args.compute_map:
        mean_average_precision = sum(average_precisions) / len(average_precisions)
        print(f"mAP: {mean_average_precision * 100:.2f}%")


if __name__ == "__main__":
    main()

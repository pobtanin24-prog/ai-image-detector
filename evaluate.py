"""Evaluate the saved CIFAKE classifier without retraining.

Run from the project root, after installing scikit-learn and matplotlib:
    python evaluate.py

Requires: ai_detector.pth, data/test/FAKE/, data/test/REAL/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "ai_detector.pth"
TEST_DIR = ROOT / "data" / "test"
OUTPUT_DIR = ROOT / "results"
LABELS = ["FAKE", "REAL"]


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not TEST_DIR.is_dir():
        raise FileNotFoundError(f"Missing test data: {TEST_DIR}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    if dataset.classes != LABELS:
        raise ValueError(f"Unexpected class order: {dataset.classes} (expected {LABELS})")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    true_labels = []
    predicted_labels = []
    with torch.inference_mode():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predictions.tolist())

    if not true_labels:
        raise RuntimeError("No test images found")

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    accuracy = sum(t == p for t, p in zip(true_labels, predicted_labels)) / len(true_labels)

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = {
        "model": "ResNet18 frozen backbone + trained linear head",
        "dataset": "Local CIFAKE subset (data/test)",
        "n_test": len(true_labels),
        "class_order": LABELS,
        "accuracy": accuracy,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "classification_report": report,
    }
    output_json = OUTPUT_DIR / "evaluation.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title("CIFAKE Test: Confusion Matrix")
    fig.tight_layout()
    output_png = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(output_png, dpi=160)
    plt.close(fig)

    print(f"Test images: {len(true_labels)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Rows = actual, columns = predicted; class order = FAKE, REAL")
    print(matrix)
    for name in LABELS:
        metrics = report[name]
        print(
            f"{name}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1-score']:.3f}"
        )
    print(f"Saved: {output_json}")
    print(f"Saved: {output_png}")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_classifier_v2.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FINAL BASE THRESHOLDS
# Validation-derived approximately 85%-specificity
# operating point.
#
# These are NOT clinically validated.
# ============================================================

CLASS_THRESHOLDS = {
    "atelectasis": 0.575,
    "cardiomegaly": 0.635,
    "emphysema": 0.520,
    "opacity": 0.600,
    "pneumothorax": 0.255,
    "pleural_effusion": 0.590,
    "nodule": 0.560,
    "scarring": 0.505,
}


# ============================================================
# IMAGE PREPROCESSING
# Must match V2 classifier training
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"V2 classifier checkpoint not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if "classes" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain 'classes'."
        )

    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    classes = checkpoint["classes"]

    num_classes = checkpoint.get(
        "num_classes",
        len(classes)
    )

    model = models.densenet121(
        weights=None
    )

    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    return (
        model,
        classes,
        checkpoint
    )


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(device)
    )

    return image_tensor


# ============================================================
# PREDICT DISEASES
# ============================================================

def predict_diseases(
    image_path,
    threshold=None
):

    model, classes, checkpoint = (
        load_classifier()
    )

    image_tensor = prepare_image(
        image_path
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

    probabilities = (
        probabilities
        .squeeze(0)
        .cpu()
        .tolist()
    )

    results = []

    for disease, probability in zip(
        classes,
        probabilities
    ):

        disease = str(disease)

        class_threshold = (
            CLASS_THRESHOLDS.get(
                disease,
                0.50
            )
        )

        results.append({
            "finding": disease,
            "probability": float(probability),
            "threshold": float(class_threshold),
            "positive": bool(
                probability >= class_threshold
            )
        })

    results.sort(
        key=lambda item:
            item["probability"],
        reverse=True
    )

    return results


# ============================================================
# BASE-THRESHOLD POSITIVE FINDINGS
# ============================================================

def get_positive_findings(
    image_path,
    threshold=None
):

    results = predict_diseases(
        image_path
    )

    return [
        result
        for result in results
        if result["positive"]
    ]


# ============================================================
# TOP FINDING
# ============================================================

def get_top_finding(image_path):

    results = predict_diseases(
        image_path
    )

    if not results:
        return None

    return results[0]


# ============================================================
# FORMAT PREDICTIONS
# ============================================================

def format_predictions(
    results,
    threshold=None
):

    lines = [
        "Possible Findings / Model Probabilities",
        "-" * 70
    ]

    for result in results:

        finding = (
            result["finding"]
            .replace("_", " ")
            .title()
        )

        probability = (
            result["probability"]
            * 100
        )

        class_threshold = (
            result["threshold"]
            * 100
        )

        marker = ""

        if result["positive"]:

            marker = (
                "  <-- ABOVE BASE THRESHOLD"
            )

        lines.append(
            f"{finding:22s} "
            f"{probability:6.2f}% "
            f"(threshold {class_threshold:.1f}%)"
            f"{marker}"
        )

    lines.extend([
        "",
        (
            "Base thresholds were derived from "
            "validation predictions."
        ),
        "",
        (
            "The final application applies an additional "
            "uncertainty decision layer."
        ),
        "",
        (
            "These are model-generated possible findings, "
            "not definitive diagnoses."
        )
    ])

    return "\n".join(lines)


# ============================================================
# ANALYZE
# ============================================================

def analyze_disease(
    image_path,
    threshold=None
):

    results = predict_diseases(
        image_path
    )

    return (
        results,
        format_predictions(results)
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            'Usage: python src\\predict_disease.py '
            '"path_to_xray.png"'
        )

        return

    image_path = sys.argv[1]

    print("=" * 70)
    print("CHEST X-RAY DISEASE CLASSIFIER V2")
    print("=" * 70)

    print(
        "\nDevice:",
        device
    )

    print(
        "Image:",
        image_path
    )

    print(
        "\nLoading V2 classifier..."
    )

    model, classes, checkpoint = (
        load_classifier()
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Validation loss:",
        checkpoint.get(
            "validation_loss",
            "unknown"
        )
    )

    print(
        "Classes:",
        len(classes)
    )

    results = predict_diseases(
        image_path
    )

    print()

    print(
        format_predictions(results)
    )

    positive = [
        result
        for result in results
        if result["positive"]
    ]

    print()
    print("=" * 70)

    if positive:

        print(
            "FINDING(S) ABOVE BASE THRESHOLD:"
        )

        for result in positive:

            name = (
                result["finding"]
                .replace("_", " ")
                .title()
            )

            print(
                f"- {name}: "
                f"{result['probability'] * 100:.2f}% "
                f"(threshold "
                f"{result['threshold'] * 100:.1f}%)"
            )

    else:

        print(
            "NO SELECTED FINDING ABOVE "
            "ITS BASE THRESHOLD"
        )

    print("=" * 70)

    print()
    print(
        "RESEARCH / EDUCATIONAL PROTOTYPE"
    )

    print(
        "These model outputs are not a medical diagnosis."
    )


if __name__ == "__main__":
    main()
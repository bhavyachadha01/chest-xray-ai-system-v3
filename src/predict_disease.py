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
    / "best_classifier_v3.pth"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "atelectasis",
    "cardiomegaly",
    "emphysema",
    "opacity",
    "pneumothorax",
    "pleural_effusion",
    "nodule",
    "scarring"
]


# ============================================================
# FINAL V3 APPLICATION THRESHOLDS
#
# Derived from V3 VALIDATION data only.
# Target: approximately 95% per-class specificity.
#
# These are conservative application thresholds.
# ============================================================

CLASS_THRESHOLDS = {
    "atelectasis": 0.740,
    "cardiomegaly": 0.640,
    "emphysema": 0.490,
    "opacity": 0.715,
    "pneumothorax": 0.270,
    "pleural_effusion": 0.735,
    "nodule": 0.595,
    "scarring": 0.645,
}


# ============================================================
# LOW-SUPPORT CLASSES
# ============================================================

# Pneumothorax remains unreliable because the dataset
# contains very few positive examples.
#
# Keep its score visible, but do not present it as a
# confident standalone positive finding.

LOW_SUPPORT_CLASSES = {
    "pneumothorax"
}


# ============================================================
# UNCERTAINTY MARGIN
# ============================================================

# 2 percentage points below the final threshold are treated
# as borderline / uncertain.
#
# Example:
#
# threshold = 0.74
# score >= 0.74       -> possible finding
# 0.72 <= score < .74 -> uncertain
# score < 0.72        -> below decision zone

UNCERTAINTY_MARGIN = 0.02


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE PREPROCESSING
#
# Must match V3 validation/test preprocessing.
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD V3 CLASSIFIER
# ============================================================

def load_classifier():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"V3 classifier checkpoint not found:\n"
            f"{MODEL_PATH}"
        )


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )


    if "model_state_dict" not in checkpoint:

        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )


    checkpoint_classes = checkpoint.get(
        "classes",
        CLASSES
    )


    num_classes = checkpoint.get(
        "num_classes",
        len(checkpoint_classes)
    )


    # --------------------------------------------------------
    # DenseNet121 architecture
    # --------------------------------------------------------

    model = models.densenet121(
        weights=None
    )


    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )


    # --------------------------------------------------------
    # Load V3 weights
    # --------------------------------------------------------

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model = model.to(
        device
    )


    model.eval()


    return (
        model,
        checkpoint_classes,
        checkpoint
    )


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )


    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


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

        disease = (
            str(disease)
            .strip()
            .lower()
        )


        class_threshold = (
            CLASS_THRESHOLDS.get(
                disease,
                0.50
            )
        )


        # ----------------------------------------------------
        # ABOVE THRESHOLD
        # ----------------------------------------------------

        above_threshold = (
            probability
            >=
            class_threshold
        )


        # ----------------------------------------------------
        # LOW-SUPPORT CLASS
        # ----------------------------------------------------

        low_support = (
            disease
            in
            LOW_SUPPORT_CLASSES
        )


        # ----------------------------------------------------
        # CONFIDENT POSITIVE
        # ----------------------------------------------------

        # Pneumothorax is excluded from confident positives.

        positive = (
            above_threshold
            and
            not low_support
        )


        # ----------------------------------------------------
        # UNCERTAINTY
        # ----------------------------------------------------

        lower_boundary = (
            class_threshold
            -
            UNCERTAINTY_MARGIN
        )


        uncertain = (
            lower_boundary
            <= probability
            <
            class_threshold
        )


        # If low-support pneumothorax crosses threshold,
        # keep it uncertain instead of positive.

        if (
            low_support
            and
            above_threshold
        ):

            uncertain = True


        results.append({

            "finding":
                disease,

            "probability":
                float(probability),

            "threshold":
                float(class_threshold),

            "above_threshold":
                bool(above_threshold),

            "positive":
                bool(positive),

            "uncertain":
                bool(uncertain),

            "low_support":
                bool(low_support)
        })


    # Highest score first

    results.sort(
        key=lambda item:
            item["probability"],
        reverse=True
    )


    return results


# ============================================================
# POSITIVE FINDINGS
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
# UNCERTAIN FINDINGS
# ============================================================

def get_uncertain_findings(
    image_path
):

    results = predict_diseases(
        image_path
    )


    return [
        result
        for result in results
        if result["uncertain"]
    ]


# ============================================================
# TOP FINDING
# ============================================================

def get_top_finding(
    image_path
):

    results = predict_diseases(
        image_path
    )


    if not results:

        return None


    return results[0]


# ============================================================
# OVERALL STATUS
# ============================================================

def get_model_status(
    results
):

    positive = [
        result
        for result in results
        if result["positive"]
    ]


    uncertain = [
        result
        for result in results
        if result["uncertain"]
    ]


    if positive:

        return (
            "POSSIBLE FINDING DETECTED"
        )


    if uncertain:

        return (
            "MODEL NOT CONFIDENT"
        )


    return (
        "NO SELECTED FINDING DETECTED"
    )


# ============================================================
# FORMAT PREDICTIONS
# ============================================================

def format_predictions(
    results,
    threshold=None
):

    lines = []


    lines.append(
        "Possible Findings / V3 Model Probabilities"
    )


    lines.append(
        "-" * 76
    )


    for result in results:

        finding = (
            result["finding"]
            .replace(
                "_",
                " "
            )
            .title()
        )


        probability = (
            result["probability"]
            *
            100
        )


        class_threshold = (
            result["threshold"]
            *
            100
        )


        marker = ""


        # ----------------------------------------------------
        # POSITIVE
        # ----------------------------------------------------

        if result["positive"]:

            marker = (
                "  <-- POSSIBLE FINDING"
            )


        # ----------------------------------------------------
        # LOW SUPPORT
        # ----------------------------------------------------

        elif (
            result["low_support"]
            and
            result["above_threshold"]
        ):

            marker = (
                "  <-- LOW SUPPORT / UNCERTAIN"
            )


        # ----------------------------------------------------
        # BORDERLINE
        # ----------------------------------------------------

        elif result["uncertain"]:

            marker = (
                "  <-- BORDERLINE / UNCERTAIN"
            )


        lines.append(

            f"{finding:22s}: "

            f"{probability:6.2f}% "

            f"| threshold "

            f"{class_threshold:.1f}%"

            f"{marker}"
        )


    lines.append("")


    lines.append(
        "V3 application thresholds target approximately "
        "95% per-class specificity on the validation split."
    )


    lines.append("")


    lines.append(
        "These conservative thresholds are separate from "
        "the F1-optimized thresholds used for the frozen "
        "held-out test evaluation."
    )


    lines.append("")


    lines.append(
        "Pneumothorax is treated as a low-support "
        "experimental class in this prototype."
    )


    lines.append("")


    lines.append(
        "Model outputs are not medical diagnoses."
    )


    return "\n".join(
        lines
    )


# ============================================================
# ANALYZE DISEASE
# ============================================================

def analyze_disease(
    image_path,
    threshold=None
):

    results = predict_diseases(
        image_path
    )


    formatted_text = (
        format_predictions(
            results
        )
    )


    return (
        results,
        formatted_text
    )


# ============================================================
# COMMAND LINE PROGRAM
# ============================================================

def main():

    # --------------------------------------------------------
    # IMAGE ARGUMENT
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print()

        print(
            "Please provide a chest X-ray image."
        )

        print()

        print(
            "Example:"
        )

        print(
            'python src\\predict_disease.py '
            '"data/images/CXR1045_IM-0036-1001.png"'
        )

        print()

        return


    image_path = (
        sys.argv[1]
    )


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "=" * 76
    )


    print(
        "CHEST X-RAY DISEASE CLASSIFIER V3"
    )


    print(
        "=" * 76
    )


    print()

    print(
        "Device:",
        device
    )


    print(
        "Image:",
        image_path
    )


    print()

    print(
        "Loading V3 classifier..."
    )


    # ========================================================
    # CHECKPOINT INFORMATION
    # ========================================================

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


    print(
        "Fine-tuned blocks:",
        checkpoint.get(
            "fine_tuned_blocks",
            "unknown"
        )
    )


    # ========================================================
    # PREDICT
    # ========================================================

    results = predict_diseases(
        image_path
    )


    print()


    print(
        format_predictions(
            results
        )
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    status = get_model_status(
        results
    )


    positive = [
        result
        for result in results
        if result["positive"]
    ]


    uncertain = [
        result
        for result in results
        if result["uncertain"]
    ]


    print()

    print(
        "=" * 76
    )


    print(
        status
    )


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if positive:

        print()

        for result in positive:

            finding = (
                result["finding"]
                .replace(
                    "_",
                    " "
                )
                .title()
            )


            print(
                f"- {finding}: "
                f"{result['probability'] * 100:.2f}%"
            )


    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain:

        print()

        print(
            "Uncertain / low-support output(s):"
        )


        for result in uncertain:

            finding = (
                result["finding"]
                .replace(
                    "_",
                    " "
                )
                .title()
            )


            extra = ""


            if result["low_support"]:

                extra = (
                    " [LOW-SUPPORT CLASS]"
                )


            print(
                f"- {finding}: "
                f"{result['probability'] * 100:.2f}%"
                f"{extra}"
            )


    print(
        "=" * 76
    )


    # ========================================================
    # SAFETY NOTE
    # ========================================================

    print()

    print(
        "RESEARCH / EDUCATIONAL PROTOTYPE"
    )


    print(
        "The output is not a medical diagnosis."
    )


    print(
        "All chest X-rays require qualified "
        "professional interpretation."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
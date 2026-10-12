
import pandas as pd
from pathlib import Path

# Medical information dataset
DATA_PATH = Path("data/medical_knowledge.csv")


def get_medical_info(disease):
    """
    Find educational medical information
    for a disease or radiology finding.
    """

    if not DATA_PATH.exists():
        return {
            "status": "error",
            "message": "Medical knowledge dataset not found."
        }

    df = pd.read_csv(DATA_PATH).fillna("")

    # Normalize disease names
    disease = str(disease).strip().lower()
    df["finding"] = df["finding"].str.strip().str.lower()

    # Exact matching prevents accidental matches
    matches = df[df["finding"] == disease]

    if matches.empty:
        return {
            "status": "not_found",
            "disease": disease,
            "message": "No verified reference entry available for this finding.",
            "medication_information": "",
            "specialist": "",
            "source_url": ""
        }

    result = matches.iloc[0]

    return {
        "status": "found",
        "disease": disease,
        "medication_information": result["medication_information"],
        "specialist": result["specialist"],
        "source_url": result["source_url"]
    }


# Test the lookup
if __name__ == "__main__":

    disease = "pneumonia"

    result = get_medical_info(disease)

    print("\nMEDICAL INFORMATION")
    print("=" * 50)

    print("Finding:", disease)
    print("Status:", result["status"])

    if result["status"] == "found":
        print("Medication information:", result["medication_information"])
        print("Recommended specialist:", result["specialist"])
        print("Medical source:", result["source_url"])
    else:
        print("Message:", result["message"])

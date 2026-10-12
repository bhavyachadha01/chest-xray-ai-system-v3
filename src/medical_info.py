# ============================================================
# EDUCATIONAL MEDICAL INFORMATION
# Research prototype - NOT prescription
# ============================================================


MEDICAL_INFO = {

    "atelectasis": {
        "treatment": (
            "Management depends on the cause and severity. "
            "Clinical management may include breathing exercises, "
            "incentive spirometry, mobilization, treatment of airway "
            "obstruction, and treatment of the underlying condition. "
            "Medication is selected only after clinical assessment."
        ),
        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        )
    },

    "cardiomegaly": {
        "treatment": (
            "An enlarged cardiac silhouette requires evaluation of "
            "the underlying cause. Treatment may involve management "
            "of blood pressure, heart failure, valve disease, or "
            "other cardiac conditions when clinically confirmed. "
            "Medication depends on the final diagnosis."
        ),
        "doctor": (
            "Cardiologist / General Physician / Radiologist"
        )
    },

    "emphysema": {
        "treatment": (
            "Management may include smoking cessation when relevant, "
            "pulmonary rehabilitation, vaccination, and inhaled "
            "medicines such as bronchodilators when prescribed for "
            "appropriate patients. Treatment requires clinical "
            "confirmation and assessment."
        ),
        "doctor": (
            "Pulmonologist / Respiratory Physician"
        )
    },

    "opacity": {
        "treatment": (
            "Pulmonary opacity is a radiological finding rather than "
            "a single diagnosis. Treatment depends on its cause, "
            "which may include infection, inflammation, fluid, "
            "collapse, or other conditions. Antibiotics are used "
            "only when an appropriate bacterial infection is "
            "clinically diagnosed."
        ),
        "doctor": (
            "Pulmonologist / General Physician / Radiologist"
        )
    },

    "pneumothorax": {
        "treatment": (
            "A pneumothorax can require urgent medical assessment. "
            "Management depends on its size, symptoms and cause and "
            "may range from observation to procedures that remove "
            "air from the pleural space. It should not be treated "
            "from an AI prediction alone."
        ),
        "doctor": (
            "Emergency Physician / Pulmonologist / Thoracic Surgeon"
        )
    },

    "pleural_effusion": {
        "treatment": (
            "Treatment depends on the cause of the pleural fluid. "
            "Clinical management may involve treating infection, "
            "heart disease or another underlying condition, and some "
            "patients may require drainage after medical evaluation."
        ),
        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        )
    },

    "nodule": {
        "treatment": (
            "A pulmonary nodule does not automatically require "
            "medication. Management commonly depends on its size, "
            "appearance, previous imaging and individual risk "
            "factors. Follow-up imaging or further investigation "
            "may be recommended by a clinician."
        ),
        "doctor": (
            "Pulmonologist / Radiologist"
        )
    },

    "scarring": {
        "treatment": (
            "Lung scarring can have several causes. Existing scar "
            "tissue is not generally treated simply with a medicine "
            "based on an X-ray result. Management focuses on finding "
            "and treating any active underlying condition and "
            "assessing respiratory function when appropriate."
        ),
        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        )
    }
}


# ============================================================
# NORMALIZE FINDING NAME
# ============================================================

def normalize_finding(name):

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# ============================================================
# MEDICAL INFORMATION
# ============================================================

def medical_information(predictions):

    if not predictions:

        return (
            "No finding information is available.",
            "General Physician / Radiologist"
        )


    # Only use findings that actually crossed
    # their V2 class-specific threshold.

    positive = []

    for result in predictions:

        if not isinstance(result, dict):
            continue

        if result.get("positive", False):

            finding = normalize_finding(
                result.get(
                    "finding",
                    ""
                )
            )

            if finding:
                positive.append(finding)


    # ========================================================
    # NO THRESHOLD-POSITIVE FINDING
    # ========================================================

    if not positive:

        medication_text = (
            "No selected finding crossed its "
            "validation-tuned threshold.\n\n"
            "No disease-specific medicine can be recommended "
            "from this model output. Clinical evaluation is "
            "required if symptoms or concerns are present.\n\n"
            "This information is educational only and is not "
            "a prescription."
        )

        doctor_text = (
            "General Physician / Radiologist"
        )

        return (
            medication_text,
            doctor_text
        )


    # ========================================================
    # BUILD INFORMATION FOR POSITIVE FINDINGS
    # ========================================================

    treatment_sections = []
    doctors = []


    for finding in positive:

        info = MEDICAL_INFO.get(
            finding
        )

        if info is None:
            continue


        display_name = (
            finding
            .replace("_", " ")
            .title()
        )


        treatment_sections.append(
            f"{display_name}\n"
            f"{info['treatment']}"
        )


        if info["doctor"] not in doctors:
            doctors.append(
                info["doctor"]
            )


    # ========================================================
    # FALLBACK
    # ========================================================

    if not treatment_sections:

        return (
            "General treatment information could not be "
            "matched to the threshold-positive finding(s). "
            "Treatment must be determined after clinical "
            "evaluation.",
            "General Physician / Radiologist"
        )


    # ========================================================
    # FINAL TEXT
    # ========================================================

    medication_text = (
        "\n\n".join(
            treatment_sections
        )
        +
        "\n\nIMPORTANT:\n"
        "This information is educational only. "
        "This system does not prescribe medication or dosage. "
        "Treatment must be selected by a qualified clinician "
        "after confirming the diagnosis."
    )


    doctor_text = (
        "\n".join(
            f"- {doctor}"
            for doctor in doctors
        )
    )


    return (
        medication_text,
        doctor_text
    )
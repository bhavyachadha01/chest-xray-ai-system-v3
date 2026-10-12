import sys
from pathlib import Path

import gradio as gr


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# IMPORT FINAL V3 PREDICTOR
# ============================================================

from predict_disease import (
    predict_diseases,
    CLASS_THRESHOLDS,
    LOW_SUPPORT_CLASSES,
    UNCERTAINTY_MARGIN,
    MODEL_PATH,
)

# Compatibility alias
CLASSIFIER_PATH = MODEL_PATH


# ============================================================
# FINDING INFORMATION
# ============================================================

FINDING_INFO = {

    # ========================================================
    # ATELECTASIS
    # ========================================================

    "atelectasis": {

        "report": (
            "Possible atelectatic change."
        ),

        "guidance": (
            "Atelectasis represents partial collapse or reduced "
            "expansion of part of the lung. Clinical correlation "
            "and professional radiographic review are recommended."
        ),

        "management": (
            "Management depends on the cause and severity. "
            "Clinicians may recommend early mobilization, "
            "deep-breathing exercises, coughing exercises, "
            "chest physiotherapy, or incentive spirometry "
            "in appropriate patients. Treatment of an airway "
            "obstruction or other underlying cause may also "
            "be required."
        ),

        "medication_examples": (
            "There is no single medicine that treats all "
            "atelectasis. Paracetamol (acetaminophen) may be "
            "used for associated mild pain or fever when it is "
            "normally safe for that person. Other medicines "
            "depend on the confirmed underlying cause."
        ),

        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        ),
    },


    # ========================================================
    # CARDIOMEGALY
    # ========================================================

    "cardiomegaly": {

        "report": (
            "Possible enlargement of the cardiac silhouette."
        ),

        "guidance": (
            "An enlarged cardiac silhouette can have several "
            "causes and should be correlated with clinical "
            "findings and professional cardiac/radiological "
            "assessment."
        ),

        "management": (
            "Management depends on the condition responsible "
            "for the apparent cardiac enlargement. Additional "
            "assessment such as clinical examination, ECG, "
            "echocardiography, or other investigations may be "
            "considered by a healthcare professional."
        ),

        "medication_examples": (
            "There is no general over-the-counter medicine "
            "for cardiomegaly itself. Depending on a confirmed "
            "cardiac diagnosis, clinicians may use medicines "
            "such as diuretics, ACE inhibitors, ARBs, beta "
            "blockers, or other cardiac medicines. These require "
            "individual medical assessment and should not be "
            "started because of an AI X-ray result."
        ),

        "doctor": (
            "Cardiologist / General Physician / Radiologist"
        ),
    },


    # ========================================================
    # EMPHYSEMA
    # ========================================================

    "emphysema": {

        "report": (
            "Possible emphysematous change."
        ),

        "guidance": (
            "Possible emphysematous change should be correlated "
            "with respiratory symptoms, smoking/exposure history, "
            "pulmonary function testing, and professional "
            "radiographic interpretation."
        ),

        "management": (
            "Management may include smoking cessation when "
            "relevant, avoidance of respiratory irritants, "
            "vaccination, pulmonary rehabilitation, exercise "
            "and respiratory follow-up."
        ),

        "medication_examples": (
            "Depending on symptoms and confirmed COPD/emphysema, "
            "clinicians may prescribe bronchodilator inhalers, "
            "including beta-agonist or antimuscarinic inhalers. "
            "Inhaled corticosteroids may be used in selected "
            "patients rather than routinely for every patient."
        ),

        "doctor": (
            "Pulmonologist / Respiratory Physician"
        ),
    },


    # ========================================================
    # OPACITY
    # ========================================================

    "opacity": {

        "report": (
            "Possible pulmonary opacity."
        ),

        "guidance": (
            "Pulmonary opacity is an imaging finding rather "
            "than a single diagnosis. Causes can include "
            "infection, inflammation, atelectatic change, "
            "fluid, a mass, or other processes. Professional "
            "interpretation is recommended."
        ),

        "management": (
            "Management depends entirely on the cause. "
            "Clinical symptoms, examination, laboratory tests, "
            "comparison with previous imaging, or additional "
            "imaging may be required before treatment is chosen."
        ),

        "medication_examples": (
            "For associated mild fever or discomfort, "
            "paracetamol (acetaminophen) may be appropriate "
            "for some people when normally safe for them. "
            "Antibiotics should only be used when bacterial "
            "infection is clinically suspected or confirmed "
            "and a clinician determines they are indicated."
        ),

        "doctor": (
            "Pulmonologist / General Physician / Radiologist"
        ),
    },


    # ========================================================
    # PNEUMOTHORAX
    # ========================================================

    "pneumothorax": {

        "report": (
            "Elevated pneumothorax model output."
        ),

        "guidance": (
            "Pneumothorax remains a low-support experimental "
            "class in this prototype because relatively few "
            "positive examples were available. The AI output "
            "must not be interpreted as a confirmed "
            "pneumothorax diagnosis."
        ),

        "management": (
            "A clinically suspected pneumothorax requires "
            "medical assessment. Depending on its size, cause, "
            "symptoms and patient stability, professional "
            "management may include observation, supplemental "
            "oxygen, needle aspiration, chest drainage, or "
            "other intervention."
        ),

        "medication_examples": (
            "There is no home medication that repairs a "
            "collapsed lung. Pain medicines may sometimes be "
            "used as part of clinician-directed care, but "
            "definitive management depends on the severity "
            "and cause of the pneumothorax."
        ),

        "doctor": (
            "Emergency Physician / Pulmonologist / Radiologist"
        ),
    },


    # ========================================================
    # PLEURAL EFFUSION
    # ========================================================

    "pleural_effusion": {

        "report": (
            "Possible pleural effusion."
        ),

        "guidance": (
            "Possible pleural fluid should be assessed in "
            "clinical context because pleural effusion can "
            "have several underlying causes."
        ),

        "management": (
            "Treatment targets the underlying cause. "
            "Some patients may require additional imaging, "
            "diagnostic sampling, or therapeutic drainage "
            "depending on the size and clinical situation."
        ),

        "medication_examples": (
            "Medication depends on the confirmed cause. "
            "For example, clinicians may use diuretics for "
            "appropriate heart-failure-related effusions or "
            "antibiotics for bacterial infection. These "
            "medicines should not be selected from the "
            "X-ray model output alone."
        ),

        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        ),
    },


    # ========================================================
    # NODULE
    # ========================================================

    "nodule": {

        "report": (
            "Possible pulmonary nodule."
        ),

        "guidance": (
            "A possible pulmonary nodule generally requires "
            "professional assessment rather than immediate "
            "symptom treatment. Previous imaging and individual "
            "risk factors can be important."
        ),

        "management": (
            "Management may involve comparison with previous "
            "imaging, interval follow-up imaging, CT evaluation, "
            "or additional investigation depending on size, "
            "appearance and clinical risk."
        ),

        "medication_examples": (
            "There is no standard medication for an unidentified "
            "pulmonary nodule. Treatment, if required, depends "
            "on the eventual diagnosis. Medication should not "
            "be started solely because the model produced an "
            "elevated nodule score."
        ),

        "doctor": (
            "Pulmonologist / Radiologist"
        ),
    },


    # ========================================================
    # SCARRING
    # ========================================================

    "scarring": {

        "report": (
            "Possible chronic pulmonary scarring-related change."
        ),

        "guidance": (
            "Possible pulmonary scarring should be interpreted "
            "in conjunction with symptoms, previous imaging, "
            "medical history, and professional radiographic "
            "assessment."
        ),

        "management": (
            "Management depends on the cause, extent, symptoms "
            "and whether there is active or progressive lung "
            "disease. Some patients may require pulmonary "
            "function testing or specialist follow-up."
        ),

        "medication_examples": (
            "There is no over-the-counter medicine that removes "
            "established lung scarring. Some specific progressive "
            "fibrotic lung diseases may be treated with "
            "specialist-prescribed antifibrotic or other "
            "disease-specific medicines after a confirmed "
            "diagnosis."
        ),

        "doctor": (
            "Pulmonologist / Respiratory Physician / Radiologist"
        ),
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_name(name):

    return (
        str(name)
        .replace("_", " ")
        .strip()
        .title()
    )


def get_positive_findings(results):

    return [
        result
        for result in results
        if result.get(
            "positive",
            False
        )
    ]


def get_uncertain_findings(results):

    return [
        result
        for result in results
        if result.get(
            "uncertain",
            False
        )
    ]


# ============================================================
# MODEL ASSESSMENT
# ============================================================

def create_status(results):

    positive = get_positive_findings(
        results
    )

    uncertain = get_uncertain_findings(
        results
    )


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if positive:

        lines = [
            "POSSIBLE FINDING DETECTED",
            ""
        ]


        for result in positive:

            lines.append(
                f"- {display_name(result['finding'])}: "
                f"{result['probability'] * 100:.2f}% "
                f"(threshold "
                f"{result['threshold'] * 100:.1f}%)"
            )


        if uncertain:

            lines.extend([
                "",
                "Additional uncertain model output(s):"
            ])


            for result in uncertain:

                extra = ""

                if result.get(
                    "low_support",
                    False
                ):

                    extra = (
                        " [LOW-SUPPORT CLASS]"
                    )


                lines.append(
                    f"- {display_name(result['finding'])}: "
                    f"{result['probability'] * 100:.2f}%"
                    f"{extra}"
                )


        return "\n".join(
            lines
        )


    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain:

        lines = [
            "MODEL NOT CONFIDENT",
            "",
            "Borderline / low-support model output(s):"
        ]


        for result in uncertain:

            extra = ""

            if result.get(
                "low_support",
                False
            ):

                extra = (
                    " [LOW-SUPPORT CLASS]"
                )


            lines.append(
                f"- {display_name(result['finding'])}: "
                f"{result['probability'] * 100:.2f}%"
                f"{extra}"
            )


        lines.extend([
            "",
            (
                "Professional radiologist review "
                "is recommended."
            )
        ])


        return "\n".join(
            lines
        )


    # --------------------------------------------------------
    # NO FINDING
    # --------------------------------------------------------

    return (
        "NO SELECTED FINDING DETECTED\n\n"
        "None of the eight selected findings crossed the "
        "conservative V3 application threshold or its "
        "borderline decision zone.\n\n"
        "This does NOT establish that the chest X-ray "
        "is normal."
    )


# ============================================================
# MODEL SCORES
# ============================================================

def create_scores(results):

    lines = []


    for result in results:

        finding = display_name(
            result["finding"]
        )


        probability = (
            float(
                result["probability"]
            )
            * 100
        )


        threshold = (
            float(
                result["threshold"]
            )
            * 100
        )


        marker = ""


        if result.get(
            "positive",
            False
        ):

            marker = (
                "  <-- POSSIBLE FINDING"
            )


        elif (
            result.get(
                "low_support",
                False
            )
            and
            result.get(
                "above_threshold",
                False
            )
        ):

            marker = (
                "  <-- LOW SUPPORT / UNCERTAIN"
            )


        elif result.get(
            "uncertain",
            False
        ):

            marker = (
                "  <-- BORDERLINE / UNCERTAIN"
            )


        lines.append(
            f"{finding:22s}: "
            f"{probability:6.2f}% "
            f"| threshold "
            f"{threshold:.1f}%"
            f"{marker}"
        )


    return "\n".join(
        lines
    )


# ============================================================
# CLASSIFIER-GUIDED RADIOLOGY SUMMARY
# ============================================================

def create_report(results):

    positive = get_positive_findings(
        results
    )

    uncertain = get_uncertain_findings(
        results
    )


    lines = [
        "CLASSIFIER-GUIDED RADIOLOGY SUMMARY",
        "",
        "FINDINGS:"
    ]


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if positive:

        for result in positive:

            finding = result[
                "finding"
            ]


            info = FINDING_INFO.get(
                finding
            )


            if info:

                lines.append(
                    f"- {info['report']} "
                    f"(model score "
                    f"{result['probability'] * 100:.2f}%)"
                )


        if uncertain:

            lines.extend([
                "",
                "Additional uncertain model output(s):"
            ])


            for result in uncertain:

                extra = ""

                if result.get(
                    "low_support",
                    False
                ):

                    extra = (
                        " [LOW-SUPPORT CLASS]"
                    )


                lines.append(
                    f"- {display_name(result['finding'])}: "
                    f"{result['probability'] * 100:.2f}%"
                    f"{extra}"
                )


        lines.extend([
            "",
            "IMPRESSION:",
            (
                "One or more selected classifier findings "
                "crossed the conservative V3 application "
                "threshold."
            ),
            "",
            "RECOMMENDATION:",
            (
                "Correlation with the original chest "
                "radiograph, clinical information, and "
                "professional radiologist interpretation "
                "is recommended."
            )
        ])


    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    elif uncertain:

        lines.append(
            "No selected finding crossed a supported "
            "positive threshold."
        )


        lines.extend([
            "",
            "Borderline / low-support output(s):"
        ])


        for result in uncertain:

            extra = ""

            if result.get(
                "low_support",
                False
            ):

                extra = (
                    " [LOW-SUPPORT CLASS]"
                )


            lines.append(
                f"- {display_name(result['finding'])}: "
                f"{result['probability'] * 100:.2f}%"
                f"{extra}"
            )


        lines.extend([
            "",
            "IMPRESSION:",
            (
                "No confident classifier finding. "
                "One or more model outputs are borderline "
                "or low-support and require professional review."
            ),
            "",
            "RECOMMENDATION:",
            (
                "Professional radiologist interpretation "
                "is recommended."
            )
        ])


    # --------------------------------------------------------
    # NO FINDING
    # --------------------------------------------------------

    else:

        lines.extend([
            (
                "No selected classifier finding crossed "
                "the conservative V3 application threshold."
            ),
            "",
            "IMPRESSION:",
            (
                "No threshold-positive finding among the "
                "eight selected conditions evaluated by "
                "this research classifier."
            ),
            "",
            "RECOMMENDATION:",
            (
                "This result does not establish a normal "
                "chest radiograph. Professional review "
                "should be obtained when clinically indicated."
            )
        ])


    lines.extend([
        "",
        (
            "This is a classifier-guided automated summary "
            "for research use. It is not a radiologist's "
            "report or a medical diagnosis."
        )
    ])


    return "\n".join(
        lines
    )


# ============================================================
# GENERAL GUIDANCE
# ============================================================

def create_guidance(results):

    positive = get_positive_findings(
        results
    )

    uncertain = get_uncertain_findings(
        results
    )


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    if positive:

        sections = []


        for result in positive:

            finding = result[
                "finding"
            ]


            info = FINDING_INFO.get(
                finding
            )


            if info:

                sections.append(
                    f"{display_name(finding)}\n\n"
                    f"{info['guidance']}"
                )


        return (
            "\n\n"
            "----------------------------------------"
            "\n\n"
        ).join(
            sections
        )


    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if uncertain:

        return (
            "The classifier result is uncertain.\n\n"
            "One or more model scores are borderline or "
            "belong to a low-support experimental class.\n\n"
            "Professional interpretation is recommended "
            "rather than treating this result as "
            "definitively normal or abnormal."
        )


    # --------------------------------------------------------
    # NO FINDING
    # --------------------------------------------------------

    return (
        "No selected finding crossed the conservative V3 "
        "application threshold.\n\n"
        "This does not prove that the chest X-ray is normal. "
        "The V3 model can miss abnormalities at this "
        "high-specificity operating point.\n\n"
        "Symptoms, medical history, physical examination, "
        "and professional radiology review should still "
        "be considered."
    )


# ============================================================
# MANAGEMENT / MEDICATION INFORMATION
# ============================================================

def create_treatment(results):

    positive = get_positive_findings(
        results
    )

    uncertain = get_uncertain_findings(
        results
    )


    # ========================================================
    # CLEAR POSITIVE FINDING
    # ========================================================

    if positive:

        sections = []


        for result in positive:

            finding = result[
                "finding"
            ]


            info = FINDING_INFO.get(
                finding
            )


            if not info:

                continue


            section = (
                f"{display_name(finding)}\n\n"
                f"MANAGEMENT:\n"
                f"{info['management']}\n\n"
                f"MEDICATION / TREATMENT EXAMPLES:\n"
                f"{info['medication_examples']}"
            )


            sections.append(
                section
            )


        if sections:

            text = (
                "\n\n"
                "========================================"
                "\n\n"
            ).join(
                sections
            )


        else:

            text = (
                "No finding-specific educational treatment "
                "information is available."
            )


        text += (
            "\n\n"
            "IMPORTANT EDUCATIONAL DISCLAIMER:\n"
            "The management and medication examples shown "
            "above are general educational information only. "
            "They are NOT prescriptions, diagnoses, or "
            "individual treatment recommendations.\n\n"
            "Medication must not be selected solely from an "
            "AI chest X-ray finding. A qualified healthcare "
            "professional must first confirm the diagnosis "
            "and consider symptoms, medical history, allergies, "
            "other medicines, pregnancy status when relevant, "
            "kidney/liver function, and contraindications."
        )


        return text


    # ========================================================
    # UNCERTAIN RESULT
    # ========================================================

    if uncertain:

        lines = [
            "GENERAL SUPPORTIVE / SYMPTOM-RELIEF INFORMATION",
            "",
            (
                "The classifier result is uncertain, so "
                "disease-specific medication or treatment "
                "should not be selected from this result."
            ),
            "",
            (
                "For mild pain or fever, paracetamol "
                "(acetaminophen) may be appropriate for "
                "some people when it can normally be "
                "taken safely."
            ),
            "",
            (
                "Rest, adequate hydration, and avoiding "
                "smoking or other respiratory irritants "
                "may be reasonable general supportive measures."
            ),
            "",
            (
                "Do not start antibiotics, inhalers, steroids, "
                "diuretics, antifibrotic medicines, or other "
                "prescription medicines based only on this "
                "AI output."
            ),
            "",
            (
                "Professional medical assessment is recommended "
                "when symptoms are present or persistent."
            ),
        ]


        # ----------------------------------------------------
        # PNEUMOTHORAX SAFETY NOTE
        # ----------------------------------------------------

        if any(
            result.get(
                "finding"
            )
            ==
            "pneumothorax"

            for result in uncertain
        ):

            lines.extend([
                "",
                "PNEUMOTHORAX SAFETY NOTE:",
                (
                    "The pneumothorax output is a low-support "
                    "experimental model result and must not "
                    "be interpreted as a confirmed diagnosis."
                ),
                "",
                (
                    "Seek urgent medical assessment for severe "
                    "or sudden chest pain, significant breathing "
                    "difficulty, fainting, bluish lips or skin, "
                    "or another emergency symptom."
                ),
            ])


        lines.extend([
            "",
            "IMPORTANT EDUCATIONAL DISCLAIMER:",
            (
                "This information is general education only "
                "and is not a prescription or treatment plan."
            )
        ])


        return "\n".join(
            lines
        )


    # ========================================================
    # NO SELECTED FINDING
    # ========================================================

    return (
        "NO FINDING-SPECIFIC TREATMENT SUGGESTED\n\n"
        "The V3 classifier did not identify a selected finding "
        "above its conservative application threshold.\n\n"
        "No medication should be selected from this AI result "
        "alone.\n\n"
        "If symptoms such as persistent cough, fever, chest "
        "pain, or breathing difficulty are present, medical "
        "assessment may still be appropriate because this "
        "model can miss abnormalities.\n\n"
        "IMPORTANT EDUCATIONAL DISCLAIMER:\n"
        "This information is general education only and is "
        "not a prescription or medical treatment plan."
    )


# ============================================================
# SPECIALIST RECOMMENDATION
# ============================================================

def create_specialist(results):

    positive = get_positive_findings(
        results
    )

    uncertain = get_uncertain_findings(
        results
    )


    # --------------------------------------------------------
    # UNCERTAIN
    # --------------------------------------------------------

    if (
        not positive
        and
        uncertain
    ):

        has_pneumothorax = any(
            result.get(
                "finding"
            )
            ==
            "pneumothorax"

            for result in uncertain
        )


        if has_pneumothorax:

            return (
                "Radiologist / General Physician / Pulmonologist\n\n"
                "The pneumothorax model score is a low-support "
                "experimental output and is not a diagnosis.\n\n"
                "Urgent medical assessment is appropriate for "
                "severe chest pain or significant breathing "
                "difficulty."
            )


        return (
            "General Physician / Radiologist\n\n"
            "The classifier result is uncertain. "
            "Professional radiology review is recommended."
        )


    # --------------------------------------------------------
    # NO POSITIVE
    # --------------------------------------------------------

    if not positive:

        return (
            "General Physician / Radiologist"
        )


    # --------------------------------------------------------
    # POSITIVE
    # --------------------------------------------------------

    doctors = []


    for result in positive:

        finding = result[
            "finding"
        ]


        info = FINDING_INFO.get(
            finding
        )


        if info:

            doctor = info[
                "doctor"
            ]


            if doctor not in doctors:

                doctors.append(
                    doctor
                )


    if not doctors:

        doctors.append(
            "General Physician / Radiologist"
        )


    text = "\n".join(
        f"- {doctor}"
        for doctor in doctors
    )


    text += (
        "\n\nSeek urgent medical care for severe chest pain, "
        "major breathing difficulty, fainting, bluish lips "
        "or skin, or another medical emergency."
    )


    return text


# ============================================================
# ANALYZE X-RAY
# ============================================================

def analyze_xray(image_path):

    # --------------------------------------------------------
    # NO IMAGE
    # --------------------------------------------------------

    if not image_path:

        return (
            "Please upload or paste a chest X-ray.",
            "",
            "",
            "",
            "",
            ""
        )


    # --------------------------------------------------------
    # RUN V3
    # --------------------------------------------------------

    try:

        results = predict_diseases(
            image_path
        )


    except Exception as error:

        return (
            "CLASSIFIER ERROR",
            str(error),
            "",
            "",
            "",
            ""
        )


    # --------------------------------------------------------
    # EMPTY RESULT
    # --------------------------------------------------------

    if not results:

        return (
            "CLASSIFIER ERROR",
            "No model predictions were returned.",
            "",
            "",
            "",
            ""
        )


    # --------------------------------------------------------
    # RETURN OUTPUTS
    # --------------------------------------------------------

    return (
        create_status(
            results
        ),

        create_scores(
            results
        ),

        create_report(
            results
        ),

        create_guidance(
            results
        ),

        create_treatment(
            results
        ),

        create_specialist(
            results
        )
    )


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Chest X-Ray AI Analysis System V3"
) as demo:


    # ========================================================
    # HEADER
    # ========================================================

    gr.Markdown(
        """
# Chest X-Ray AI Analysis System — V3

Upload, drag-and-drop, or paste a chest X-ray and click
**Analyze Chest X-Ray**.

### System Outputs

- V3 DenseNet121 finding classifier
- Eight finding model scores
- Possible Finding / Model Not Confident /
  No Selected Finding Detected
- Classifier-guided structured radiology summary
- General guidance
- Educational disease-specific management and medication examples
- Recommended medical specialist

**Research and educational prototype only.  
This system does not provide a medical diagnosis,
radiologist's report, prescription, or individualized
treatment plan.**
"""
    )


    # ========================================================
    # IMAGE INPUT
    # ========================================================

    image_input = gr.Image(
        label="Upload / Paste Chest X-Ray",
        type="filepath",
        sources=[
            "upload",
            "clipboard"
        ],
        height=400
    )


    # ========================================================
    # BUTTONS
    # ========================================================

    with gr.Row():

        analyze_button = gr.Button(
            "Analyze Chest X-Ray",
            variant="primary"
        )


        clear_button = gr.Button(
            "Clear"
        )


    # ========================================================
    # MODEL ASSESSMENT
    # ========================================================

    status_output = gr.Textbox(
        label="V3 Model Assessment",
        lines=9,
        interactive=False
    )


    # ========================================================
    # MODEL SCORES
    # ========================================================

    scores_output = gr.Textbox(
        label="Possible Findings / V3 Model Scores",
        lines=12,
        interactive=False
    )


    # ========================================================
    # RADIOLOGY SUMMARY
    # ========================================================

    report_output = gr.Textbox(
        label="Classifier-Guided Radiology Summary",
        lines=15,
        interactive=False
    )


    # ========================================================
    # GUIDANCE
    # ========================================================

    guidance_output = gr.Textbox(
        label="General Guidance",
        lines=10,
        interactive=False
    )


    # ========================================================
    # MANAGEMENT / MEDICATION
    # ========================================================

    treatment_output = gr.Textbox(
        label=(
            "Educational Management / "
            "Medication & Treatment Examples"
        ),
        lines=20,
        interactive=False
    )


    # ========================================================
    # SPECIALIST
    # ========================================================

    specialist_output = gr.Textbox(
        label="Recommended Doctor / Specialist",
        lines=8,
        interactive=False
    )


    # ========================================================
    # ANALYZE EVENT
    # ========================================================

    analyze_button.click(

        fn=analyze_xray,

        inputs=[
            image_input
        ],

        outputs=[
            status_output,
            scores_output,
            report_output,
            guidance_output,
            treatment_output,
            specialist_output
        ]
    )


    # ========================================================
    # CLEAR EVENT
    # ========================================================

    clear_button.click(

        fn=lambda: (
            None,
            "",
            "",
            "",
            "",
            "",
            ""
        ),

        inputs=[],

        outputs=[
            image_input,
            status_output,
            scores_output,
            report_output,
            guidance_output,
            treatment_output,
            specialist_output
        ]
    )


    # ========================================================
    # FINAL DISCLAIMER
    # ========================================================

    gr.Markdown(
        """
---

## Important Research / Safety Information

The V3 classifier evaluates eight selected findings:

**Atelectasis · Cardiomegaly · Emphysema · Opacity ·
Pneumothorax · Pleural Effusion · Nodule · Scarring**

The application uses conservative class-specific thresholds
derived from the **V3 validation split**, targeting approximately
**95% per-class specificity**.

This high-specificity operating point reduces false-positive
predictions but can miss true abnormalities.

**Pneumothorax remains a low-support experimental class** and is
not presented as a confident standalone positive finding.

Displayed percentages are machine-learning model scores,
not confirmed probabilities of disease.

The structured summary is classifier-guided and is **not a
radiologist's report**.

### Medication / Treatment Disclaimer

Medication and treatment examples displayed by this application
are provided **only for educational purposes**.

They are **not prescriptions, diagnoses, or individualized
treatment recommendations**.

A medication must not be selected solely from an AI chest X-ray
finding. Appropriate treatment requires a confirmed diagnosis and
consideration of symptoms, medical history, allergies, other
medications, pregnancy status when relevant, kidney/liver
function, contraindications, and professional clinical judgment.

Do not start, stop, or change prescription medication because of
this application's output.

All chest X-rays and AI-generated outputs require review by
qualified healthcare professionals.
"""
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print(
        "FINAL CHEST X-RAY AI SYSTEM V3"
    )
    print("=" * 72)


    print(
        "Model:",
        MODEL_PATH
    )


    print(
        "V3 model found:",
        MODEL_PATH.exists()
    )


    print(
        "Uncertainty margin:",
        f"{UNCERTAINTY_MARGIN * 100:.0f}%"
    )


    print(
        "Low-support classes:",
        LOW_SUPPORT_CLASSES
    )


    print()
    print(
        "Application thresholds:"
    )


    for name, threshold in (
        CLASS_THRESHOLDS.items()
    ):

        print(
            f"  {name:20s}: "
            f"{threshold * 100:.1f}%"
        )


    print("=" * 72)
    print()


    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True
    )
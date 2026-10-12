import os
import csv
import xml.etree.ElementTree as ET

REPORT_DIR = "data/reports/ecgen-radiology"
IMAGE_DIR = "data/images"
OUTPUT_FILE = "data/dataset.csv"

rows = []

for filename in os.listdir(REPORT_DIR):

    if not filename.endswith(".xml"):
        continue

    xml_path = os.path.join(REPORT_DIR, filename)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        findings = ""
        impression = ""

        # Extract report text
        for element in root.iter():

            if element.tag.endswith("AbstractText"):

                label = element.attrib.get("Label", "").upper()

                text = "".join(element.itertext()).strip()

                if label == "FINDINGS":
                    findings = text

                elif label == "IMPRESSION":
                    impression = text

        # Extract image IDs
        image_ids = []

        for element in root.iter():

            if element.tag.endswith("parentImage"):

                image_id = element.attrib.get("id")

                if image_id:
                    image_ids.append(image_id)

        # Match report with actual PNG
        for image_id in image_ids:

            image_name = image_id + ".png"
            image_path = os.path.join(IMAGE_DIR, image_name)

            if os.path.exists(image_path):

                rows.append({
                    "report_id": filename.replace(".xml", ""),
                    "image_name": image_name,
                    "image_path": image_path,
                    "findings": findings,
                    "impression": impression
                })

    except Exception as e:
        print(f"ERROR {filename}: {e}")


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "report_id",
            "image_name",
            "image_path",
            "findings",
            "impression"
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


print("--------------------------------")
print("DATASET CREATED")
print("--------------------------------")
print("XML reports:", len(os.listdir(REPORT_DIR)))
print("Matched image-report pairs:", len(rows))
print("Saved:", OUTPUT_FILE)
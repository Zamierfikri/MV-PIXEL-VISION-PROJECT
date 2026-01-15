# ==========================================================
# FIX OPENMP ISSUE (MUST BE FIRST)
# ==========================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==========================================================
# IMPORTS
# ==========================================================
from ultralytics import YOLO
import cv2
import easyocr
import re

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "bestnya6.pt"

FOLDER_MAP = {
    "random_images_01": "output_folder_1_easyOCR",
    "random_images_02": "output_folder_2_easyOCR"
}

# ==========================================================
# INITIALIZE MODEL & OCR
# ==========================================================
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ==========================================================
# CLEAN PLATE TEXT
# ==========================================================
def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

# ==========================================================
# REORDER PLATE TEXT: ALPHABET FIRST, THEN NUMBER
# ==========================================================
def reorder_plate_text(text):
    letters = ''.join([c for c in text if c.isalpha()])
    numbers = ''.join([c for c in text if c.isdigit()])
    return letters + numbers

# ==========================================================
# DRAW STATUS AT TOP-RIGHT (SUPPORT MULTIPLE PLATES)
# ==========================================================
def draw_status_top_right(img, texts):
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.2
    thickness = 4

    if isinstance(texts, list):
        text = " | ".join(texts)   # 🔹 combine all plates
    else:
        text = texts

    (text_w, text_h), _ = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x = w - text_w - 30
    y = text_h + 40

    cv2.rectangle(
        img,
        (x - 10, y - text_h - 10),
        (x + text_w + 10, y + 10),
        (0, 0, 0),
        -1
    )

    color = (0, 255, 0) if text != "NO PLATE DETECTED" else (0, 0, 255)

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness
    )

# ==========================================================
# PROCESS SINGLE IMAGE
# ==========================================================
def process_image(image_path, output_path, txt_path):
    img = cv2.imread(image_path)
    if img is None:
        return False

    results = model(img)[0]
    detected_plates = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]

        if plate_crop.size == 0:
            continue

        plate_crop = cv2.resize(
            plate_crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC
        )

        ocr_results = reader.readtext(
            plate_crop,
            detail=1,
            paragraph=False
        )

        plate_text = ""

        ocr_results = sorted(ocr_results, key=lambda r: r[0][0][0])

        for _, text, conf in ocr_results:
            plate_text += text + " "

        plate_text = clean_plate_text(plate_text)
        plate_text = reorder_plate_text(plate_text)

        if plate_text:
            detected_plates.append(plate_text)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if detected_plates:
        status_text = detected_plates          # 🔹 LIST now
        detected = True
    else:
        status_text = "NO PLATE DETECTED"
        detected = False

    draw_status_top_right(img, status_text)

    cv2.imwrite(output_path, img)

    with open(txt_path, "w") as f:
        if detected_plates:
            for plate in detected_plates:
                f.write(plate + "\n")
        else:
            f.write("NO PLATE DETECTED")

    return detected

# ==========================================================
# MAIN LOOP
# ==========================================================
overall_detected = 0
overall_not_detected = 0

for input_folder, output_folder in FOLDER_MAP.items():

    print(f"\n📂 Processing folder: {input_folder}")
    os.makedirs(output_folder, exist_ok=True)

    folder_detected = 0
    folder_not_detected = 0
    total_images = 0

    for file in os.listdir(input_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            total_images += 1

            input_path = os.path.join(input_folder, file)
            output_img_path = os.path.join(output_folder, file)
            output_txt_path = os.path.join(
                output_folder, file.rsplit('.', 1)[0] + ".txt"
            )

            detected = process_image(
                input_path, output_img_path, output_txt_path
            )

            if detected:
                folder_detected += 1
                overall_detected += 1
                print(f"  ✅ {file} → PLATE(S) DETECTED")
            else:
                folder_not_detected += 1
                overall_not_detected += 1
                print(f"  ❌ {file} → NO PLATE DETECTED")

    print(f"\n📊 Summary for {input_folder}:")
    print(f"   Total images        : {total_images}")
    print(f"   Plate detected      : {folder_detected}")
    print(f"   No plate detected   : {folder_not_detected}")
    print("-" * 50)

print("\n📈 OVERALL SUMMARY")
print(f"   Total detected      : {overall_detected}")
print(f"   Total not detected  : {overall_not_detected}")
print("\n✅ ALL FOLDERS PROCESSED SUCCESSFULLY")
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
import numpy as np

# ==========================================================
# CONFIGURATION
# ==========================================================
MODEL_PATH = "bestnya6.pt"

FOLDER_MAP = {
    "random_images_01": "output_folder_1_adaptiveOCR",
    "random_images_02": "output_folder_2_adaptiveOCR"
}

NMS_IOU_THRESHOLD = 0.5
OCR_CONF_THRESHOLD = 0.40
SCALES = [1.5, 2.0, 2.5]

FAILED_DIR = "failed_ocr"
os.makedirs(FAILED_DIR, exist_ok=True)

# ==========================================================
# INITIALIZE MODEL & OCR
# ==========================================================
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ==========================================================
# TEXT UTILITIES (MALAYSIA PLATE RULES)
# ==========================================================
def clean_plate_text(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace('O', '').replace('I', '')
    return text

def reorder_plate_text(text):
    letters = ''.join(c for c in text if c.isalpha())
    numbers = ''.join(c for c in text if c.isdigit())
    return letters + numbers

def valid_malaysia_plate(text):
    pattern = r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]?$"
    return bool(re.match(pattern, text))

# ==========================================================
# IOU + NMS
# ==========================================================
def compute_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    union = areaA + areaB - inter
    return 0 if union == 0 else inter / union

def non_max_suppression(boxes, scores, thr):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if compute_iou(boxes[i], boxes[j]) < thr]
    return keep

# ==========================================================
# IMAGE ENHANCEMENT
# ==========================================================
def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    return clahe.apply(gray)

# ==========================================================
# OCR QUALITY
# ==========================================================
def ocr_quality(results):
    if not results:
        return 0
    return np.mean([r[2] for r in results])

# ==========================================================
# MULTI-SCALE OCR WITH VOTING
# ==========================================================
def multi_scale_ocr(crop):
    best_text = ""
    best_score = 0

    for scale in SCALES:
        resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(
            resized,
            detail=1,
            paragraph=False,
            allowlist='ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
        )

        # sort top-left → bottom-right
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        text = ""
        for _, t, conf in results:
            if conf >= OCR_CONF_THRESHOLD:
                text += t + " "

        text = reorder_plate_text(clean_plate_text(text))
        score = ocr_quality(results)

        if score > best_score and valid_malaysia_plate(text):
            best_text = text
            best_score = score

    return best_text, best_score

# ==========================================================
# DRAW STATUS
# ==========================================================
def draw_status_top_right(img, texts):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 2.2, 4

    text = " | ".join(texts) if isinstance(texts, list) else texts
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)

    x, y = w - tw - 30, th + 40
    cv2.rectangle(img, (x-10, y-th-10), (x+tw+10, y+10), (0,0,0), -1)
    color = (0,255,0) if text != "NO PLATE DETECTED" else (0,0,255)
    cv2.putText(img, text, (x,y), font, scale, color, thick)

# ==========================================================
# PROCESS IMAGE
# ==========================================================
def process_image(img_path, out_img, out_txt):
    img = cv2.imread(img_path)
    if img is None:
        return False

    results = model(img)[0]
    boxes, scores = [], []

    for b in results.boxes:
        x1,y1,x2,y2 = map(int, b.xyxy[0])
        boxes.append([x1,y1,x2,y2])
        scores.append(float(b.conf[0]))

    keep = non_max_suppression(boxes, scores, NMS_IOU_THRESHOLD)
    detected_plates = []

    for i in keep:
        x1,y1,x2,y2 = boxes[i]
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop = cv2.resize(crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

        # BASIC OCR
        text, score = multi_scale_ocr(crop)

        # ENHANCED OCR IF NEEDED
        if not text:
            enhanced = enhance_for_ocr(crop)
            text, score = multi_scale_ocr(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

        if text:
            detected_plates.append(text)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        else:
            cv2.imwrite(os.path.join(FAILED_DIR, os.path.basename(img_path)), crop)

    draw_status_top_right(img, detected_plates if detected_plates else "NO PLATE DETECTED")
    cv2.imwrite(out_img, img)

    with open(out_txt, "w") as f:
        f.write("\n".join(detected_plates) if detected_plates else "NO PLATE DETECTED")

    return bool(detected_plates)

# ==========================================================
# MAIN LOOP
# ==========================================================
detected, not_detected = 0, 0

for in_dir, out_dir in FOLDER_MAP.items():
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n📂 Processing {in_dir}")

    for file in os.listdir(in_dir):
        if file.lower().endswith(('.jpg','.png','.jpeg')):
            ok = process_image(
                os.path.join(in_dir, file),
                os.path.join(out_dir, file),
                os.path.join(out_dir, file.rsplit('.',1)[0]+".txt")
            )
            detected += int(ok)
            not_detected += int(not ok)

print("\n📈 FINAL OCR SUMMARY")
print(f"   Detected     : {detected}")
print(f"   Not detected : {not_detected}")
print("✅ ADAPTIVE OCR PLUS COMPLETED")

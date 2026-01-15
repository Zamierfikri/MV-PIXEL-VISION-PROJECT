import os
import tkinter as tk
from PIL import Image, ImageTk

# ==========================================================
# CONFIGURATION
# ==========================================================
BASE_OCR_FOLDERS = [
    "output_folder_1_easyOCR",
    "output_folder_2_easyOCR"
]

ADAPTIVE_OCR_FOLDERS = [
    "output_folder_1_adaptiveOCR",
    "output_folder_2_adaptiveOCR"
]

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 700
IMAGE_MAX_SIZE = (650, 500)

# ==========================================================
# IMAGE VIEWER CLASS
# ==========================================================
class OCRComparisonViewer:

    def __init__(self, root):
        self.root = root
        self.root.title("Base OCR vs Adaptive OCR Comparison")

        self.base_images = self.collect_images(BASE_OCR_FOLDERS)
        self.adaptive_images = self.collect_images(ADAPTIVE_OCR_FOLDERS)

        self.total = min(len(self.base_images), len(self.adaptive_images))
        self.index = 0

        # ---------------- IMAGE FRAME ----------------
        image_frame = tk.Frame(root)
        image_frame.pack(expand=True, pady=10)

        # BASE OCR IMAGE (LEFT)
        self.base_label = tk.Label(image_frame)
        self.base_label.pack(side=tk.LEFT, padx=10)

        # ADAPTIVE OCR IMAGE (RIGHT)
        self.adaptive_label = tk.Label(image_frame)
        self.adaptive_label.pack(side=tk.LEFT, padx=10)

        # ---------------- INFO LABEL ----------------
        self.info_label = tk.Label(root, font=("Arial", 14))
        self.info_label.pack(pady=5)

        # ---------------- BUTTONS ----------------
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(
            btn_frame, text="⬅ Previous", width=18, height=2,
            command=self.prev_image
        ).pack(side=tk.LEFT, padx=20)

        tk.Button(
            btn_frame, text="Next ➡", width=18, height=2,
            command=self.next_image
        ).pack(side=tk.LEFT, padx=20)

        # ---------------- KEYBOARD ----------------
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())

        # ---------------- LOAD FIRST ----------------
        if self.total > 0:
            self.show_images()
        else:
            self.info_label.config(text="No matching images found")

    # ------------------------------------------------------
    def collect_images(self, folders):
        images = []
        for folder in folders:
            if os.path.exists(folder):
                for file in sorted(os.listdir(folder)):
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        images.append(os.path.join(folder, file))
        return images

    # ------------------------------------------------------
    def show_images(self):
        base_path = self.base_images[self.index]
        adaptive_path = self.adaptive_images[self.index]

        base_img = Image.open(base_path)
        adaptive_img = Image.open(adaptive_path)

        base_img.thumbnail(IMAGE_MAX_SIZE)
        adaptive_img.thumbnail(IMAGE_MAX_SIZE)

        self.base_tk = ImageTk.PhotoImage(base_img)
        self.adaptive_tk = ImageTk.PhotoImage(adaptive_img)

        self.base_label.config(image=self.base_tk)
        self.base_label.image = self.base_tk

        self.adaptive_label.config(image=self.adaptive_tk)
        self.adaptive_label.image = self.adaptive_tk

        self.info_label.config(
            text=f"Image {self.index + 1} of {self.total}    |    "
                 f"Left: Base EasyOCR    |    Right: Adaptive EasyOCR"
        )

        self.root.title(
            f"OCR Comparison - {os.path.basename(base_path)}"
        )

    # ------------------------------------------------------
    def next_image(self):
        if self.index < self.total - 1:
            self.index += 1
            self.show_images()

    # ------------------------------------------------------
    def prev_image(self):
        if self.index > 0:
            self.index -= 1
            self.show_images()

# ==========================================================
# RUN UI
# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    app = OCRComparisonViewer(root)
    root.mainloop()

🚗 Adaptive License Plate Recognition (YOLO + EasyOCR)

A high-accuracy Malaysian license plate recognition system using YOLO object detection and an adaptive multi-scale OCR pipeline with confidence filtering, image enhancement, and validation rules.

This project compares a baseline OCR approach with an advanced adaptive OCR strategy designed to handle blurred, low-resolution, angled, and low-contrast images.

**System Overview**
Input Image
     ↓
YOLO License Plate Detection
     ↓
IoU-based NMS Filtering
     ↓
Plate Cropping & Resizing
     ↓
Multi-Scale OCR (EasyOCR)
     ↓
Confidence Voting & Validation
     ↓
Final Plate Output + Visualization

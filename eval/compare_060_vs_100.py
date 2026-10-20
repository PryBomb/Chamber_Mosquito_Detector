import os
import sys
import cv2
import numpy as np

sys.path.insert(0, r"c:\Users\LAPTOP\OneDrive\ドキュメント\agent skill\Chamber_Mosquito_Detector")
import config
from eval.experiment_crop_tuning import (
    get_testing_images, split_data, detect_and_cache_boxes, evaluate_setting,
    AEGYPTI_DIR, ALBOPICTUS_DIR
)
from chamber_detector import MosquitoClassifier, ChamberDetector

print("=" * 80)
print("  EVALUATING PAD 0.60 SQUARE vs PAD 1.00 SQUARE ON FINAL HALF (N = 360)")
print("=" * 80)

aeg_items = get_testing_images(AEGYPTI_DIR, "Aedes aegypti")
albo_items = get_testing_images(ALBOPICTUS_DIR, "Aedes albopictus")

_, aeg_final = split_data(aeg_items, seed=42)
_, albo_final = split_data(albo_items, seed=42)
final_items = aeg_final + albo_final

print(f"Final Half Items: {len(final_items)} (180 Aegypti, 180 Albopictus)")

clf = MosquitoClassifier()
detector = ChamberDetector()
detector.worker_running = False

print("\nRunning YOLO on Final Half...")
final_cached, final_missed = detect_and_cache_boxes(detector, final_items)
print(f"Final Half Detected: {sum(1 for d in final_cached if d['detected'])} / {len(final_cached)}")

# Evaluate pad 0.60 square
res_060 = evaluate_setting(final_cached, clf, padding=0.60, is_square=True)

# Evaluate pad 1.00 square
res_100 = evaluate_setting(final_cached, clf, padding=1.00, is_square=True)

# Also get baseline pad 0.20 rect
res_baseline = evaluate_setting(final_cached, clf, padding=0.20, is_square=False)

print("\n" + "=" * 80)
print("  FINAL HALF COMPARISON RESULTS (N = 360 images)")
print("=" * 80)
print(f"{'Metric':<30} | {'Baseline (0.20 rect)':<20} | {'Pad 0.60 Square':<18} | {'Pad 1.00 Square':<18}")
print("-" * 92)
print(f"{'Overall Acc (All 360)':<30} | {res_baseline['overall_acc_all']:>18.2f}% | {res_060['overall_acc_all']:>16.2f}% | {res_100['overall_acc_all']:>16.2f}%")
print(f"{'Overall Acc (Detected)':<30} | {res_baseline['overall_acc_det']:>18.2f}% | {res_060['overall_acc_det']:>16.2f}% | {res_100['overall_acc_det']:>16.2f}%")
print(f"{'Aedes aegypti Accuracy':<30} | {res_baseline['aeg_acc_det']:>18.2f}% | {res_060['aeg_acc_det']:>16.2f}% | {res_100['aeg_acc_det']:>16.2f}%")
print(f"{'Aedes albopictus Accuracy':<30} | {res_baseline['albo_acc_det']:>18.2f}% | {res_060['albo_acc_det']:>16.2f}% | {res_100['albo_acc_det']:>16.2f}%")
print(f"{'Species Balance (|Aeg - Albo|)':<30} | {abs(res_baseline['aeg_acc_det'] - res_baseline['albo_acc_det']):>18.2f}% | {abs(res_060['aeg_acc_det'] - res_060['albo_acc_det']):>16.2f}% | {abs(res_100['aeg_acc_det'] - res_100['albo_acc_det']):>16.2f}%")

cm_060 = res_060['cm']
cm_100 = res_100['cm']
print("\nConfusion Matrix - Pad 0.60 Square:")
print(f"  True Aegypti   : [Pred Aeg: {cm_060['aeg_aeg']:>3}, Pred Albo: {cm_060['aeg_albo']:>3}]")
print(f"  True Albopictus: [Pred Aeg: {cm_060['albo_aeg']:>3}, Pred Albo: {cm_060['albo_albo']:>3}]")

print("\nConfusion Matrix - Pad 1.00 Square:")
print(f"  True Aegypti   : [Pred Aeg: {cm_100['aeg_aeg']:>3}, Pred Albo: {cm_100['aeg_albo']:>3}]")
print(f"  True Albopictus: [Pred Aeg: {cm_100['albo_aeg']:>3}, Pred Albo: {cm_100['albo_albo']:>3}]")

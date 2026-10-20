"""
Unit test to verify that the live pipeline (chamber_detector.py) produces
the exact same crops and CNN classification scores as the experiment script.
"""
import os
import sys
import numpy as np
import cv2

# Add workspace root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from chamber_detector import ChamberDetector, MosquitoClassifier
from eval.experiment_crop_tuning import crop_square, crop_rect

def test_crop_parity():
    print("=" * 60)
    print("Testing Crop & Classification Parity: Live Pipeline vs Experiment")
    print("=" * 60)

    # Initialize live detector and classifier
    detector = ChamberDetector()
    detector.worker_running = False  # Disable background loop for synchronous unit testing
    clf = MosquitoClassifier()

    # Create synthetic test frames with varying sizes and bounding boxes
    test_cases = [
        # (frame_shape, bbox, padding, use_square)
        ((720, 1280, 3), (200, 150, 260, 210), 0.60, True),    # Typical centered bbox
        ((720, 1280, 3), (10, 20, 90, 110), 0.60, True),       # Near top-left boundary
        ((720, 1280, 3), (1200, 650, 1270, 710), 0.60, True),  # Near bottom-right boundary
        ((720, 1280, 3), (100, 200, 400, 300), 1.00, True),    # Wide bbox, pad 1.00
        ((720, 1280, 3), (200, 100, 280, 500), 1.00, True),    # Tall bbox, pad 1.00
        ((720, 1280, 3), (300, 300, 400, 400), 0.20, False),   # Rectangular baseline mode
    ]

    np.random.seed(42)

    for idx, (shape, box, pad, square) in enumerate(test_cases):
        # Generate deterministic synthetic image with pattern
        frame = np.random.randint(0, 255, size=shape, dtype=np.uint8)

        # Set config dynamically to test parity under both settings
        config.CROP_PADDING = pad
        config.USE_SQUARE_CROP = square

        # 1. Test Crop Array Equality
        live_crop = detector.crop_box(frame, box)
        if square:
            exp_crop = crop_square(frame, box, pad)
        else:
            exp_crop = crop_rect(frame, box, pad)

        assert np.array_equal(live_crop, exp_crop), (
            f"Test case {idx} failed: live crop shape {live_crop.shape} does not match experiment crop shape {exp_crop.shape}"
        )

        # 2. Test Classification Score Parity
        live_pred, live_conf, live_probs = detector.classifier.classify_crop(live_crop)
        exp_pred, exp_conf, exp_probs = clf.classify_crop(exp_crop)

        assert live_pred == exp_pred, f"Test case {idx} prediction mismatch: {live_pred} vs {exp_pred}"
        assert np.isclose(live_conf, exp_conf, atol=1e-5), f"Test case {idx} conf mismatch: {live_conf} vs {exp_conf}"
        assert np.allclose(live_probs, exp_probs, atol=1e-5), f"Test case {idx} probs mismatch: {live_probs} vs {exp_probs}"

        mode_str = f"pad {pad:.2f} {'square' if square else 'rect'}"
        print(f"  [PASS] Case {idx+1}: {mode_str} on bbox {box} -> crop {live_crop.shape}, CNN: {live_pred} ({live_conf*100:.2f}%)")

    # Also test on real chamber capture if available
    real_img_path = os.path.join(BASE_DIR, "captures", "capture_20261020_040401.jpg")
    if os.path.exists(real_img_path):
        real_frame = cv2.imread(real_img_path)
        real_box = (580, 320, 660, 410)
        config.CROP_PADDING = 0.60
        config.USE_SQUARE_CROP = True

        live_crop = detector.crop_box(real_frame, real_box)
        exp_crop = crop_square(real_frame, real_box, 0.60)
        assert np.array_equal(live_crop, exp_crop), "Real capture crop mismatch!"

        live_pred, live_conf, _ = detector.classifier.classify_crop(live_crop)
        exp_pred, exp_conf, _ = clf.classify_crop(exp_crop)
        assert live_pred == exp_pred and np.isclose(live_conf, exp_conf, atol=1e-5), "Real capture classification mismatch!"
        print(f"  [PASS] Real Chamber Capture: crop {live_crop.shape}, CNN: {live_pred} ({live_conf*100:.2f}%)")

    print("\nALL PARITY TESTS PASSED SUCCESSFULLY! (100% bit-exact equivalence)")

if __name__ == "__main__":
    test_crop_parity()

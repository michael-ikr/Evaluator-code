# test_parabola_display.py

import cv2
import numpy as np
from new_classification_fixed import Classification

def test_parabola_on_video(input_path="Supination.mp4", max_frames=50):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    cln = Classification()

    cln.string_points = [
        (300, 200),  # top-left
        (600, 210),  # top-right
        (600, 400),  # bottom-right
        (300, 390)   # bottom-left
    ]
    cln.bow_points = [
        (320, 220), (580, 230), (580, 380), (320, 370)
    ]

    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        success, frame = cap.read()
        if not success:
            break

        annotated = cln.display_classification(0, 0, frame)

        # Show the frame with overlay
        cv2.imshow("Parabola Overlay Test", annotated)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_parabola_on_video()

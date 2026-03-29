import os
import sys
import cv2
import numpy as np

print("test başladı")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from backend.modules.landmark.landmark import process_landmark_pipeline

print("import tamam")


def main():
    input_path = os.path.join(PROJECT_ROOT, "static", "uploads", "original_20260329_160138.jpg")
    output_path = os.path.join(PROJECT_ROOT, "static", "results", "landmarks_overlay.jpg")

    print("input path:", input_path)

    image = cv2.imdecode(np.fromfile(input_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Image could not be loaded: {input_path}")
        return

    print("görüntü yüklendi")

    result = process_landmark_pipeline(image, output_path=output_path)

    print("----- LANDMARK TEST RESULT -----")
    print("Success:", result["success"])
    print("Landmark count:", result["num_landmarks"])
    print("Validation:", result["validation"])
    print("Output path:", result["output_path"])

    if result["landmarks"]:
        print("First 10 landmarks:")
        for point in result["landmarks"][:10]:
            print(point)


if __name__ == "__main__":
    main()
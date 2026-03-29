import os
from typing import List, Tuple, Dict, Any

import cv2
import mediapipe as mp
import numpy as np

LandmarkList = List[Tuple[int, int]]


def detect_landmarks(image: np.ndarray) -> LandmarkList:
    if image is None:
        raise ValueError("Input image is None.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return []

    h, w = image.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    landmarks = []
    for lm in face_landmarks.landmark:
        x = int(lm.x * w)
        y = int(lm.y * h)

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))

        landmarks.append((x, y))

    return landmarks


def draw_landmarks(
    image: np.ndarray,
    landmarks: LandmarkList,
    radius: int = 1
) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")

    output = image.copy()

    for (x, y) in landmarks:
        cv2.circle(output, (x, y), radius, (0, 255, 0), -1)

    return output


def validate_landmarks(
    landmarks: LandmarkList,
    image_shape: Tuple[int, int, int]
) -> Dict[str, Any]:
    if not landmarks:
        return {
            "is_valid": False,
            "reason": "No landmarks detected.",
            "count": 0,
            "inside_ratio": 0.0
        }

    h, w = image_shape[:2]

    inside_count = 0
    for (x, y) in landmarks:
        if 0 <= x < w and 0 <= y < h:
            inside_count += 1

    inside_ratio = inside_count / len(landmarks)

    if len(landmarks) < 100:
        return {
            "is_valid": False,
            "reason": "Too few landmarks detected.",
            "count": len(landmarks),
            "inside_ratio": inside_ratio
        }

    if inside_ratio < 0.95:
        return {
            "is_valid": False,
            "reason": "Many landmarks are outside image bounds.",
            "count": len(landmarks),
            "inside_ratio": inside_ratio
        }

    return {
        "is_valid": True,
        "reason": "Landmarks successfully detected.",
        "count": len(landmarks),
        "inside_ratio": inside_ratio
    }


def save_image(image: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    success, encoded_image = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Image encoding failed.")

    encoded_image.tofile(output_path)


def process_landmark_pipeline(
    image: np.ndarray,
    output_path: str = None
) -> Dict[str, Any]:
    landmarks = detect_landmarks(image)
    validation = validate_landmarks(landmarks, image.shape)

    landmark_image = draw_landmarks(image, landmarks) if landmarks else image.copy()

    if output_path:
        save_image(landmark_image, output_path)

    return {
        "success": validation["is_valid"],
        "num_landmarks": validation["count"],
        "landmarks": landmarks,
        "validation": validation,
        "output_path": output_path,
        "image_with_landmarks": landmark_image
    }
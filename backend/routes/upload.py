import os
import shutil
import cv2
import numpy as np

from flask import Blueprint, request, current_app
from werkzeug.utils import secure_filename

from backend.modules.utils.helpers import (
    allowed_file,
    error_response,
    success_response,
    timestamped_filename,
)
import shutil
#from backend.modules.landmark.landmark import process_landmark_pipeline
#from backend.modules.warping.warping import apply_expression

upload_bp = Blueprint("upload", __name__)

TRANSFORM_MAP = {
    "smile": "smile",
    "eyebrow": "eyebrow_raise",
    "lip_widen": "lip_widen",
    "slim_face": "face_slimming",
}


def apply_aging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))
    aged = image.copy()

    gray = cv2.cvtColor(aged, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    aged = cv2.addWeighted(
        aged,
        1 - 0.35 * intensity,
        gray_bgr,
        0.35 * intensity,
        0
    )

    return aged


def apply_deaging_effect(image, intensity=0.5):
    intensity = float(max(0.0, min(1.0, intensity)))

    smooth = cv2.bilateralFilter(image, 15, 90, 90)
    deaged = cv2.addWeighted(image, 1 - intensity, smooth, intensity, 0)

    bright = np.full_like(deaged, (12, 12, 12))
    deaged = cv2.addWeighted(deaged, 1.0, bright, 0.25 * intensity, 0)

    return deaged


@upload_bp.route("/", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return error_response("No image file found in request.", 400)

    file = request.files["image"]

    if file.filename == "":
        return error_response("No file selected.", 400)

    if not allowed_file(file.filename):
        return error_response("Unsupported file format. Please upload JPG or PNG.", 400)

    transform_type = request.form.get("transform_type", "smile")
    intensity = request.form.get("intensity", 0.5)

    try:
        intensity = float(intensity)
    except (TypeError, ValueError):
        return error_response("intensity must be a number.", 400)

    intensity = max(0.0, min(1.0, intensity))

    valid_extra_transforms = {"aging", "deaging", "landmarks"}

    if transform_type not in TRANSFORM_MAP and transform_type not in valid_extra_transforms:
        return error_response("Invalid transform_type.", 400)

    filename = secure_filename(file.filename)
    filename = timestamped_filename(filename)

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)
    file.save(file_path)

    original_path = os.path.join(upload_folder, "original.jpg")
    transformed_path = os.path.join(upload_folder, "transformed.jpg")

    shutil.copy(file_path, original_path)

    image = cv2.imread(file_path)

    if image is None:
        return error_response("Image could not be read.", 400)

    try:
        if transform_type == "aging":
            output_image = apply_aging_effect(image, intensity)

        elif transform_type == "deaging":
            output_image = apply_deaging_effect(image, intensity)

        else:
            # Mediapipe/landmark pipeline disabled for now.
            # Temporary fallback: use uploaded image as transformed image.
            output_image = image.copy()

        saved = cv2.imwrite(transformed_path, output_image)

        if not saved:
            return error_response("Transformed image could not be saved.", 500)

    except Exception as e:
        return error_response(f"Transform failed: {str(e)}", 500)

    return success_response(
        "Image uploaded and transformed successfully.",
        data={
            "filename": filename,
            "file_path": file_path,
            "original_path": original_path,
            "transformed_path": transformed_path,
            "transform_type": transform_type,
            "intensity": intensity
        },
        status_code=201
    )
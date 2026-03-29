from flask import Blueprint, request
from backend.modules.utils.helpers import error_response, success_response

transform_bp = Blueprint("transform", __name__)

VALID_TRANSFORMS = {
    "smile",
    "eyebrow",
    "lip_widen",
    "slim_face",
    "aging",
    "deaging"
}


@transform_bp.route("/", methods=["POST"])
def transform_image():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path")
    transform_type = data.get("transform_type")
    intensity = data.get("intensity", 1.0)

    if not image_path:
        return error_response("image_path is required.", 400)

    if transform_type not in VALID_TRANSFORMS:
        return error_response("Invalid transform_type.", 400)

    return success_response(
        "Transform endpoint is ready. Warping or aging module will be integrated here.",
        data={
            "image_path": image_path,
            "transform_type": transform_type,
            "intensity": intensity
        }
    )
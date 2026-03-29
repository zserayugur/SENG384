from flask import Blueprint, request
from backend.modules.utils.helpers import error_response, success_response

process_bp = Blueprint("process", __name__)


@process_bp.route("/", methods=["POST"])
def process_image():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    image_path = data.get("image_path")

    if not image_path:
        return error_response("image_path is required.", 400)

    return success_response(
        "Process endpoint is ready. Preprocessing module will be integrated here.",
        data={
            "image_path": image_path,
            "status": "waiting_for_preprocessing_module"
        }
    )
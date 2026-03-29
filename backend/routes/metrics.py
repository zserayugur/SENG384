# =====================================================
# METRICS ROUTE
# MSE, PSNR, SSIM gibi metrikler için endpoint
# =====================================================

from flask import Blueprint, request
from backend.modules.utils.helpers import error_response, success_response

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/", methods=["POST"])
def calculate_metrics():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    original_path = data.get("original_path")
    transformed_path = data.get("transformed_path")

    if not original_path or not transformed_path:
        return error_response("original_path and transformed_path are required.", 400)

    return success_response(
        "Metrics endpoint is ready. Evaluation module will be integrated here.",
        data={
            "original_path": original_path,
            "transformed_path": transformed_path
        }
    )
    
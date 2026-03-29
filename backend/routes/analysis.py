# =====================================================
# ANALYSIS ROUTE
# Endpoint for FFT and analysis operations
# Sinem's FFT module is integrated here
# =====================================================

from flask import Blueprint, request
from backend.modules.utils.helpers import error_response, success_response

# Bu sinemin yoluna göre düzelcek?
from analysis.fft_metrics import analyze_images
analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/", methods=["POST"])
def analyze_image():
    try:
        data = request.get_json()

        if not data:
            return error_response("JSON body is required.", 400)

        original_path = data.get("original_path")
        transformed_path = data.get("transformed_path")

        if not original_path or not transformed_path:
            return error_response(
                "original_path and transformed_path are required.",
                400
            )

    
        results = analyze_images(original_path, transformed_path)

        return success_response(
            "Analysis completed successfully.",
            data=results
        )

    except Exception as e:
        return error_response(str(e), 500)
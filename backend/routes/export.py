from flask import Blueprint, request
from backend.modules.utils.helpers import error_response, success_response

export_bp = Blueprint("export", __name__)


@export_bp.route("/", methods=["POST"])
def export_results():
    data = request.get_json()

    if not data:
        return error_response("JSON body is required.", 400)

    export_type = data.get("export_type", "csv")

    if export_type not in ["csv", "pdf"]:
        return error_response("export_type must be 'csv' or 'pdf'.", 400)

    return success_response(
        "Export endpoint is ready. Export logic will be integrated here.",
        data={
            "export_type": export_type
        }
    )
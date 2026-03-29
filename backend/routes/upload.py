"""
checks if there is an image in the incoming file
checks if the extension is suitable
saves the file to static/uploads
returns the saved path as JSON.
"""


import os
from flask import Blueprint, request, current_app
from werkzeug.utils import secure_filename

from backend.modules.utils.helpers import (
    allowed_file,
    error_response,
    success_response,
    timestamped_filename,
)

upload_bp = Blueprint("upload", __name__)


@upload_bp.route("/", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return error_response("No image file found in request.", 400)

    file = request.files["image"]

    if file.filename == "":
        return error_response("No file selected.", 400)

    if not allowed_file(file.filename):
        return error_response("Unsupported file format. Please upload JPG or PNG.", 400)

    filename = secure_filename(file.filename)
    filename = timestamped_filename(filename)

    upload_folder = current_app.config["UPLOAD_FOLDER"]
    file_path = os.path.join(upload_folder, filename)

    file.save(file_path)

    return success_response(
        "Image uploaded successfully.",
        data={
            "filename": filename,
            "file_path": file_path
        },
        status_code=201
    )
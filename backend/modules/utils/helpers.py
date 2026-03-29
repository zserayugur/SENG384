#File extension check
#Folder creation
#Generating file names with dates
#Returning standard successful/failed JSON responses


import os
from datetime import datetime
from flask import jsonify

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamped_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}{ext}"


def success_response(message: str, data=None, status_code: int = 200):
    return jsonify({
        "success": True,
        "message": message,
        "data": data or {}
    }), status_code


def error_response(message: str, status_code: int = 400):
    return jsonify({
        "success": False,
        "message": message
    }), status_code
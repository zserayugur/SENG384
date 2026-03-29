"""
Launching Flask
Adding route files to the system
Creating upload and results folders
Returning a test message on the main page
"""

from flask import Flask
from flask_cors import CORS

from backend.routes.upload import upload_bp
from backend.routes.process import process_bp
from backend.routes.transform import transform_bp
from backend.routes.analysis import analysis_bp
from backend.routes.metrics import metrics_bp
from backend.routes.export import export_bp
from backend.modules.utils.helpers import ensure_dir


def create_app():
    app = Flask(__name__)
    CORS(app)

    app.config["UPLOAD_FOLDER"] = "static/uploads"
    app.config["RESULT_FOLDER"] = "static/results"
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

    ensure_dir(app.config["UPLOAD_FOLDER"])
    ensure_dir(app.config["RESULT_FOLDER"])

    app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(process_bp, url_prefix="/process")
    app.register_blueprint(transform_bp, url_prefix="/transform")
    app.register_blueprint(analysis_bp, url_prefix="/analyze")
    app.register_blueprint(metrics_bp, url_prefix="/metrics")
    app.register_blueprint(export_bp, url_prefix="/export")

    @app.route("/")
    def home():
        return {
            "message": "Facial Image Warping API is running."
        }

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
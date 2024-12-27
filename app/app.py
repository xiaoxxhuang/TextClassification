import os

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

from src.inference import load_model, predict

app = Flask(__name__)

model_dir = load_model(os.environ["SM_MODEL_DIR"])
print("Files in /opt/ml/model:", os.listdir(os.environ["SM_MODEL_DIR"]))

app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)

@app.route("/ping", methods=["GET"])
def health_check():
    return "SUCCESS"

@app.route("/invocations", methods=["POST"])
def invocations():
    body = request.json
    return predict(body, model_dir)
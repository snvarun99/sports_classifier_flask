import io
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from torchvision.models import resnet18
import requests

app = Flask(__name__)

# ---- Load model ----
class SimpleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = resnet18(weights=None)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Load checkpoint
# Resolve checkpoint path relative to this file so it works regardless of CWD
checkpoint_path = os.path.join(os.path.dirname(__file__), "sports_classifier_resnet18_05.pth")
state_dict = torch.load(checkpoint_path, map_location="cpu")

# Infer number of classes from the checkpoint's final layer
inferred_num_classes = state_dict["fc.weight"].shape[0]
model = SimpleModel(inferred_num_classes)

# Load weights into the inner ResNet to match key names (no 'base.' prefix in checkpoint)
model.base.load_state_dict(state_dict)
model.eval()

# ---- Image transforms ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Helper: load image from file or URL ----
def load_image(image_bytes):
    return transform(Image.open(io.BytesIO(image_bytes)).convert("RGB")).unsqueeze(0)

# ---- Helper: class name mapping (optional) ----
_BASE_DIR = os.path.dirname(__file__)
_JSON_CLASSES = os.path.join(_BASE_DIR, "class_names.json")
_TXT_CLASSES = os.path.join(_BASE_DIR, "classes.txt")

def _load_class_names():
    # Try JSON first
    if os.path.exists(_JSON_CLASSES):
        try:
            with open(_JSON_CLASSES, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            pass
    # Fallback to plain text (one class per line)
    if os.path.exists(_TXT_CLASSES):
        try:
            with open(_TXT_CLASSES, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            return lines
        except Exception:
            pass
    return None

CLASS_NAMES = _load_class_names()

def _index_to_label(index):
    if CLASS_NAMES is None:
        return None
    # If dict, try integer or string keys
    if isinstance(CLASS_NAMES, dict):
        return CLASS_NAMES.get(index) or CLASS_NAMES.get(str(index))
    # If list, index directly when in range
    if isinstance(CLASS_NAMES, list) and 0 <= index < len(CLASS_NAMES):
        return CLASS_NAMES[index]
    return None

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        elif "url" in request.json:
            response = requests.get(request.json["url"])
            image_bytes = response.content
        else:
            return jsonify({"error": "No image provided"}), 400

        img_tensor = load_image(image_bytes)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        idx = int(predicted.item())
        label = _index_to_label(idx)
        response = {"prediction": idx}
        if label is not None:
            response["label"] = label
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

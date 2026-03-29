import os
import numpy as np
import gdown
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# =============================
# MODEL DOWNLOAD
# =============================
MODEL_PATH = "pneumonia_mobilenet_best.h5"
FILE_ID = "1mgTHrOYig_syGE6kOtTc5PMC2x-MHqjZ"  # ✅ Fixed: ID only, no extra text

if not os.path.exists(MODEL_PATH):
    print("⏳ Downloading model from Google Drive...")
    gdown.download(
        id=FILE_ID,        # ✅ Fixed: pass id directly
        output=MODEL_PATH,
        quiet=False,
        fuzzy=True         # ✅ Fixed: handles permission issues
    )

# =============================
# FLASK SETUP
# =============================
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =============================
# MODEL LOADING (only once ✅)
# =============================
print("⏳ Loading MediScan AI Model...")
model = load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")

# CLASS LABELS
CLASS_NAMES = ['BACTERIAL', 'NORMAL', 'VIRAL']

# =============================
# PREDICTION LOGIC
# =============================
def predict_logic(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)[0]
    predicted_index = np.argmax(preds)

    result = CLASS_NAMES[predicted_index]
    confidence = round(float(preds[predicted_index]) * 100, 2)

    return result, confidence

# =============================
# ROUTES
# =============================

@app.route("/")
def home():
    return render_template("first.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():
    if request.method == "GET":
        return render_template("index.html", prediction=None)

    file = request.files.get("file")
    if not file or file.filename == "":
        return render_template("index.html",
            prediction="UNCERTAIN",
            confidence=0
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    prediction, confidence = predict_logic(filepath)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=filepath
    )

@app.route("/charts")
def charts():
    return render_template("charts.html")

@app.route("/timeline")
def timeline():
    return render_template("timeline.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Fixed: Render sets PORT automatically
    app.run(debug=False, host="0.0.0.0", port=port)
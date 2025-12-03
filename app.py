import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# =============================
# FLASK SETUP
# =============================
app = Flask(__name__)

# Upload Folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =============================
# MODEL LOADING
# =============================
print("⏳ Loading MediScan AI Model...")
model = load_model('pneumonia_mobilenet_best.h5')
print("✅ Model Loaded Successfully!")

# CLASS LABELS
CLASS_NAMES = ['BACTERIAL', 'NORMAL', 'VIRAL']

# =============================
# PREDICTION LOGIC
# =============================
def predict_logic(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)[0]
    predicted_index = np.argmax(preds)

    result = CLASS_NAMES[predicted_index]
    confidence = round(float(preds[predicted_index])*100,2)

    return result, confidence

# =============================
# ROUTES
# =============================

# Home Page
@app.route("/")
def home():
    return render_template("first.html")

# Xray Detection Page + Upload
@app.route("/detect", methods=["GET","POST"])
def detect():

    # Page first load
    if request.method=="GET":
        return render_template("index.html", prediction=None)

    # When image uploaded
    file = request.files.get("file")
    if not file or file.filename=="":
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


# INFORMATION MODULE PAGES
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
    app.run(debug=True, port=5000)

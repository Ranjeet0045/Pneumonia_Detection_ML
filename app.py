import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 1. Configure Upload Folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 2. Load the Mega-Model
print("⏳ Loading the Mega-Model... Please wait...")
model = load_model('pneumonia_mega_model.h5')
print("✅ Model Loaded Successfully!")

def predict_logic(img_path):
    # Resize image to 224x224 (Must match Training Step!)
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Convert to Array & Normalize (1./255)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 

    # Predict
    # The output is a single probability between 0 and 1
    prediction_score = model.predict(x)[0][0]
    
    # Threshold logic (0.5 cut-off)
    if prediction_score > 0.5:
        result = "PNEUMONIA"
        confidence = round(prediction_score * 100, 2)
    else:
        result = "NORMAL"
        confidence = round((1 - prediction_score) * 100, 2)
        
    return result, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get Prediction
            res, conf = predict_logic(filepath)
            
            return render_template('index.html', 
                                   prediction=res, 
                                   confidence=conf, 
                                   img_path=filepath)
            
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
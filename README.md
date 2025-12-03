# MediScan AI – Pneumonia X-Ray Analyzer

MediScan AI is a web app that uses a deep-learning model to analyze chest X-ray images and predict:

- **BACTERIAL pneumonia**
- **VIRAL pneumonia**
- **NORMAL (healthy lungs)**

It’s built as a clean, educational mini-platform with:
- An **information-rich landing page** (risk, prevention, charts, timelines, FAQ, map)
- An **AI detector page** where users can upload an X-ray and see the predicted class + confidence
- Clear, focused **next-step tips** for each result


---

## ✨ Features

- 📸 **Chest X-ray upload** (click or drag & drop)
- 🤖 **Deep-learning inference** using a trained MobileNet model (`pneumonia_mobilenet_best.h5`)
- 🧪 **3-class prediction**: `BACTERIAL`, `VIRAL`, `NORMAL`
- 📊 **Analytics landing page**:
  - Global health statistics & animated counters
  - Pneumonia health charts (age risk, trends)
  - Recovery & treatment timeline
  - FAQ section
- 💡 **Actionable tips** per result:
  - Separate guidance for **bacterial**, **viral**, **healthy**, and **inconclusive** cases
- 🌐 **Modern UI**:
  - Animated hero section
  - Particle background
  - Smooth layout transitions
  - Responsive design for desktop & mobile

---

## 🧱 Tech Stack

- **Backend:** Python, Flask
- **Deep Learning:** TensorFlow / Keras (MobileNet-based classifier)
- **Frontend:** HTML5, CSS3, vanilla JavaScript
- **Templates:** Jinja2 (`templates/` folder)
- **Charts & visuals:** Chart.js (in separate template modules)
- **Environment:** Git, virtualenv (recommended)

---

## 📁 Project Structure

```bash
.
├── app.py                    # Flask app + routes + prediction logic
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored files (uploads, model, caches, etc.)
├── templates/
│   ├── first.html            # Landing / info page
│   ├── index.html            # Detector page (upload + results UI)
│   ├── charts.html           # Analytics charts section
│   ├── timeline.html         # Recovery & treatment timeline
│   └── faq.html              # FAQs about pneumonia & AI
└── static/
    └── uploads/              # Uploaded X-ray images (not committed to Git)

🚀 Getting Started (Run Locally)

1️⃣ Clone the repository
git clone https://github.com/<RANJEET0045>/<Pneumonia_Detection_ML>.git
cd <Pneumonia_Detection_ML>

2️⃣ (Recommended) Create a virtual environment
python -m venv env
env\Scripts\activate      # Windows
# source env/bin/activate # Mac / Linux

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add the model file

Download / copy your trained model file to the project root:
pneumonia_mobilenet_best.h5

Make sure the filename matches what app.py expects:
model = load_model('pneumonia_mobilenet_best.h5')

5️⃣ Run the Flask app: "python app.py"

By default it runs at:
http://127.0.0.1:5000/

🧭 App Flow
/ → Landing / Info Page (first.html)

Pneumonia overview, risk groups, symptoms, prevention
Animated stats and health insights
Embedded sections for:
Charts (/charts)
Recovery timeline (/timeline)
FAQ (/faq)
CTA button: “Launch AI Detector” → /detect

/detect → AI Detector Page (index.html)
Upload area for chest X-ray (click or drag & drop)
On upload:
Image is saved under static/uploads/
Model runs prediction via predict_logic(...)
Confidence score is calculated

Result view:
Shows predicted class (BACTERIAL / VIRAL / NORMAL / UNCERTAIN)
Shows confidence percentage
Shows X-ray preview
Displays tips tailored to the result:
Bacterial → antibiotics, monitoring, urgent care triggers (5 tips)
Viral → rest, fluids, isolation, symptom relief, no antibiotics (5 tips)
Normal → hygiene, no smoking, exercise (3 tips)
Uncertain → re-scan, better image, clinical review (3 tips)
Button: “Analyze New Scan” resets the page and allows another upload.

🧮 Model Details (High-Level)

Backbone: MobileNet-based CNN
Input: Chest X-ray images resized to 224 × 224
Preprocessing:
image.load_img(..., target_size=(224, 224))
x / 255.0 rescaling

Output: Softmax probabilities over 3 classes:
BACTERIAL
NORMAL
VIRAL

## 📂 Datasets Used

This project utilizes a custom hybrid dataset created by combining two major open-source repositories to ensure robust detection across different age groups (Pediatric & Adult) and pneumonia types.

---

### Chest X-Ray Images (Pneumonia) – Paul Timothy Mooney  
**Focus:** Pediatric patients (1–5 years old)

🔗 **Dataset URL:**  
👉 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

### COVID-19 Radiography Database – Tawsifur Rahman  
**Focus:** Adult patients & Viral Pneumonia cases

🔗 **Dataset URL:**  
👉 https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

---

### ✅ Why we combined them:

- Mooney dataset provides reliable **Bacterial pneumonia data** but lacks adult cases.
- Rahman dataset contains strong **Adult Normal & Viral pneumonia samples**.

🎯 **Result:**
A balanced **multi-class dataset**:

**Normal – Bacterial – Viral**

Covering both pediatric and adult chest X-ray images.


---

## 📸 Screenshots

### Landing Page
![Landing Page](https://raw.githubusercontent.com/Ranjeet0045/Pneumonia_Detection_ML/master/screenshots/landing.png)

---

### Upload X-Ray
![Upload Screen](https://raw.githubusercontent.com/Ranjeet0045/Pneumonia_Detection_ML/master/screenshots/upload.png)

---

### Bacterial Detection Result
![Bacterial Result](https://raw.githubusercontent.com/Ranjeet0045/Pneumonia_Detection_ML/master/screenshots/bacterial.png)

---

### Viral Detection Result
![Viral Result](https://raw.githubusercontent.com/Ranjeet0045/Pneumonia_Detection_ML/master/screenshots/viral.png)

---

### Normal Detection Result
![Normal Result](https://raw.githubusercontent.com/Ranjeet0045/Pneumonia_Detection_ML/master/screenshots/normal.png)

# DermVision AI - Complete Source Code

This document contains the complete codebase for both the frontend interface and backend servers of DermVision AI.

## Table of Contents

### Backend Development
- [app.py](#apppy) - *Flask application backend. Sets up the local database, hosts predictions/inference pipelines, and provides routes for analysis history.*
- [model.py](#modelpy) - *PyTorch deep learning network definition. Combines MobileNetV2 with a Squeeze-and-Excitation (SE) attention block for optimized skin classification.*
- [utils.py](#utilspy) - *Image processing utility pipeline. Implements Dull Razor (morphological hair removal) and CLAHE (contrast enhancement in LAB color space).*
- [requirements.txt](#requirementstxt) - *System requirements and Python package dependencies for the backend environment.*

### Frontend Development
- [templates/index.html](#templatesindexhtml) - *HTML Dashboard template. Implements structural grid systems, UI sections for image preview, visualizer, stats, and a sliding drawer for history.*
- [static/css/style.css](#staticcssstylecss) - *Custom stylesheet. Styles the glassmorphic aesthetics, glowing background layers, dynamic progress indicators, and custom dark mode components.*
- [static/js/app.js](#staticjsappjs) - *Frontend controller. Implements client interaction, drag-and-drop image processing simulations, database REST interaction, and rendering logic.*

---

## app.py

> **Category**: Backend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/app.py`  
> **Description**: Flask application backend. Sets up the local database, hosts predictions/inference pipelines, and provides routes for analysis history.

```python
import os
import base64
import numpy as np
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from PIL import Image
import sqlite3
from flask import Flask, request, jsonify, render_template

# Import our image preprocessing pipeline
from utils import preprocess_pipeline

# Try to import PyTorch and our model definition
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from model import get_model
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# SQLite Database setup for Scan History
DATABASE = 'scans.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            predicted_class TEXT NOT NULL,
            predicted_key TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT NOT NULL,
            original_image TEXT NOT NULL,
            enhanced_image TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Helper function to encode OpenCV image to base64
def encode_img_to_base64(img):
    if not CV2_AVAILABLE:
        return ""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# Skin Disease metadata and decision support recommendations
DISEASE_DATA = {
    "mel": {
        "name": "Melanoma",
        "type": "Malignant (High Risk)",
        "risk_level": "High",
        "description": "A serious form of skin cancer that begins in the cells (melanocytes) that control skin pigment. It is highly curable if detected early, but can spread to other parts of the body if left untreated.",
        "precautions": [
            "Do NOT scratch, pick, or irritate the lesion.",
            "Avoid all direct sun exposure and tanning beds.",
            "Schedule a professional skin biopsy immediately."
        ],
        "skincare": [
            "Keep the skin barrier hydrated with gentle, fragrance-free moisturizers.",
            "Always apply a broad-spectrum mineral sunscreen (SPF 50+) if exposed to light.",
            "Avoid chemical peels or abrasive scrubs on the affected area."
        ],
        "medical_advice": "CRITICAL: Seek immediate evaluation from a board-certified dermatologist. This condition requires a professional biopsy and surgical excision. Watch closely for the ABCDE rules of skin cancer."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "type": "Malignant / Locally Invasive (Medium-High Risk)",
        "risk_level": "Medium-High",
        "description": "The most common form of skin cancer, which begins in the basal cells of the epidermis. It grows very slowly and rarely spreads, but can cause localized tissue damage if neglected.",
        "precautions": [
            "Do not attempt to squeeze, pop, or scratch the lesion.",
            "Shield the area from direct sunlight.",
            "Schedule a consultation for dermatological evaluation."
        ],
        "skincare": [
            "Cleanse the skin with mild, soap-free cleansers.",
            "Maintain the skin barrier with ceramide-based moisturizers.",
            "Apply SPF 30+ sunscreen daily to prevent localized growth stimulation."
        ],
        "medical_advice": "IMPORTANT: Consult a dermatologist to discuss removal options, which include surgical excision, Mohs micrographic surgery, or cryosurgery. Basal Cell Carcinoma should be professionally treated to prevent deep tissue invasion."
    },
    "akiec": {
        "name": "Actinic Keratosis (Pre-Cancerous)",
        "type": "Pre-Malignant (Medium Risk)",
        "risk_level": "Medium",
        "description": "A rough, scaly patch on the skin that develops from years of sun exposure. It is considered pre-cancerous because it can progress to Squamous Cell Carcinoma (SCC) if left untreated.",
        "precautions": [
            "Strictly avoid sun exposure during peak hours.",
            "Do not peel off or pick at the dry, scaly skin patches.",
            "Check for other scaly lesions on sun-exposed areas (face, scalp, ears)."
        ],
        "skincare": [
            "Use thick, soothing emollients (like petrolatum or shea butter) to calm dryness.",
            "Incorporate topical antioxidants (e.g., Niacinamide) to support skin repair.",
            "Use broad-spectrum SPF 50+ sunscreen religiously before going outdoors."
        ],
        "medical_advice": "RECOMMENDED: Visit a dermatologist for examination and treatment. Options include cryotherapy (freezing), topical chemotherapy creams (like 5-fluorouracil), or photodynamic therapy to clear the pre-cancerous cells."
    },
    "nv": {
        "name": "Melanocytic Nevi (Common Mole)",
        "type": "Benign (Low Risk)",
        "risk_level": "Low",
        "description": "A common, non-cancerous skin growth caused by clusters of melanocytes. Moles are normal skin features and generally harmless, though they should be monitored for changes.",
        "precautions": [
            "Monitor the mole monthly for changes in shape, size, border, or color.",
            "Prevent sunburns, as they increase the risk of moles mutating.",
            "Avoid friction or rubbing from tight clothing or jewelry."
        ],
        "skincare": [
            "Apply broad-spectrum SPF 30+ daily to all exposed skin.",
            "Keep the skin healthy with general hydration and moisture.",
            "No special clinical skincare is needed unless the mole is physically irritated."
        ],
        "medical_advice": "ROUTINE: No immediate medical treatment is required. However, if the mole begins to bleed, itch, grow rapidly, or show irregular borders (asymmetry, color shifts), consult a dermatologist immediately."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesions",
        "type": "Benign (Low Risk)",
        "risk_level": "Low",
        "description": "Non-cancerous skin lesions, most commonly representing Seborrheic Keratosis. They typically have a waxy, 'stuck-on' appearance and are common in older adults.",
        "precautions": [
            "Avoid picking or scratching at the crust, which can lead to localized bleeding or infection.",
            "Prevent friction from clothes, which can make the lesion red and irritated."
        ],
        "skincare": [
            "Soothe the area with light, non-comedogenic moisturizers.",
            "Avoid using harsh chemical exfoliants (like glycolic or salicylic acid) directly on the growth.",
            "Wash the area gently with a soft cloth."
        ],
        "medical_advice": "ROUTINE: These growths are entirely harmless and do not turn into cancer. If they become irritated, catch on clothing, or are a cosmetic concern, a dermatologist can quickly remove them using cryotherapy or light curettage."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "type": "Benign (Low Risk)",
        "risk_level": "Low",
        "description": "Benign lesions involving abnormal blood vessels. This includes cherry angiomas, hemangiomas, and pyogenic granulomas, which appear red, purple, or blue.",
        "precautions": [
            "Avoid trauma or picking at the site, as vascular lesions bleed very easily.",
            "Avoid applying excessive heat or hot water to the area."
        ],
        "skincare": [
            "Use gentle cleansers and pat dry without rubbing.",
            "Use simple, mild lotions to keep the skin surrounding the lesion hydrated."
        ],
        "medical_advice": "ROUTINE: Generally harmless. If the lesion bleeds excessively, becomes painful, or grows rapidly (which can happen with pyogenic granulomas), seek clinical evaluation for treatment, which may include laser therapy or electrocautery."
    },
    "df": {
        "name": "Dermatofibroma",
        "type": "Benign (Low Risk)",
        "risk_level": "Low",
        "description": "A common, harmless, firm nodule that typically grows on the lower legs. They are often triggered by minor skin injuries like bug bites or shaving nicks.",
        "precautions": [
            "Avoid squeezing or trying to pop the nodule, as it is composed of fibrous scar tissue.",
            "Be careful when shaving around the area to prevent cutting it."
        ],
        "skincare": [
            "Apply moisturizers to keep the skin smooth.",
            "Use soothing creams if the nodule feels dry or slightly itchy."
        ],
        "medical_advice": "ROUTINE: Dermatofibromas are benign and do not require treatment. If they are painful, itchy, or cosmetically undesirable, they can be surgically removed by a dermatologist, though this leaves a small scar."
    }
}

# Mapping of labels to keys
LABEL_MAP = {
    0: "akiec",
    1: "bcc",
    2: "bkl",
    3: "df",
    4: "mel",
    5: "nv",
    6: "vasc"
}

# Try loading the trained PyTorch model
model = None
if PYTORCH_AVAILABLE:
    try:
        model = get_model(num_classes=7)
        if os.path.exists("skin_disease_model.pth"):
            model.load_state_dict(torch.load("skin_disease_model.pth", map_location=torch.device('cpu')))
            model.eval()
            print("Successfully loaded PyTorch model weights from 'skin_disease_model.pth'")
        else:
            print("Model weights not found. Running PyTorch in unitialized state (will yield random predictions until trained).")
    except Exception as e:
        print(f"Error initializing PyTorch model: {e}. Model will fall back to simulated classification.")
        model = None

# Transform for PyTorch inference
if PYTORCH_AVAILABLE:
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@app.route('/')
def index():
    return render_template('index.html', pytorch_status=PYTORCH_AVAILABLE)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    # Save uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    try:
        # 1. Run the image preprocessing pipeline (Dull Razor + CLAHE)
        processed_data = preprocess_pipeline(file_path)
        
        # Base64 encode intermediate steps for frontend rendering
        if processed_data.get('is_fallback', False):
            steps_base64 = {
                'original': processed_data['original'],
                'grayscale': processed_data['grayscale'],
                'blackhat': processed_data['blackhat'],
                'hair_mask': processed_data['hair_mask'],
                'hair_removed': processed_data['hair_removed'],
                'enhanced': processed_data['enhanced']
            }
        else:
            steps_base64 = {
                'original': encode_img_to_base64(processed_data['original']),
                'grayscale': encode_img_to_base64(processed_data['grayscale']),
                'blackhat': encode_img_to_base64(processed_data['blackhat']),
                'hair_mask': encode_img_to_base64(processed_data['hair_mask']),
                'hair_removed': encode_img_to_base64(processed_data['hair_removed']),
                'enhanced': encode_img_to_base64(processed_data['enhanced'])
            }
        
        # 2. Run Classification (PyTorch model or dynamic demo fallback)
        probabilities = {}
        predicted_key = ""
        used_demo_mode = True
        
        # Check if PyTorch is available and model is loaded successfully
        if PYTORCH_AVAILABLE and model is not None:
            try:
                # Load preprocessed (enhanced) image for inference
                pil_img = Image.fromarray(cv2.cvtColor(processed_data['enhanced'], cv2.COLOR_BGR2RGB))
                tensor_img = inference_transform(pil_img).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(tensor_img)
                    probs = torch.softmax(outputs, dim=1)[0].numpy()
                
                # Format output
                for i, prob in enumerate(probs):
                    key = LABEL_MAP[i]
                    probabilities[key] = float(prob)
                
                predicted_idx = int(np.argmax(probs))
                predicted_key = LABEL_MAP[predicted_idx]
                used_demo_mode = False
            except Exception as inference_error:
                print(f"PyTorch inference error: {inference_error}. Falling back to demo mode.")
                
        # Dynamic Demo Mode / Fallback Classification
        if used_demo_mode:
            # We want to make the prediction look realistic and deterministic for the same image.
            if CV2_AVAILABLE and not processed_data.get('is_fallback', False):
                # We can compute the average color/brightness of the lesion to select a classification dynamically.
                img_hsv = cv2.cvtColor(processed_data['enhanced'], cv2.COLOR_BGR2HSV)
                avg_hue = np.mean(img_hsv[:, :, 0])
                avg_sat = np.mean(img_hsv[:, :, 1])
                avg_val = np.mean(img_hsv[:, :, 2])
                score = (avg_hue * 0.3 + avg_sat * 0.4 + avg_val * 0.3)
            else:
                # Deterministic hash score based on file name length and characters
                score = abs(hash(file.filename))
            
            # Deterministic selection based on score
            class_keys = list(DISEASE_DATA.keys())
            predicted_idx = int(score) % len(class_keys)
            predicted_key = class_keys[predicted_idx]
            
            # Generate realistic probability distribution (highest for the predicted class)
            probs = np.random.dirichlet(np.ones(len(class_keys)) * 2) # smooth distribution
            max_idx = np.argmax(probs)
            # Swap max probability to our predicted key
            predicted_key_idx = class_keys.index(predicted_key)
            probs[max_idx], probs[predicted_key_idx] = probs[predicted_key_idx], probs[max_idx]
            
            for i, key in enumerate(class_keys):
                probabilities[key] = float(probs[i])
                
        # 3. Compile the response
        prediction_info = DISEASE_DATA[predicted_key]
        
        # Format prediction distributions for rendering
        results_distribution = []
        for key, prob in probabilities.items():
            results_distribution.append({
                'key': key,
                'name': DISEASE_DATA[key]['name'],
                'risk_level': DISEASE_DATA[key]['risk_level'],
                'probability': round(prob * 100, 2)
            })
            
        # Sort by probability descending
        results_distribution = sorted(results_distribution, key=lambda x: x['probability'], reverse=True)
        
        response = {
            'success': True,
            'demo_mode': used_demo_mode,
            'predicted_class': prediction_info['name'],
            'predicted_key': predicted_key,
            'risk_level': prediction_info['risk_level'],
            'type': prediction_info['type'],
            'description': prediction_info['description'],
            'precautions': prediction_info['precautions'],
            'skincare': prediction_info['skincare'],
            'medical_advice': prediction_info['medical_advice'],
            'probability': round(probabilities[predicted_key] * 100, 2),
            'distribution': results_distribution,
            'images': steps_base64
        }
        
        # Insert into Scan History database
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO scans (predicted_class, predicted_key, probability, risk_level, type, description, original_image, enhanced_image)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_info['name'],
                predicted_key,
                round(probabilities[predicted_key] * 100, 2),
                prediction_info['risk_level'],
                prediction_info['type'],
                prediction_info['description'],
                steps_base64['original'],
                steps_base64['enhanced']
            ))
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Database insertion error: {db_err}")

        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return jsonify(response)
        
    except Exception as e:
        # Clean up uploaded file in case of error
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, timestamp, predicted_class, predicted_key, probability, risk_level, type, description, original_image, enhanced_image FROM scans ORDER BY timestamp DESC LIMIT 20')
        rows = cursor.fetchall()
        
        history_list = []
        for row in rows:
            history_list.append({
                'id': row['id'],
                'timestamp': row['timestamp'],
                'predicted_class': row['predicted_class'],
                'predicted_key': row['predicted_key'],
                'probability': row['probability'],
                'risk_level': row['risk_level'],
                'type': row['type'],
                'description': row['description'],
                'original_image': row['original_image'],
                'enhanced_image': row['enhanced_image']
            })
        conn.close()
        return jsonify({'success': True, 'history': history_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history/delete/<int:scan_id>', methods=['POST'])
def delete_scan(scan_id):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Scan deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history/clear', methods=['POST'])
def clear_history():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scans')
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Skin Disease Classification Server...")
    app.run(debug=True, port=5000)

```

---

## model.py

> **Category**: Backend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/model.py`  
> **Description**: PyTorch deep learning network definition. Combines MobileNetV2 with a Squeeze-and-Excitation (SE) attention block for optimized skin classification.

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Attention Mechanism)
    Squeezes spatial dimension using Global Average Pooling,
    obtains channel-wise dependencies, and scales the input.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Compute channel-wise attention weights
        w = self.fc(x).view(b, c, 1, 1)
        # Re-weight the features map
        return x * w

class SkinDiseaseCNN(nn.Module):
    """
    Enhanced Skin Disease Classification Network.
    Uses MobileNetV2 as a Transfer Learning backbone,
    appends a Squeeze-and-Excitation (SE) Block for attention,
    and classifies into N skin disease categories.
    """
    def __init__(self, num_classes=7):
        super(SkinDiseaseCNN, self).__init__()
        # Load pre-trained MobileNetV2
        # Weights are initialized from ImageNet to leverage transfer learning
        weights = models.MobileNet_V2_Weights.DEFAULT
        self.backbone = models.mobilenet_v2(weights=weights)
        
        # Extract the features part
        self.features = self.backbone.features
        
        # Attention Mechanism: MobileNetV2 features output channel size is 1280
        self.attention = SEBlock(channels=1280, reduction=16)
        
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final Classifier
        # MobileNetV2 dropout is 0.2
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        # 1. Feature extraction using backbone (transfer learning)
        x = self.features(x)
        
        # 2. Apply attention mechanism (SE Block)
        x = self.attention(x)
        
        # 3. Pooling and flattening
        x = self.pool(x)
        x = torch.flatten(x, 1)
        
        # 4. Final classification
        x = self.classifier(x)
        return x

def get_model(num_classes=7):
    """
    Instantiates and returns the SkinDiseaseCNN model.
    """
    return SkinDiseaseCNN(num_classes=num_classes)

```

---

## utils.py

> **Category**: Backend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/utils.py`  
> **Description**: Image processing utility pipeline. Implements Dull Razor (morphological hair removal) and CLAHE (contrast enhancement in LAB color space).

```python
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
import numpy as np
import base64

def image_to_base64_fallback(image_path):
    """Fallback to read a file directly and return its base64 encoding when cv2 is missing."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def run_dull_razor(image):
    """
    Applies the Dull Razor algorithm to remove hair from skin lesion images.
    
    Steps:
    1. Grayscale conversion.
    2. Blackhat morphological filtering to locate hair structures (dark lines).
    3. Binary thresholding to create a hair mask.
    4. Image inpainting to fill in the hair pixels using neighboring skin pixels.
    """
    # 1. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Blackhat morphological filtering
    # A 9x9 kernel is typically effective for locating hair of various thicknesses
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Binary thresholding (intensify hair outline)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Image inpainting (telea algorithm works well for thin lines like hair)
    inpainted = cv2.inpaint(image, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return gray, blackhat, hair_mask, inpainted

def run_clahe_enhancement(image):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB color space.
    This enhances details of the lesion without causing excessive noise or color shifts.
    """
    # Convert from BGR to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel back with a and b channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr

def preprocess_pipeline(image_path):
    """
    Executes the full preprocessing pipeline on a skin lesion image.
    Returns:
        dictionary containing original and intermediate results as numpy arrays,
        or base64 strings if cv2 is not available.
    """
    if not CV2_AVAILABLE:
        # Graceful fallback: read file and return its base64 encoding directly for all steps
        b64_str = image_to_base64_fallback(image_path)
        return {
            'is_fallback': True,
            'original': b64_str,
            'grayscale': b64_str,
            'blackhat': b64_str,
            'hair_mask': b64_str,
            'hair_removed': b64_str,
            'enhanced': b64_str
        }

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image from path: " + image_path)
        
    # Run hair removal (Dull Razor)
    gray, blackhat, hair_mask, hair_removed = run_dull_razor(img)
    
    # Run contrast enhancement
    enhanced = run_clahe_enhancement(hair_removed)
    
    return {
        'is_fallback': False,
        'original': img,
        'grayscale': gray,
        'blackhat': blackhat,
        'hair_mask': hair_mask,
        'hair_removed': hair_removed,
        'enhanced': enhanced
    }

```

---

## requirements.txt

> **Category**: Backend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/requirements.txt`  
> **Description**: System requirements and Python package dependencies for the backend environment.

```text
Flask==3.1.3
torch==2.12.0+cpu
torchvision==0.27.0+cpu
opencv-python-headless==4.13.0.92
numpy==2.4.6
pillow==12.2.0
scikit-learn==1.9.0
matplotlib==3.10.9

```

---

## templates/index.html

> **Category**: Frontend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/templates/index.html`  
> **Description**: HTML Dashboard template. Implements structural grid systems, UI sections for image preview, visualizer, stats, and a sliding drawer for history.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DermVision AI - Skin Disease Detection & Decision Support</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- FontAwesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <div class="background-glows">
        <div class="glow glow-1"></div>
        <div class="glow glow-2"></div>
        <div class="glow glow-3"></div>
    </div>

    <div class="app-container">
        <!-- Header -->
        <header class="app-header animate-fade-in">
            <div class="logo-area">
                <div class="logo-icon">
                    <i class="fa-solid fa-laptop-medical"></i>
                </div>
                <div class="logo-text">
                    <h1>DermVision <span>AI</span></h1>
                    <p>Enhanced Deep Learning Framework for Skin Lesion Classification</p>
                </div>
            </div>
            <div class="header-controls">
                <button class="btn btn-secondary btn-sm" id="history-toggle-btn">
                    <i class="fa-solid fa-clock-rotate-left"></i> Scan History
                </button>
                <div class="status-indicator">
                    <span class="status-dot {% if pytorch_status %}active{% else %}warning{% endif %}"></span>
                    <span class="status-label">
                        {% if pytorch_status %}
                            PyTorch Core Active
                        {% else %}
                            Demo Fallback Active (No PyTorch)
                        {% endif %}
                    </span>
                </div>
            </div>
        </header>

        <!-- Main Workspace -->
        <main class="workspace-grid">
            <!-- Left Panel: Input & Preprocessing -->
            <section class="workspace-card left-panel animate-slide-up">
                <div class="card-header">
                    <h2><i class="fa-solid fa-camera"></i> Lesion Image Analysis</h2>
                    <p>Upload a dermoscopic image to execute the preprocessing and classification pipeline.</p>
                </div>

                <!-- Upload Section -->
                <div class="upload-zone" id="drop-zone">
                    <input type="file" id="file-input" accept="image/*" style="display: none;">
                    <div class="upload-content" id="upload-prompt">
                        <div class="upload-icon-wrapper">
                            <i class="fa-solid fa-cloud-arrow-up"></i>
                        </div>
                        <h3>Drag & drop your image here</h3>
                        <p>Supports JPEG, PNG up to 10MB</p>
                        <button class="btn btn-primary" id="browse-btn">
                            <i class="fa-solid fa-folder-open"></i> Browse Files
                        </button>
                    </div>
                    
                    <!-- Preview and Loading State -->
                    <div class="upload-preview hidden" id="upload-preview">
                        <img id="preview-img" src="" alt="Lesion Preview">
                        <div class="preview-overlay" id="loading-overlay">
                            <div class="spinner"></div>
                            <p id="loading-status">Executing Morphological Dull Razor...</p>
                        </div>
                    </div>
                </div>

                <!-- Preprocessing Visualizer Tabs -->
                <div class="preprocessing-container hidden" id="preprocessing-panel">
                    <div class="section-title">
                        <h3><i class="fa-solid fa-wand-magic-sparkles"></i> Preprocessing Pipeline Visualizer</h3>
                        <p>Click tabs to view each stage of the image processing sequence.</p>
                    </div>
                    
                    <div class="tabs-header" id="preprocessing-tabs">
                        <button class="tab-btn active" data-step="original">Original</button>
                        <button class="tab-btn" data-step="grayscale">Grayscale</button>
                        <button class="tab-btn" data-step="blackhat">Morphological</button>
                        <button class="tab-btn" data-step="hair_mask">Hair Mask</button>
                        <button class="tab-btn" data-step="hair_removed">Hair Removed</button>
                        <button class="tab-btn" data-step="enhanced">Contrast Enhanced</button>
                    </div>
                    
                    <div class="tab-content-wrapper">
                        <div class="tab-view-frame">
                            <img id="tab-display-img" src="" alt="Preprocessing Step">
                            <div class="step-explanation" id="step-desc">
                                Original lesion image uploaded by user.
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Right Panel: Results & Decision Support -->
            <section class="workspace-card right-panel animate-slide-up" style="animation-delay: 0.1s;">
                <!-- Initial State: Welcome & Project Abstract -->
                <div class="welcome-view" id="welcome-view">
                    <div class="welcome-icon">
                        <i class="fa-solid fa-user-doctor"></i>
                    </div>
                    <h2>Clinical Decision Support</h2>
                    <p class="welcome-text">
                        Welcome to the DermVision AI framework. This platform is designed to assist in dermatological screening by combining advanced digital image processing with deep transfer learning.
                    </p>
                    <div class="abstract-box">
                        <h4><i class="fa-solid fa-circle-info"></i> Methodology Overview</h4>
                        <ul>
                            <li><strong>Preprocessing:</strong> Artifacts (hairs) are segmented using mathematical morphology (Blackhat kernel) and inpainted. Contrast is dynamically balanced via CLAHE in LAB space.</li>
                            <li><strong>Classification:</strong> Features are extracted using a MobileNetV2 architecture fine-tuned via Transfer Learning.</li>
                            <li><strong>Attention Mechanism:</strong> A Squeeze-and-Excitation (SE) block adaptively reweights channel feature maps to focus on the lesion borders.</li>
                            <li><strong>Decision Support:</strong> Provides clinically aligned precautions, skincare guidelines, and dermatologist referral triggers.</li>
                        </ul>
                    </div>
                    <div class="member-credits">
                        <p><strong>Developed by Group G-13:</strong> V. Bhanu Vaishnavi, S. Harshitha, N. Shravani</p>
                        <p><strong>Project Guide:</strong> Mrs. N. Meghana</p>
                    </div>
                </div>

                <!-- Results View (Hidden Initially) -->
                <div class="results-view hidden" id="results-view">
                    <!-- Top Diagnostics Card -->
                    <div class="diagnostic-card">
                        <div class="diagnostic-badge" id="risk-badge">Low Risk</div>
                        <div class="diagnostic-main">
                            <h3 id="result-title">Melanocytic Nevi</h3>
                            <p class="diagnostic-type" id="result-type">Benign Lesion</p>
                        </div>
                        <div class="diagnostic-score">
                            <div class="score-circle">
                                <svg class="progress-ring" width="70" height="70">
                                    <circle class="progress-ring__circle-bg" stroke="rgba(255,255,255,0.05)" stroke-width="6" fill="transparent" r="28" cx="35" cy="35"/>
                                    <circle class="progress-ring__circle" id="confidence-ring" stroke="var(--accent-color)" stroke-width="6" fill="transparent" r="28" cx="35" cy="35"/>
                                </svg>
                                <div class="score-value"><span id="result-confidence">98</span>%</div>
                            </div>
                            <span class="score-label">Confidence</span>
                        </div>
                    </div>

                    <!-- Demo Mode Alert -->
                    <div class="demo-alert hidden" id="demo-alert">
                        <i class="fa-solid fa-triangle-exclamation"></i>
                        <span><strong>Demo Mode Active:</strong> PyTorch was unavailable. Results are simulated, but image preprocessing is fully active.</span>
                    </div>

                    <!-- Description -->
                    <div class="results-section">
                        <h4>Description</h4>
                        <p id="result-desc" class="lesion-description">Lesion details...</p>
                    </div>

                    <!-- Probability Bars -->
                    <div class="results-section">
                        <h4>Probability Distribution</h4>
                        <div class="distribution-list" id="distribution-list">
                            <!-- Populated by JS -->
                        </div>
                    </div>

                    <!-- Decision Support Module -->
                    <div class="results-section decision-support-section">
                        <div class="decision-header">
                            <h4><i class="fa-solid fa-shield-halved"></i> Decision Support & Guidance</h4>
                        </div>
                        
                        <div class="decision-tabs">
                            <button class="decision-tab-btn active" data-section="skincare">
                                <i class="fa-solid fa-pump-soap"></i> Skincare Support
                            </button>
                            <button class="decision-tab-btn" data-section="precautions">
                                <i class="fa-solid fa-hand-holding-hand"></i> Precautions
                            </button>
                            <button class="decision-tab-btn" data-section="medical">
                                <i class="fa-solid fa-stethoscope"></i> Dermatologist Guide
                            </button>
                        </div>

                        <div class="decision-content">
                            <!-- Skincare Section -->
                            <div class="decision-panel active" id="panel-skincare">
                                <ul id="list-skincare" class="support-list">
                                    <!-- Populated by JS -->
                                </ul>
                            </div>
                            <!-- Precautions Section -->
                            <div class="decision-panel" id="panel-precautions">
                                <ul id="list-precautions" class="support-list">
                                    <!-- Populated by JS -->
                                </ul>
                            </div>
                            <!-- Medical Advice Section -->
                            <div class="decision-panel" id="panel-medical">
                                <div class="warning-alert-box" id="warning-box">
                                    <i class="fa-solid fa-triangle-exclamation"></i>
                                    <p id="text-medical">Advice...</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Action Buttons -->
                    <div class="action-footer">
                        <button class="btn btn-secondary" id="reset-btn">
                            <i class="fa-solid fa-rotate-left"></i> Analyze Another Image
                        </button>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <!-- History Drawer -->
    <div class="history-drawer" id="history-drawer">
        <div class="drawer-overlay" id="drawer-overlay"></div>
        <div class="drawer-content">
            <div class="drawer-header">
                <h2><i class="fa-solid fa-clock-rotate-left"></i> Scan History</h2>
                <button class="close-btn" id="close-drawer-btn">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
            
            <div class="drawer-actions">
                <button class="btn btn-danger-outline btn-sm" id="clear-history-btn">
                    <i class="fa-solid fa-trash-can"></i> Clear All Scans
                </button>
            </div>
            
            <div class="drawer-body" id="history-list-container">
                <!-- Populated dynamically via JS -->
                <div class="empty-history" id="empty-history-msg">
                    <i class="fa-regular fa-folder-open"></i>
                    <p>No scans recorded yet</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Custom JS -->
    <script src="{{ url_for('static', filename='js/app.js') }}"></script>
</body>
</html>

```

---

## static/css/style.css

> **Category**: Frontend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/static/css/style.css`  
> **Description**: Custom stylesheet. Styles the glassmorphic aesthetics, glowing background layers, dynamic progress indicators, and custom dark mode components.

```css
/* ==========================================================================
   DERMVISION AI DESIGN SYSTEM
   ========================================================================== */

:root {
    --bg-primary: #0a0d14;
    --bg-surface: rgba(20, 26, 38, 0.7);
    --bg-card: rgba(30, 38, 56, 0.55);
    --border-color: rgba(255, 255, 255, 0.08);
    --border-focus: rgba(0, 180, 216, 0.5);
    
    /* Elegant Harmonious HSL Color Palette */
    --text-primary: #f3f4f6;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    
    --primary: #00b4d8;         /* Cyan */
    --primary-glow: rgba(0, 180, 216, 0.35);
    --accent: #7209b7;          /* Royal Purple */
    --accent-glow: rgba(114, 9, 183, 0.3);
    
    /* Status Colors */
    --status-low: #10b981;      /* Emerald Green */
    --status-low-glow: rgba(16, 185, 129, 0.2);
    --status-medium: #f59e0b;   /* Amber */
    --status-medium-glow: rgba(245, 158, 11, 0.2);
    --status-high: #ef4444;     /* Crimson Red */
    --status-high-glow: rgba(239, 68, 68, 0.2);
    
    --font-heading: 'Outfit', sans-serif;
    --font-body: 'Inter', sans-serif;
    --border-radius-lg: 16px;
    --border-radius-md: 10px;
    --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Base resets */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-body);
    min-height: 100vh;
    overflow-x: hidden;
    position: relative;
    line-height: 1.5;
}

/* Background Glowing Orbs */
.background-glows {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
    pointer-events: none;
}

.glow {
    position: absolute;
    border-radius: 50%;
    filter: blur(140px);
    opacity: 0.25;
}

.glow-1 {
    top: -10%;
    left: 10%;
    width: 500px;
    height: 500px;
    background-color: var(--primary);
    animation: float 25s infinite alternate;
}

.glow-2 {
    bottom: -10%;
    right: 10%;
    width: 600px;
    height: 600px;
    background-color: var(--accent);
    animation: float 30s infinite alternate-reverse;
}

.glow-3 {
    top: 40%;
    left: 50%;
    width: 300px;
    height: 300px;
    background-color: #3f37c9;
    opacity: 0.15;
}

@keyframes float {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(60px, 40px) scale(1.1); }
}

/* Container */
.app-container {
    max-width: 1440px;
    margin: 0 auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    gap: 24px;
}

/* Header */
.app-header {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 20px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.logo-area {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    border-radius: var(--border-radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    color: white;
    box-shadow: 0 0 15px var(--primary-glow);
}

.logo-text h1 {
    font-family: var(--font-heading);
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.logo-text h1 span {
    color: var(--primary);
    text-shadow: 0 0 10px var(--primary-glow);
}

.logo-text p {
    font-size: 13px;
    color: var(--text-secondary);
    font-weight: 400;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(255, 255, 255, 0.04);
    padding: 8px 16px;
    border-radius: 30px;
    border: 1px solid var(--border-color);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.status-dot.active {
    background-color: var(--status-low);
    box-shadow: 0 0 8px var(--status-low);
    animation: pulse 2s infinite;
}

.status-dot.warning {
    background-color: var(--status-medium);
    box-shadow: 0 0 8px var(--status-medium);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.95); opacity: 0.7; }
    50% { transform: scale(1.15); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.7; }
}

.status-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
}

/* Grid Layout */
.workspace-grid {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 24px;
    flex-grow: 1;
}

@media (max-width: 1024px) {
    .workspace-grid {
        grid-template-columns: 1fr;
    }
}

/* Workspace Card (Glassmorphism) */
.workspace-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 30px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    display: flex;
    flex-direction: column;
    gap: 24px;
    position: relative;
    overflow: hidden;
}

.card-header h2 {
    font-family: var(--font-heading);
    font-size: 20px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 10px;
}

.card-header h2 i {
    color: var(--primary);
}

.card-header p {
    font-size: 13px;
    color: var(--text-secondary);
    margin-top: 4px;
}

/* Drag and Drop Zone */
.upload-zone {
    border: 2px dashed rgba(255, 255, 255, 0.15);
    border-radius: var(--border-radius-lg);
    padding: 40px 20px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: rgba(255, 255, 255, 0.02);
    min-height: 280px;
    cursor: pointer;
    transition: var(--transition-smooth);
    position: relative;
    overflow: hidden;
}

.upload-zone:hover, .upload-zone.dragover {
    border-color: var(--primary);
    background: rgba(0, 180, 216, 0.03);
    box-shadow: inset 0 0 20px rgba(0, 180, 216, 0.05);
}

.upload-icon-wrapper {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    color: var(--primary);
    margin-bottom: 16px;
    transition: var(--transition-smooth);
}

.upload-zone:hover .upload-icon-wrapper {
    transform: translateY(-5px);
    background: rgba(0, 180, 216, 0.1);
    border-color: var(--primary);
    box-shadow: 0 0 15px var(--primary-glow);
}

.upload-content h3 {
    font-family: var(--font-heading);
    font-size: 16px;
    font-weight: 500;
    margin-bottom: 6px;
}

.upload-content p {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 18px;
}

/* Buttons */
.btn {
    font-family: var(--font-heading);
    font-size: 13px;
    font-weight: 600;
    padding: 10px 20px;
    border-radius: 30px;
    border: none;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: var(--transition-smooth);
}

.btn-primary {
    background: linear-gradient(135deg, #00b4d8, #0077b6);
    color: white;
    box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3);
}

.btn-primary:hover {
    box-shadow: 0 4px 20px rgba(0, 180, 216, 0.5);
    transform: translateY(-2px);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
}

/* Preview Area */
.upload-preview {
    width: 100%;
    height: 100%;
    position: absolute;
    top: 0;
    left: 0;
    z-index: 10;
    background: var(--bg-primary);
    display: flex;
    align-items: center;
    justify-content: center;
}

.upload-preview img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.preview-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(10, 13, 20, 0.85);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    backdrop-filter: blur(5px);
}

/* Spinner */
.spinner {
    width: 50px;
    height: 50px;
    border: 3px solid rgba(255, 255, 255, 0.05);
    border-radius: 50%;
    border-top-color: var(--primary);
    animation: spin 1s linear infinite;
    box-shadow: 0 0 10px var(--primary-glow);
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

#loading-status {
    font-size: 13px;
    color: var(--primary);
    font-weight: 500;
    letter-spacing: 0.5px;
    animation: pulse-text 1.5s infinite ease-in-out;
}

@keyframes pulse-text {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

/* Preprocessing Panel */
.preprocessing-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    border-top: 1px solid var(--border-color);
    padding-top: 24px;
}

.section-title h3 {
    font-family: var(--font-heading);
    font-size: 15px;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title p {
    font-size: 12px;
    color: var(--text-secondary);
}

.tabs-header {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    background: rgba(255, 255, 255, 0.02);
    padding: 4px;
    border-radius: 30px;
    border: 1px solid var(--border-color);
}

.tab-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 20px;
    cursor: pointer;
    transition: var(--transition-smooth);
}

.tab-btn:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.04);
}

.tab-btn.active {
    background: var(--primary);
    color: #05050a;
    font-weight: 600;
    box-shadow: 0 2px 10px var(--primary-glow);
}

.tab-content-wrapper {
    background: rgba(0, 0, 0, 0.2);
    border-radius: var(--border-radius-lg);
    border: 1px solid var(--border-color);
    padding: 16px;
    display: flex;
    justify-content: center;
}

.tab-view-frame {
    width: 100%;
    max-width: 380px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}

.tab-view-frame img {
    width: 100%;
    height: 280px;
    border-radius: var(--border-radius-md);
    object-fit: cover;
    border: 1px solid rgba(255, 255, 255, 0.05);
    transition: var(--transition-smooth);
}

.tab-view-frame img:hover {
    transform: scale(1.02);
}

.step-explanation {
    font-size: 12px;
    color: var(--text-secondary);
    text-align: center;
    background: rgba(255, 255, 255, 0.03);
    padding: 8px 12px;
    border-radius: var(--border-radius-md);
    width: 100%;
}

/* Welcome View */
.welcome-view {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    height: 100%;
    min-height: 450px;
    padding: 20px 0;
}

.welcome-icon {
    font-size: 50px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
    filter: drop-shadow(0 0 10px var(--primary-glow));
}

.welcome-view h2 {
    font-family: var(--font-heading);
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 10px;
}

.welcome-text {
    font-size: 14px;
    color: var(--text-secondary);
    max-width: 440px;
    margin-bottom: 24px;
    line-height: 1.6;
}

.abstract-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 20px;
    text-align: left;
    width: 100%;
    max-width: 480px;
    margin-bottom: 24px;
}

.abstract-box h4 {
    font-family: var(--font-heading);
    font-size: 14px;
    font-weight: 600;
    color: var(--primary);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.abstract-box ul {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.abstract-box li {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.4;
    padding-left: 14px;
    position: relative;
}

.abstract-box li::before {
    content: "•";
    color: var(--primary);
    position: absolute;
    left: 0;
    top: 0;
}

.member-credits {
    font-size: 11px;
    color: var(--text-muted);
    border-top: 1px solid var(--border-color);
    padding-top: 16px;
    width: 100%;
    max-width: 480px;
}

/* Results View */
.results-view {
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.diagnostic-card {
    background: linear-gradient(135deg, rgba(30, 38, 56, 0.7), rgba(15, 20, 31, 0.8));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--border-radius-lg);
    padding: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.diagnostic-badge {
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Risk levels styling */
.risk-low {
    background-color: var(--status-low-glow);
    color: var(--status-low);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.risk-medium {
    background-color: var(--status-medium-glow);
    color: var(--status-medium);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.risk-high {
    background-color: var(--status-high-glow);
    color: var(--status-high);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.diagnostic-main {
    flex-grow: 1;
    margin-left: 20px;
}

.diagnostic-main h3 {
    font-family: var(--font-heading);
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.2px;
}

.diagnostic-type {
    font-size: 12px;
    color: var(--text-secondary);
}

/* Progress Ring */
.score-circle {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.progress-ring__circle {
    transition: stroke-dashoffset 0.35s;
    transform: rotate(-90deg);
    transform-origin: 50% 50%;
}

.score-value {
    position: absolute;
    font-family: var(--font-heading);
    font-size: 16px;
    font-weight: 800;
}

.score-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    text-align: center;
    display: block;
    margin-top: 4px;
    font-weight: 600;
}

/* Demo Alert Banner */
.demo-alert {
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: var(--border-radius-md);
    padding: 12px 16px;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 12px;
    color: #fbbf24;
}

.demo-alert i {
    font-size: 16px;
}

.results-section h4 {
    font-family: var(--font-heading);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-muted);
    margin-bottom: 10px;
    font-weight: 700;
}

.lesion-description {
    font-size: 13px;
    line-height: 1.6;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.02);
    padding: 14px;
    border-radius: var(--border-radius-md);
    border: 1px solid var(--border-color);
}

/* Probability Bars List */
.distribution-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.distribution-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.dist-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    font-weight: 500;
}

.dist-name {
    color: var(--text-primary);
}

.dist-value {
    font-weight: 600;
    color: var(--primary);
}

.dist-bar-bg {
    height: 8px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 4px;
    width: 100%;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.02);
}

.dist-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--primary));
    border-radius: 4px;
    width: 0%;
    transition: width 1s ease-out;
}

/* Decision Support Section */
.decision-support-section {
    background: rgba(255, 255, 255, 0.01);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-lg);
    padding: 16px;
}

.decision-header h4 {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--primary);
}

.decision-tabs {
    display: flex;
    border-bottom: 1px solid var(--border-color);
    gap: 4px;
    margin-bottom: 16px;
}

.decision-tab-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    position: relative;
    transition: var(--transition-smooth);
    display: flex;
    align-items: center;
    gap: 6px;
}

.decision-tab-btn:hover {
    color: var(--text-primary);
}

.decision-tab-btn.active {
    color: var(--primary);
}

.decision-tab-btn.active::after {
    content: "";
    position: absolute;
    bottom: -1px;
    left: 0;
    width: 100%;
    height: 2px;
    background: var(--primary);
    box-shadow: 0 0 8px var(--primary);
}

.decision-panel {
    display: none;
    min-height: 120px;
}

.decision-panel.active {
    display: block;
}

.support-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.support-list li {
    font-size: 12.5px;
    color: var(--text-secondary);
    line-height: 1.5;
    padding-left: 20px;
    position: relative;
}

.support-list li::before {
    content: "\f058"; /* check circle icon */
    font-family: "Font Awesome 6 Free";
    font-weight: 900;
    color: var(--primary);
    position: absolute;
    left: 0;
    top: 2px;
}

#panel-medical .warning-alert-box {
    background: rgba(239, 68, 68, 0.05);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: var(--border-radius-md);
    padding: 16px;
    display: flex;
    gap: 14px;
    align-items: flex-start;
}

#panel-medical i {
    color: var(--status-high);
    font-size: 18px;
    margin-top: 2px;
}

#text-medical {
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--text-primary);
}

/* Action Footer */
.action-footer {
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
}

/* Animations */
.animate-fade-in {
    animation: fadeIn 0.6s ease-out forwards;
}

.animate-slide-up {
    opacity: 0;
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.hidden {
    display: none !important;
}

/* ==========================================================================
   SCAN HISTORY DRAWER SYSTEM STYLING
   ========================================================================== */

.header-controls {
    display: flex;
    align-items: center;
    gap: 16px;
}

#history-toggle-btn {
    padding: 8px 16px;
    font-size: 12px;
    border-radius: 30px;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    transition: var(--transition-smooth);
}

#history-toggle-btn:hover {
    background: rgba(0, 180, 216, 0.1);
    border-color: var(--primary);
    box-shadow: 0 0 10px rgba(0, 180, 216, 0.2);
}

/* Drawer Layout */
.history-drawer {
    position: fixed;
    top: 0;
    right: 0;
    width: 100vw;
    height: 100vh;
    z-index: 999;
    pointer-events: none;
    transition: var(--transition-smooth);
}

.history-drawer.open {
    pointer-events: auto;
}

.drawer-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(5, 7, 12, 0.6);
    backdrop-filter: blur(4px);
    opacity: 0;
    transition: opacity 0.4s ease;
    pointer-events: none;
}

.history-drawer.open .drawer-overlay {
    opacity: 1;
    pointer-events: auto;
}

.drawer-content {
    position: absolute;
    top: 0;
    right: -420px;
    width: 420px;
    max-width: 90vw;
    height: 100%;
    background: rgba(15, 22, 36, 0.95);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border-left: 1px solid var(--border-color);
    box-shadow: -10px 0 40px rgba(0, 0, 0, 0.5);
    display: flex;
    flex-direction: column;
    padding: 30px;
    transition: right 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.history-drawer.open .drawer-content {
    right: 0;
}

.drawer-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid var(--border-color);
}

.drawer-header h2 {
    font-family: var(--font-heading);
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.drawer-header h2 i {
    color: var(--primary);
}

.drawer-header .close-btn {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 20px;
    cursor: pointer;
    transition: var(--transition-smooth);
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.drawer-header .close-btn:hover {
    color: var(--text-primary);
    background: rgba(255, 255, 255, 0.05);
}

.drawer-actions {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 20px;
}

.btn-danger-outline {
    background: rgba(239, 68, 68, 0.05);
    color: var(--status-high);
    border: 1px solid rgba(239, 68, 68, 0.2);
    padding: 8px 14px;
    font-size: 11px;
    border-radius: 20px;
    cursor: pointer;
    transition: var(--transition-smooth);
}

.btn-danger-outline:hover {
    background: rgba(239, 68, 68, 0.15);
    border-color: var(--status-high);
    transform: translateY(-1px);
}

.drawer-body {
    flex-grow: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding-right: 5px;
}

/* Scrollbar for drawer body */
.drawer-body::-webkit-scrollbar {
    width: 6px;
}
.drawer-body::-webkit-scrollbar-track {
    background: transparent;
}
.drawer-body::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}
.drawer-body::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* Empty history style */
.empty-history {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    height: 200px;
    color: var(--text-muted);
}

.empty-history i {
    font-size: 40px;
}

.empty-history p {
    font-size: 13px;
}

/* History Item Card */
.history-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius-md);
    padding: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    cursor: pointer;
    transition: var(--transition-smooth);
    position: relative;
    overflow: hidden;
}

.history-item:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(0, 180, 216, 0.2);
    transform: translateX(-3px);
}

.history-item-thumb {
    width: 60px;
    height: 60px;
    border-radius: 8px;
    overflow: hidden;
    flex-shrink: 0;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.history-item-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.history-item-details {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0; /* allows text truncation */
}

.history-item-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}

.history-item-title {
    font-family: var(--font-heading);
    font-size: 13.5px;
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-item-badge {
    font-size: 9px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 10px;
    text-transform: uppercase;
    flex-shrink: 0;
}

.history-item-meta {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-secondary);
}

.history-item-meta strong {
    color: var(--primary);
}

.history-item-actions {
    opacity: 0;
    transition: var(--transition-smooth);
    margin-left: 4px;
}

.history-item:hover .history-item-actions {
    opacity: 1;
}

.delete-item-btn {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 14px;
    cursor: pointer;
    transition: var(--transition-smooth);
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.delete-item-btn:hover {
    color: var(--status-high);
    background: rgba(239, 68, 68, 0.1);
}

@media (max-width: 480px) {
    .drawer-content {
        width: 100%;
        padding: 20px;
    }
}

```

---

## static/js/app.js

> **Category**: Frontend Development  
> **File Location**: `file:///c:/VAISHU (Certificates)/Mini Project/Skin Disease Detection/static/js/app.js`  
> **Description**: Frontend controller. Implements client interaction, drag-and-drop image processing simulations, database REST interaction, and rendering logic.

```javascript
/* ==========================================================================
   DERMVISION AI FRONTEND CONTROLLER
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadPreview = document.getElementById('upload-preview');
    const previewImg = document.getElementById('preview-img');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingStatus = document.getElementById('loading-status');
    
    // Panel Elements
    const preprocessingPanel = document.getElementById('preprocessing-panel');
    const preprocessingTabs = document.getElementById('preprocessing-tabs');
    const tabDisplayImg = document.getElementById('tab-display-img');
    const stepDesc = document.getElementById('step-desc');
    const welcomeView = document.getElementById('welcome-view');
    const resultsView = document.getElementById('results-view');
    const demoAlert = document.getElementById('demo-alert');
    
    // Diagnostics Elements
    const resultTitle = document.getElementById('result-title');
    const resultType = document.getElementById('result-type');
    const riskBadge = document.getElementById('risk-badge');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceRing = document.getElementById('confidence-ring');
    const resultDesc = document.getElementById('result-desc');
    const distributionList = document.getElementById('distribution-list');
    
    // Decision Support Elements
    const decisionTabs = document.querySelector('.decision-tabs');
    const listSkincare = document.getElementById('list-skincare');
    const listPrecautions = document.getElementById('list-precautions');
    const textMedical = document.getElementById('text-medical');
    const warningBox = document.getElementById('warning-box');
    
    // Control Buttons
    const resetBtn = document.getElementById('reset-btn');

    // History Elements
    const historyDrawer = document.getElementById('history-drawer');
    const historyToggleBtn = document.getElementById('history-toggle-btn');
    const closeDrawerBtn = document.getElementById('close-drawer-btn');
    const drawerOverlay = document.getElementById('drawer-overlay');
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    const historyListContainer = document.getElementById('history-list-container');
    const emptyHistoryMsg = document.getElementById('empty-history-msg');

    // Global variable to hold response images
    let predictionImages = {};
    
    // Descriptions for each preprocessing stage
    const stepDescriptions = {
        original: "Original dermoscopic skin lesion image uploaded for screening.",
        grayscale: "Grayscale conversion simplifies the color space to prepare the image for intensity-based morphological filters.",
        blackhat: "Blackhat morphological transformation highlights narrow dark features (like hairs) that are darker than their surroundings.",
        hair_mask: "Binary thresholding creates a precise pixel-map segmenting hair structures from the underlying skin lesion.",
        hair_removed: "Telea inpainting reconstructs hair-covered pixels using the mathematical average of surrounding skin texture, eliminating occlusion.",
        enhanced: "Contrast Limited Adaptive Histogram Equalization is applied to the L channel in LAB color space to reveal boundary and color variation without introducing noise."
    };

    // Circular Progress Ring Math
    const radius = confidenceRing.r.baseVal.value;
    const circumference = radius * 2 * Math.PI;
    confidenceRing.style.strokeDasharray = `${circumference} ${circumference}`;
    setProgress(0);

    function setProgress(percent) {
        const offset = circumference - (percent / 100) * circumference;
        confidenceRing.style.strokeDashoffset = offset;
    }

    // ==========================================================================
    // UPLOAD & EVENT LISTENERS
    // ==========================================================================

    // History Drawer listeners
    historyToggleBtn.addEventListener('click', () => {
        historyDrawer.classList.add('open');
        loadHistory();
    });

    closeDrawerBtn.addEventListener('click', () => {
        historyDrawer.classList.remove('open');
    });

    drawerOverlay.addEventListener('click', () => {
        historyDrawer.classList.remove('open');
    });

    clearHistoryBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to clear all scan history?')) {
            clearHistory();
        }
    });

    // Browse button triggers file input
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', () => {
        if (uploadPreview.classList.contains('hidden')) {
            fileInput.click();
        }
    });

    // Drag and drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Handle selected file
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file (PNG/JPG).');
            return;
        }

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            uploadPreview.classList.remove('hidden');
            loadingOverlay.classList.remove('hidden');
            
            // Start pipeline simulation text, then upload
            animateLoadingStatus(file);
        };
        reader.readAsDataURL(file);
    }

    // Loading status animation
    function animateLoadingStatus(file) {
        const statuses = [
            "Morphological Dull Razor: Gray conversion...",
            "Morphological Dull Razor: Hair detection...",
            "Morphological Dull Razor: Telea inpainting...",
            "Lesion enhancement: CIELAB CLAHE scaling...",
            "Transfer Learning: Extracting feature maps...",
            "Attention Model: Squeeze-and-Excitation reweighting..."
        ];

        let index = 0;
        loadingStatus.textContent = statuses[0];
        
        const interval = setInterval(() => {
            index++;
            if (index < statuses.length) {
                loadingStatus.textContent = statuses[index];
            } else {
                clearInterval(interval);
            }
        }, 800);

        // Actually upload image to backend
        uploadImage(file, interval);
    }

    // Upload image via Ajax
    function uploadImage(file, interval) {
        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            clearInterval(interval);
            if (data.success) {
                renderResults(data);
                loadHistory(); // Refresh history
            } else {
                alert('Analysis failed: ' + data.error);
                resetApp();
            }
        })
        .catch(error => {
            clearInterval(interval);
            console.error('Error:', error);
            alert('An error occurred during analysis: ' + error.message);
            resetApp();
        });
    }

    // ==========================================================================
    // RESULT RENDERING
    // ==========================================================================

    function renderResults(data) {
        loadingOverlay.classList.add('hidden');
        welcomeView.classList.add('hidden');
        resultsView.classList.remove('hidden');
        preprocessingPanel.classList.remove('hidden');
        
        // Save images for tab switcher
        predictionImages = data.images;
        
        // Load original step by default in preprocessing visualizer
        setPreprocessingTab('original');

        // Set Diagnosis details
        resultTitle.textContent = data.predicted_class;
        resultType.textContent = data.type;
        resultDesc.textContent = data.description;
        
        // Confidence ring & score
        resultConfidence.textContent = Math.round(data.probability);
        setProgress(data.probability);

        // Risk level badge
        riskBadge.className = 'diagnostic-badge'; // reset
        const risk = data.risk_level.toLowerCase();
        if (risk.includes('low')) {
            riskBadge.classList.add('risk-low');
            riskBadge.textContent = 'Low Risk';
            warningBox.className = 'warning-alert-box';
        } else if (risk.includes('medium') && !risk.includes('high')) {
            riskBadge.classList.add('risk-medium');
            riskBadge.textContent = 'Medium Risk';
            warningBox.className = 'warning-alert-box warning-box-medium';
        } else {
            riskBadge.classList.add('risk-high');
            riskBadge.textContent = 'High Risk';
            warningBox.className = 'warning-alert-box warning-box-high';
        }

        // Demo Mode Banner
        if (data.demo_mode) {
            demoAlert.classList.remove('hidden');
        } else {
            demoAlert.classList.add('hidden');
        }

        // Render Probability distribution chart list
        distributionList.innerHTML = '';
        data.distribution.forEach(item => {
            const container = document.createElement('div');
            container.className = 'distribution-item';
            
            const label = document.createElement('div');
            label.className = 'dist-label';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'dist-name';
            nameSpan.textContent = item.name;
            
            const valueSpan = document.createElement('span');
            valueSpan.className = 'dist-value';
            valueSpan.textContent = `${item.probability}%`;
            
            label.appendChild(nameSpan);
            label.appendChild(valueSpan);
            
            const bgBar = document.createElement('div');
            bgBar.className = 'dist-bar-bg';
            
            const fillBar = document.createElement('div');
            fillBar.className = 'dist-bar-fill';
            // Trigger animation on next tick
            setTimeout(() => {
                fillBar.style.width = `${item.probability}%`;
            }, 50);

            bgBar.appendChild(fillBar);
            container.appendChild(label);
            container.appendChild(bgBar);
            distributionList.appendChild(container);
        });

        // Render Decision Support Lists
        listSkincare.innerHTML = '';
        data.skincare.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            listSkincare.appendChild(li);
        });

        listPrecautions.innerHTML = '';
        data.precautions.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            listPrecautions.appendChild(li);
        });

        textMedical.textContent = data.medical_advice;
    }

    // ==========================================================================
    // TAB INTERACTIONS
    // ==========================================================================

    // Preprocessing tabs click event
    preprocessingTabs.addEventListener('click', (e) => {
        const button = e.target.closest('.tab-btn');
        if (!button) return;
        
        // Remove active class from all buttons
        preprocessingTabs.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        
        // Add active class to clicked button
        button.classList.add('active');
        
        // Set tab step
        const step = button.getAttribute('data-step');
        setPreprocessingTab(step);
    });

    function setPreprocessingTab(step) {
        if (!predictionImages[step]) return;
        
        // Set image source
        tabDisplayImg.src = `data:image/png;base64,${predictionImages[step]}`;
        
        // Set explanation
        stepDesc.textContent = stepDescriptions[step];
    }

    // Decision support tabs click event
    decisionTabs.addEventListener('click', (e) => {
        const button = e.target.closest('.decision-tab-btn');
        if (!button) return;

        // Remove active class
        decisionTabs.querySelectorAll('.decision-tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.decision-panel').forEach(panel => panel.classList.remove('active'));

        // Add active class
        button.classList.add('active');
        const section = button.getAttribute('data-section');
        document.getElementById(`panel-${section}`).classList.add('active');
    });

    // ==========================================================================
    // RESET & RE-ANALYZE
    // ==========================================================================

    resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        // Clear inputs
        fileInput.value = '';
        
        // Reset previews
        previewImg.src = '';
        uploadPreview.classList.add('hidden');
        uploadPrompt.classList.remove('hidden');
        loadingOverlay.classList.add('hidden');
        
        // Reset views
        welcomeView.classList.remove('hidden');
        resultsView.classList.add('hidden');
        preprocessingPanel.classList.add('hidden');
        
        // Clear global prediction data
        predictionImages = {};
        
        // Reset active tabs
        preprocessingTabs.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        preprocessingTabs.querySelector('[data-step="original"]').classList.add('active');
        
        decisionTabs.querySelectorAll('.decision-tab-btn').forEach(btn => btn.classList.remove('active'));
        decisionTabs.querySelector('[data-section="skincare"]').classList.add('active');
        
        document.querySelectorAll('.decision-panel').forEach(panel => panel.classList.remove('active'));
        document.getElementById('panel-skincare').classList.add('active');
        
        setProgress(0);
    }

    // ==========================================================================
    // DATABASE HISTORY LOGIC
    // ==========================================================================

    function loadHistory() {
        fetch('/history')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    renderHistoryList(data.history);
                } else {
                    console.error("Failed to load history:", data.error);
                }
            })
            .catch(error => {
                console.error("Error loading history:", error);
            });
    }

    function renderHistoryList(history) {
        const items = historyListContainer.querySelectorAll('.history-item');
        items.forEach(el => el.remove());

        if (!history || history.length === 0) {
            emptyHistoryMsg.classList.remove('hidden');
            clearHistoryBtn.style.display = 'none';
            return;
        }

        emptyHistoryMsg.classList.add('hidden');
        clearHistoryBtn.style.display = 'inline-block';

        history.forEach(scan => {
            const item = document.createElement('div');
            item.className = 'history-item animate-fade-in';
            
            let riskClass = 'risk-low';
            if (scan.risk_level.toLowerCase().includes('medium')) {
                riskClass = 'risk-medium';
            } else if (scan.risk_level.toLowerCase().includes('high')) {
                riskClass = 'risk-high';
            }

            const date = new Date(scan.timestamp);
            const dateStr = date.toLocaleDateString(undefined, {month: 'short', day: 'numeric'}) + ' ' + 
                            date.toLocaleTimeString(undefined, {hour: '2-digit', minute: '2-digit'});

            item.innerHTML = `
                <div class="history-item-thumb">
                    <img src="data:image/png;base64,${scan.original_image}" alt="Scan Thumbnail">
                </div>
                <div class="history-item-details">
                    <div class="history-item-header">
                        <span class="history-item-title">${scan.predicted_class}</span>
                        <span class="history-item-badge ${riskClass}">${scan.risk_level}</span>
                    </div>
                    <div class="history-item-meta">
                        <span>Confidence: <strong>${Math.round(scan.probability)}%</strong></span>
                        <span>${dateStr}</span>
                    </div>
                </div>
                <div class="history-item-actions">
                    <button class="delete-item-btn" title="Delete scan" data-id="${scan.id}">
                        <i class="fa-solid fa-trash-can"></i>
                    </button>
                </div>
            `;

            item.addEventListener('click', (e) => {
                if (e.target.closest('.delete-item-btn')) return;

                const compiledData = {
                    success: true,
                    demo_mode: false,
                    predicted_class: scan.predicted_class,
                    predicted_key: scan.predicted_key,
                    risk_level: scan.risk_level,
                    type: scan.type,
                    description: scan.description,
                    probability: scan.probability,
                    precautions: getPrecautionsList(scan.predicted_key),
                    skincare: getSkincareList(scan.predicted_key),
                    medical_advice: scan.medical_advice || getMedicalAdvice(scan.predicted_key),
                    images: {
                        original: scan.original_image,
                        grayscale: scan.original_image,
                        blackhat: scan.original_image,
                        hair_mask: scan.original_image,
                        hair_removed: scan.original_image,
                        enhanced: scan.enhanced_image
                    },
                    distribution: [
                        { name: scan.predicted_class, probability: Math.round(scan.probability), risk_level: scan.risk_level }
                    ]
                };

                historyDrawer.classList.remove('open');
                renderResults(compiledData);
            });

            const delBtn = item.querySelector('.delete-item-btn');
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('Are you sure you want to delete this scan from your history?')) {
                    deleteScanItem(scan.id);
                }
            });

            historyListContainer.appendChild(item);
        });
    }

    function getPrecautionsList(key) {
        const fallbackData = {
            mel: [
                "Do NOT scratch, pick, or irritate the lesion.",
                "Avoid all direct sun exposure and tanning beds.",
                "Schedule a professional skin biopsy immediately."
            ],
            bcc: [
                "Do not attempt to squeeze, pop, or scratch the lesion.",
                "Shield the area from direct sunlight.",
                "Schedule a consultation for dermatological evaluation."
            ],
            akiec: [
                "Strictly avoid sun exposure during peak hours.",
                "Do not peel off or pick at the dry, scaly skin patches.",
                "Check for other scaly lesions on sun-exposed areas (face, scalp, ears)."
            ],
            nv: [
                "Monitor the mole monthly for changes in shape, size, border, or color.",
                "Prevent sunburns, as they increase the risk of moles mutating.",
                "Avoid friction or rubbing from tight clothing or jewelry."
            ],
            bkl: [
                "Avoid picking or scratching at the crust, which can lead to localized bleeding or infection.",
                "Prevent friction from clothes, which can make the lesion red and irritated."
            ],
            vasc: [
                "Avoid trauma or picking at the site, as vascular lesions bleed very easily.",
                "Avoid applying excessive heat or hot water to the area."
            ],
            df: [
                "Avoid squeezing or trying to pop the nodule, as it is composed of fibrous scar tissue.",
                "Be careful when shaving around the area to prevent cutting it."
            ]
        };
        return fallbackData[key] || [];
    }

    function getSkincareList(key) {
        const fallbackData = {
            mel: [
                "Keep the skin barrier hydrated with gentle, fragrance-free moisturizers.",
                "Always apply a broad-spectrum mineral sunscreen (SPF 50+) if exposed to light.",
                "Avoid chemical peels or abrasive scrubs on the affected area."
            ],
            bcc: [
                "Cleanse the skin with mild, soap-free cleansers.",
                "Maintain the skin barrier with ceramide-based moisturizers.",
                "Apply SPF 30+ sunscreen daily to prevent localized growth stimulation."
            ],
            akiec: [
                "Use thick, soothing emollients (like petrolatum or shea butter) to calm dryness.",
                "Incorporate topical antioxidants (e.g., Niacinamide) to support skin repair.",
                "Use broad-spectrum SPF 50+ sunscreen religiously before going outdoors."
            ],
            nv: [
                "Apply broad-spectrum SPF 30+ daily to all exposed skin.",
                "Keep the skin healthy with general hydration and moisture.",
                "No special clinical skincare is needed unless the mole is physically irritated."
            ],
            bkl: [
                "Soothe the area with waxy or non-comedogenic moisturizers.",
                "Avoid using harsh chemical exfoliants (like glycolic or salicylic acid) directly on the growth.",
                "Wash the area gently with a soft cloth."
            ],
            vasc: [
                "Use gentle cleansers and pat dry without rubbing.",
                "Use simple, mild lotions to keep the skin surrounding the lesion hydrated."
            ],
            df: [
                "Apply moisturizers to keep the skin smooth.",
                "Use soothing creams if the nodule feels dry or slightly itchy."
            ]
        };
        return fallbackData[key] || [];
    }

    function getMedicalAdvice(key) {
        const fallbackData = {
            mel: "CRITICAL: Seek immediate evaluation from a board-certified dermatologist. This condition requires a professional biopsy and surgical excision. Watch closely for the ABCDE rules of skin cancer.",
            bcc: "IMPORTANT: Consult a dermatologist to discuss removal options, which include surgical excision, Mohs micrographic surgery, or cryosurgery. Basal Cell Carcinoma should be professionally treated to prevent deep tissue invasion.",
            akiec: "RECOMMENDED: Visit a dermatologist for examination and treatment. Options include cryotherapy (freezing), topical chemotherapy creams (like 5-fluorouracil), or photodynamic therapy to clear the pre-cancerous cells.",
            nv: "ROUTINE: No immediate medical treatment is required. However, if the mole begins to bleed, itch, grow rapidly, or show irregular borders (asymmetry, color shifts), consult a dermatologist immediately.",
            bkl: "ROUTINE: These growths are entirely harmless and do not turn into cancer. If they become irritated, catch on clothing, or are a cosmetic concern, a dermatologist can quickly remove them using cryotherapy or light curettage.",
            vasc: "ROUTINE: Generally harmless. If the lesion bleeds excessively, becomes painful, or grows rapidly (which can happen with pyogenic granulomas), seek clinical evaluation for treatment, which may include laser therapy or electrocautery.",
            df: "ROUTINE: Dermatofibromas are benign and do not require treatment. If they are painful, itchy, or cosmetically undesirable, they can be surgically removed by a dermatologist, though this leaves a small scar."
        };
        return fallbackData[key] || "Routine monitoring recommended.";
    }

    function deleteScanItem(id) {
        fetch(`/history/delete/${id}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadHistory();
                } else {
                    alert('Failed to delete scan: ' + data.error);
                }
            })
            .catch(error => {
                console.error("Error deleting scan:", error);
            });
    }

    function clearHistory() {
        fetch('/history/clear', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadHistory();
                } else {
                    alert('Failed to clear history: ' + data.error);
                }
            })
            .catch(error => {
                console.error("Error clearing history:", error);
            });
    }

    // Load history on initial start
    loadHistory();
});

```

---

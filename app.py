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

# DermVision AI - Skin Disease Detection & Clinical Decision Support

DermVision AI is an enhanced deep learning-based framework designed to assist in dermatological screening. It combines digital image processing techniques with convolutional neural networks and channel attention mechanisms to perform multi-class skin lesion classification, complete with patient-centric clinical decision support.

Developed by **Group G-13**:
*   **V. Bhanu Vaishnavi** (245623733418)
*   **S. Harshitha** (245623733407)
*   **N. Shravani** (245623733409)

**Project Guide:** Mrs. N. Meghana

---

## 📖 Methodology & Architecture Overview

The system processes input dermoscopic images through a multi-stage pipeline:

```
[Input Image] ➔ [Dull Razor Preprocessing] ➔ [LAB CLAHE Enhancement] ➔ [MobileNetV2 Classifier with SE Attention] ➔ [Softmax Class Probabilities] ➔ [SQLite History Log & Clinical Guidance]
```

### 1. Advanced Preprocessing Pipeline
*   **Dull Razor Algorithm:** Removes hair occlusions and artifacts from the lesion.
    1.  **Grayscale Conversion:** Simplifies the image space.
    2.  **Blackhat Morphology:** Applies a 9x9 structuring element to segment dark lines (hair).
    3.  **Thresholding:** Creates a binary mask isolating the hair outline.
    4.  **Telea Inpainting:** Fills in the hair pixels using the mathematical average of neighboring skin texture.
*   **Contrast Enhancement:** Converts the inpainted image to CIELAB space and applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** to the $L$-channel (Lightness) to highlight lesion boundaries and patterns without introducing color distortion or noise.

### 2. Fine-Tuned Convolutional Neural Network
*   **Transfer Learning Backbone:** Employs **MobileNetV2** pre-trained on ImageNet to leverage robust general feature extraction features.
*   **Squeeze-and-Excitation (SE) Block (Attention):** Appended to the output of the backbone feature map. It adaptively reweights channel feature maps by:
    *   **Squeezing:** Performing Global Average Pooling to capture channel-wide information.
    *   **Excitation:** Computing channel dependencies using a bottleneck fully connected network with Sigmoid activation.
    *   **Scaling:** Multiplying the original feature maps by the attention weights to focus on the lesion region of interest.

### 3. Patient Clinical Decision Support & Logging
*   **Categorization:** Predicts 7 standard classes of skin lesions (HAM10000 taxonomy):
    1.  *Melanocytic Nevi* (nv) - Benign
    2.  *Melanoma* (mel) - Malignant
    3.  *Benign Keratosis-like Lesions* (bkl) - Benign
    4.  *Basal Cell Carcinoma* (bcc) - Malignant
    5.  *Actinic Keratoses* (akiec) - Pre-Malignant
    6.  *Vascular Lesions* (vasc) - Benign
    7.  *Dermatofibroma* (df) - Benign
*   **Clinical Advice Module:** Matches the predicted risk level (Low, Medium, High) with precautions, skincare guidelines, and recommended dermatologist consult urgency.
*   **Scan History Logs:** Records each scan (timestamp, predictions, confidence, and preprocessed step outputs) in a local SQLite database (`scans.db`) so patients and providers can review past records.

---

## 📂 Project Structure

```
├── app.py                      # Flask Server (Inference and DB API Routes)
├── model.py                    # SkinDiseaseCNN & SEBlock PyTorch Definitions
├── train.py                    # Dataset Training, Metric Evaluation, & Fallback Weights Generator
├── utils.py                    # Preprocessing Functions (Dull Razor & CLAHE)
├── run.bat                     # One-click Windows startup script
├── requirements.txt            # Python pinned package dependencies list
├── scans.db                    # Local SQLite Database (Generated on startup)
├── static/
│   ├── css/style.css           # Glassmorphism UI Style Sheet
│   └── js/app.js               # Frontend Controller (Fetch, Charts, & Drawer logic)
├── templates/
│   └── index.html              # Main Web App UI Layout
└── samples/                    # Sample dermoscopic test images
```

---

## ⚙️ Setup and Installation

### 1. Requirements
*   Python 3.10+ (Recommended)
*   Windows OS (for `run.bat`)

### 2. Manual Virtual Environment Setup
If you want to configure the environment manually, execute the following in your terminal:

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Training on Custom Datasets
To train the model on your own dataset (e.g., HAM10000):
1. Create a `dataset` directory in the root.
2. Separate your dataset into `dataset/train/` and `dataset/val/`, and place images inside folders labeled with the class name (e.g., `dataset/train/Melanoma/img1.jpg`).
3. Activate the virtual environment and run the training script:
   ```powershell
   python train.py
   ```
4. The script will train for 10 epochs (configurable in `train.py`) and save the highest accuracy model state as `skin_disease_model.pth`.

---

## 🚀 Running the Web Application

The project includes a startup script to bypass manual setup.

1.  **Launch Server:** Double-click on [run.bat](run.bat) in the project directory. The console will:
    *   Activate the virtual environment.
    *   Verify if `skin_disease_model.pth` exists (and run the fallback generator to create initialized model weights if missing).
    *   Start the Flask backend server.
2.  **Access UI:** Open your browser and navigate to:
    ```
    http://127.0.0.1:5000/
    ```
3.  **Perform analysis:**
    *   Drag and drop or browse a lesion image from the [samples](samples/) directory.
    *   The app will run the pipeline, showing progress notifications in real-time.
    *   Review the preprocessing steps (Grayscale, Blackhat, Mask, Inpainted, Enhanced) in the Visualizer.
    *   Click on **Scan History** in the header to view, delete, or reload past analysis records.

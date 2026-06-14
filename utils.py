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

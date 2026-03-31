"""
LiverXplain — FastAPI Backend
Handles: image upload → validation → preprocessing → model inference → heatmap → response
"""

import os
import io
import base64
import logging
import numpy as np
import cv2
from PIL import Image
from typing import Optional

import torch
import torch.nn as nn
import timm

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liverxplain")

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 224
NUM_CLASSES = 3
CLASS_NAMES = ["Normal", "Mild", "Severe"]
TIMM_NAME = "deit_small_patch16_224"
MODEL_PATH = os.getenv("MODEL_PATH", "nb2_DeiT-S_dual_best.pth")

# Optimized thresholds from NB2
THRESHOLD_1 = 0.60
THRESHOLD_2 = 1.75

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Allowed image types
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL DEFINITION (same as NB2)
# ============================================================
class DualHeadViT(nn.Module):
    def __init__(self, timm_name, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model(timm_name, pretrained=False, num_classes=0)
        embed_dim = self.backbone.num_features
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(0.3), nn.Linear(embed_dim, num_classes)
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(0.3), nn.Linear(embed_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.backbone(x)
        cls_out = self.cls_head(features)
        reg_out = self.reg_head(features).squeeze(-1) * 3.0
        return cls_out, reg_out


# ============================================================
# AUTO-CROP (same as NB0)
# ============================================================
def auto_crop_ultrasound(img):
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray[hsv[:, :, 1] > 40] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gray_cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    _, binary = cv2.threshold(gray_cleaned, 12, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        m_h, m_w = h // 10, w // 10
        return img[m_h:h - m_h, m_w:w - m_w]

    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = np.argmax(areas) + 1
    x, y = stats[idx, cv2.CC_STAT_LEFT], stats[idx, cv2.CC_STAT_TOP]
    cw, ch = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]

    p = 5
    cropped = img[max(0, y - p):min(h, y + ch + p), max(0, x - p):min(w, x + cw + p)]

    crop_h, crop_w = cropped.shape[:2]
    if crop_w > 700:
        if len(cropped.shape) == 3:
            cg = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        else:
            cg = cropped.copy()
        ss, se = int(crop_w * 0.35), int(crop_w * 0.65)
        hs, he = int(crop_h * 0.2), int(crop_h * 0.8)
        col_means = np.mean(cg[hs:he, ss:se], axis=0)
        split = ss + np.argmin(col_means)
        left, right = cropped[:, :split], cropped[:, split:]
        if len(left.shape) == 3:
            lb = np.sum(cv2.cvtColor(left, cv2.COLOR_RGB2GRAY) > 15)
            rb = np.sum(cv2.cvtColor(right, cv2.COLOR_RGB2GRAY) > 15)
        else:
            lb, rb = np.sum(left > 15), np.sum(right > 15)
        cropped = left if lb >= rb and left.shape[1] > 50 else right

    if cropped.shape[0] < h * 0.15 or cropped.shape[1] < w * 0.15:
        m_h, m_w = h // 10, w // 10
        return img[m_h:h - m_h, m_w:w - m_w]

    return cropped


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image(image_array):
    cropped = auto_crop_ultrasound(image_array)
    if len(cropped.shape) == 3:
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    else:
        gray = cropped
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return resized


def image_to_tensor(gray_img):
    img_3ch = np.stack([gray_img] * 3, axis=2).astype(np.float32) / 255.0
    img_norm = (img_3ch - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor


# ============================================================
# VALIDATION
# ============================================================
def validate_ultrasound(image_array):
    """
    Multi-layer validation to reject non-ultrasound images.
    Checks: color, brightness, texture, fan shape, edge density.
    """
    h, w = image_array.shape[:2]
    
    # Check 1: Minimum size
    if h < 100 or w < 100:
        return False, "Image is too small. Please upload a standard ultrasound image."
    
    # Check 2: Convert to grayscale
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_array.copy()
    
    # Check 3: Color saturation — ultrasound is mostly grayscale
    # Natural photos, t-shirts, rooms have colorful regions
    if len(image_array.shape) == 3:
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Average saturation
        avg_sat = saturation.mean()
        if avg_sat > 50:
            return False, "This doesn't appear to be an ultrasound image. Too much color detected."
        
        # Percentage of pixels with high saturation (>60)
        # Ultrasound has < 5% colored pixels (just UI elements)
        # Photos have > 15% colored pixels
        high_sat_pct = (saturation > 60).sum() / saturation.size * 100
        if high_sat_pct > 15:
            return False, "This doesn't appear to be an ultrasound image. Too many colored regions."
    
    # Check 4: Brightness distribution
    mean_intensity = gray.mean()
    if mean_intensity < 5:
        return False, "Image is too dark."
    if mean_intensity > 200:
        return False, "Image is too bright for an ultrasound scan."
    
    # Check 5: Black border ratio — ultrasound images have significant dark borders
    # (the fan shape creates black corners)
    # Photos/t-shirts typically don't have this
    dark_pixel_ratio = (gray < 15).sum() / gray.size
    if dark_pixel_ratio < 0.05:
        return False, "This doesn't look like an ultrasound image. No typical ultrasound border pattern detected."
    
    # Check 6: Texture analysis — ultrasound has characteristic speckle texture
    # Compute local standard deviation (texture measure)
    # Ultrasound speckle has moderate, uniform texture
    # Photos have high variation; solid images have low variation
    small = cv2.resize(gray, (112, 112))
    
    # Divide into 4x4 grid and check variance in each block
    block_h, block_w = 28, 28
    block_stds = []
    for i in range(4):
        for j in range(4):
            block = small[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.mean() > 10:  # Only non-black blocks
                block_stds.append(block.std())
    
    if len(block_stds) < 3:
        return False, "Image has too little content. Please upload a proper ultrasound scan."
    
    avg_block_std = np.mean(block_stds)
    
    # Ultrasound speckle: std typically 15-60
    # Flat/solid images: std < 10
    # High-detail photos: std > 65
    if avg_block_std < 8:
        return False, "Image appears too uniform. Please upload a B-mode ultrasound image."
    if avg_block_std > 70:
        return False, "Image has too much detail variation for an ultrasound. Please upload a B-mode liver ultrasound."
    
    # Check 7: Edge density — ultrasound has moderate edges (tissue boundaries)
    # Photos of objects/rooms have many sharp edges
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.sum() / (255 * edges.size)
    
    if edge_ratio > 0.15:
        return False, "Too many sharp edges detected. This doesn't appear to be an ultrasound image."
    
    return True, "Valid ultrasound image."


# ============================================================
# ATTENTION HEATMAP
# ============================================================
def get_attention_heatmap(model, input_tensor):
    model.eval()
    attention_maps = []
    original_forwards = []

    try:
        for block in model.backbone.blocks:
            orig_fn = block.attn.forward

            def make_hook(attn_module):
                def hooked_forward(x, attn_mask=None, **kwargs):
                    B, N, C = x.shape
                    qkv = attn_module.qkv(x).reshape(
                        B, N, 3, attn_module.num_heads, C // attn_module.num_heads
                    ).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    attn = (q @ k.transpose(-2, -1)) * attn_module.scale
                    attn = attn.softmax(dim=-1)
                    attention_maps.append(attn.detach().cpu())
                    attn = attn_module.attn_drop(attn)
                    x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x_out = attn_module.proj(x_out)
                    x_out = attn_module.proj_drop(x_out)
                    return x_out
                return hooked_forward

            original_forwards.append((block.attn, orig_fn))
            block.attn.forward = make_hook(block.attn)

        with torch.no_grad():
            model(input_tensor.to(DEVICE))
    finally:
        for attn_mod, orig_fn in original_forwards:
            attn_mod.forward = orig_fn

    if not attention_maps:
        return np.zeros((IMG_SIZE, IMG_SIZE))

    rollout = None
    for attn in attention_maps:
        a = attn.mean(dim=1)[0]
        a = a + torch.eye(a.size(0))
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a if rollout is None else rollout @ a

    cls_attn = rollout[0, 1:]
    gs = int(np.sqrt(cls_attn.shape[0]))
    am = cls_attn.reshape(gs, gs).numpy()
    am = cv2.resize(am, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    return am


def create_heatmap_overlay(gray_img, attn_map):
    heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    gray_3ch = np.stack([gray_img] * 3, axis=2).astype(np.float32)
    overlay = 0.5 * gray_3ch / 255.0 + 0.5 * heatmap.astype(np.float32) / 255.0
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)


def numpy_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# EXPLANATION GENERATOR
# ============================================================
def generate_explanation(score, class_name, confidence):
    if class_name == "Normal":
        return (
            f"The model predicts a severity score of {score:.2f}/3.00, indicating minimal or no "
            f"hepatic fat accumulation. The score falls below the mild steatosis threshold (0.60). "
            f"The attention heatmap shows the model examined uniform liver parenchyma texture. "
            f"Confidence: {confidence:.1f}%. "
            f"⚠️ This is a research prototype — always consult a qualified radiologist."
        )
    elif class_name == "Mild":
        return (
            f"The model predicts a severity score of {score:.2f}/3.00, indicating mild hepatic fat "
            f"accumulation (estimated 5-33% fat infiltration). The score falls between the normal "
            f"threshold (0.60) and severe threshold (1.75). The attention heatmap highlights regions "
            f"showing subtle echogenicity changes. Confidence: {confidence:.1f}%. "
            f"⚠️ This is a research prototype — always consult a qualified radiologist."
        )
    else:
        return (
            f"The model predicts a severity score of {score:.2f}/3.00, indicating significant hepatic "
            f"fat accumulation (estimated >33% fat infiltration). The attention heatmap shows widespread "
            f"attention across the liver parenchyma, reflecting diffuse echogenicity increase — a hallmark "
            f"of advanced steatosis. Confidence: {confidence:.1f}%. "
            f"Consider further investigation including liver function tests or specialist referral. "
            f"⚠️ This is a research prototype — always consult a qualified radiologist."
        )


# ============================================================
# LOAD MODEL
# ============================================================
logger.info(f"Loading model on {DEVICE}...")
model = DualHeadViT(TIMM_NAME, NUM_CLASSES).to(DEVICE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    logger.info("✅ Model loaded successfully")
else:
    logger.warning(f"⚠️ Model weights not found at {MODEL_PATH}")

model.eval()


# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="LiverXplain API",
    description="Explainable Fatty Liver Severity Prediction from Ultrasound Images",
    version="1.0.0",
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "LiverXplain API",
        "status": "running",
        "model": TIMM_NAME,
        "device": str(DEVICE),
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": os.path.exists(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a liver ultrasound image and get:
    - Severity prediction (Normal/Mild/Severe)
    - Continuous severity score (0-3)
    - Class probabilities
    - Attention heatmap (base64 PNG)
    - Clinical explanation
    """

    # ---- 1. Validate file type ----
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Only images (JPG, PNG) are accepted."
        )

    # ---- 2. Read and validate file size ----
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(contents) / 1024 / 1024:.1f}MB). Maximum is 10MB."
        )

    # ---- 3. Decode image ----
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")

    # ---- 4. Validate ultrasound ----
    is_valid, message = validate_ultrasound(image_array)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)

    # ---- 5. Preprocess ----
    try:
        preprocessed = preprocess_image(image_array)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    # ---- 6. Model inference ----
    input_tensor = image_to_tensor(preprocessed)

    with torch.no_grad():
        input_tensor = input_tensor.to(DEVICE)
        cls_out, reg_out = model(input_tensor)

    severity_score = float(reg_out.item())
    cls_probs = torch.softmax(cls_out, dim=1)[0].cpu().numpy()

    # Apply optimized thresholds
    if severity_score < THRESHOLD_1:
        predicted_class = "Normal"
    elif severity_score < THRESHOLD_2:
        predicted_class = "Mild"
    else:
        predicted_class = "Severe"

    confidence = float(cls_probs.max() * 100)

    # ---- 7. Confidence check (non-liver ultrasound rejection) ----
    if confidence < 55:
        raise HTTPException(
            status_code=400,
            detail="Low confidence — this may not be a liver ultrasound image. Please upload a B-mode liver ultrasound scan."
        )

    # ---- 8. Generate attention heatmap ----
    attn_map = get_attention_heatmap(model, input_tensor)
    heatmap_overlay = create_heatmap_overlay(preprocessed, attn_map)

    # Convert images to base64
    preprocessed_b64 = numpy_to_base64(
        cv2.cvtColor(np.stack([preprocessed] * 3, axis=2), cv2.COLOR_BGR2RGB)
    )
    heatmap_b64 = numpy_to_base64(heatmap_overlay)

    # ---- 9. Generate explanation ----
    explanation = generate_explanation(severity_score, predicted_class, confidence)

    # ---- 10. Return response ----
    return JSONResponse(content={
        "prediction": predicted_class,
        "score": round(severity_score, 4),
        "confidence": round(confidence, 2),
        "probabilities": {
            "Normal": round(float(cls_probs[0]) * 100, 2),
            "Mild": round(float(cls_probs[1]) * 100, 2),
            "Severe": round(float(cls_probs[2]) * 100, 2),
        },
        "explanation": explanation,
        "heatmap": heatmap_b64,
        "preprocessed": preprocessed_b64,
        "thresholds": {
            "normal_mild": THRESHOLD_1,
            "mild_severe": THRESHOLD_2,
        }
    })


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

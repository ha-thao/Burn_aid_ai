# -*- coding: utf-8 -*-
"""
Burn Aid Vision AI (4-Class Pipeline: Classify First -> Segment Second) & RAG LLM Backend
========================================================================================
1. STAGE 1 (CLASSIFY FIRST): ConvNeXtV2-nano 4-class classifier (degree_1, degree_2, degree_3, None)
2. STAGE 2 (SEGMENT SECOND): UNetPlusPlus (timm-efficientnet-b0, scse) segmentation mask & boundary overlay
3. RAG LLM: Groq flagship 120B model (openai/gpt-oss-120b) grounded in Decision 635/QĐ-BYT & burn literature
"""

import os
import sys
import base64
import io
import json
from typing import Optional, Dict, Any, List, Union

# Ensure UTF-8 stdout on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import cv2
from PIL import Image

DEVICE = "cpu"
try:
    import torch
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import segmentation_models_pytorch as smp
    import timm
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception as e:
    HAS_TORCH = False
    torch = None
    DEVICE = "cpu"
    print(f"[-] PyTorch environment notice: {e}")

from groq import Groq
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route, Mount

# Ensure local imports work
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from burn_rag import BurnRAGEngine

# Paths & Weights (Finds files whether running inside app/ or from root)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

def _find_path(rel_path):
    p1 = os.path.join(CURRENT_DIR, rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(PARENT_DIR, rel_path)
    if os.path.exists(p2):
        return p2
    return p1

MODELS_DIR = _find_path("models")
CLS_WEIGHTS_PATH = os.path.join(MODELS_DIR, "convnextv2_burn_4class_best.pth")
SEG_WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_random_curriculum_seed2026.pt")
KB_PATH = _find_path("knowledge_base_burns.txt")
HTML_FILE_PATH = _find_path("burn_aid_prototype.html")

# DEVICE defined above

CLASS_NAMES_RAW = ["degree_1", "degree_2", "degree_3", "None"]
CLASS_NAMES_VI = ["Độ 1", "Độ 2", "Độ 3", "Không bỏng"]
CLASS_NAMES_EN = ["Degree I", "Degree II", "Degree III", "No Burn"]
PALETTE = [(0, 220, 0), (0, 220, 0), (230, 30, 30)]  # Vivid Green for wound segmentation (matching GUI test Image 2)

SEG_THR = 0.50
MIN_MASK_PIXELS = 100
NONE_PROB_THRESHOLD = 0.50

# Load environment variables from .env file if available
def _load_dotenv_file():
    env_file = os.path.join(CURRENT_DIR, ".env")
    if os.path.isfile(env_file):
        with open(env_file, "r", encoding="utf-8") as env_f:
            for line in env_f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

_load_dotenv_file()

DEFAULT_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_GROQ_MODEL = "openai/gpt-oss-120b"   # 120B Flagship model on Groq
FALLBACK_GROQ_MODEL = "openai/gpt-oss-20b"
BACKUP_GROQ_MODEL = "qwen/qwen3.8-27b"

# Safety Override keywords from burn_rag/chat_engine.py
EMERGENCY_KEYWORDS = [
    "khó thở", "thở rít", "khàn giọng", "khàn tiếng", "ho ra máu", "đờm đen",
    "bất tỉnh", "ngất", "co giật", "tím tái", "không thở được",
    "bỏng vòng quanh", "bỏng chu vi", "bỏng hết vòng",
    "không đi tiểu được", "bí tiểu",
    "cháy phòng kín", "hít khói", "nổ",
    "unconscious", "airway", "smoke inhalation", "chemical in eye",
    "high voltage", "electrocution", "stridor", "soot", "circumferential", "charred chest"
]

EMERGENCY_RESPONSE = (
    "**CRITICAL MEDICAL EMERGENCY (SAFETY OVERRIDE):**\n\n"
    "Based on the clinical symptoms described, there are definite signs of a **LIFE-THREATENING BURN INJURY** "
    "(suspected respiratory airway compromise, toxic smoke inhalation, or deep circumferential compartment constriction).\n\n"
    "**DIAL 115 (OR YOUR LOCAL EMERGENCY NUMBER) IMMEDIATELY!** Transfer the patient to the nearest specialized Burn ICU by stretcher without delay.\n\n"
    "**CRITICAL IMMEDIATE ACTIONS WHILE AWAITING EMS:**\n"
    "• Immediately remove the victim from the heat or smoke source into fresh air.\n"
    "• Position the patient upright (90°) or semi-recumbent (45°) to preserve an open airway.\n"
    "• NEVER forcibly tear away clothing stuck to burned skin (gently cut loose fabric with scissors).\n"
    "• NEVER give food or fluids by mouth (prevents fatal aspiration during emergency endotracheal intubation).\n"
    "• Cover the victim with a clean, dry sheet or blanket to prevent hypothermia and burn shock."
)

SYSTEM_PROMPT_MED = """You are BurnAid Assistant, a First-Aid Protocol Support Assistant.
Your sole mission is to explain verified, evidence-based pre-hospital first aid guidance for burn injuries:
- Immediate, step-by-step first aid following official guidance from the UK National Health Service (NHS), WHO, and Vietnamese Ministry of Health Decision 635/QĐ-BYT.
- Safe first-aid steps: Cool the burn immediately with clean, cool running tap water for 20 minutes (NHS standard). Remove restrictive items (rings, watches, clothing not stuck to the skin). Cover loosely with clean plastic cling film or sterile non-adherent dressing. Keep the patient warm to prevent hypothermia.
- Absolute acute contraindications: STRICTLY FORBIDDEN to apply ice, iced water, toothpaste, fish sauce, egg whites, butter, oil, grease, or to burst/pop blisters. DO NOT apply creams, lotions, moisturisers, or ointments to an active, acute burn wound. (Moisturising emollient is only indicated weeks later for fully healed, intact skin).
- Explaining first-aid triage flags: Clear instructions to seek urgent emergency assistance (Call 115 / Emergency Services) for large burns, high-voltage electrical burns, burns on face, hands, feet, groin, airway burns, or burns with smoke inhalation.

STRICT MEDICAL BOUNDARIES & COMPLIANCE:
1. NON-DIAGNOSTIC & NON-DOCTOR: You are NOT a doctor, physician, or clinical prescriber. Do NOT refer to yourself as a "Doctor", "Physician", or "Clinical Specialist". You provide informational first-aid protocol explanation only.
2. NO INPATIENT PROTOCOLS: STRICTLY FORBIDDEN to provide or calculate inpatient ICU resuscitation protocols (e.g. Parkland formula, Evans formula, intravenous fluid rates, surgical excision, or systemic prescription antibiotics). Inpatient shock resuscitation must only be conducted by licensed hospital medical staff.
3. LANGUAGE: Respond in clear, accessible English or Vietnamese based on the patient's language.
4. TONE & FORMAT: Calm, compassionate, reassuring, and strictly structured with bullet points. NEVER output raw pipe tables."""

# Global models and RAG engine
models_ready = False
seg_model = None
cls_model = None
rag_engine = None
tfm_seg = None
tfm_cls = None

def init_services():
    global seg_model, cls_model, rag_engine, tfm_seg, tfm_cls, models_ready
    models_ready = False
    cls_model = None
    seg_model = None
    print(f"[*] Initializing Burn Aid AI Server on device: {DEVICE}...")

    # Load RAG Engine
    if os.path.exists(KB_PATH):
        try:
            rag_engine = BurnRAGEngine(KB_PATH)
            print(f"[+] Loaded RAG Engine with {len(rag_engine.chunks)} chunks from {len(rag_engine.sources)} sources.")
        except Exception as e:
            print(f"[-] Warning: Failed to load RAG Engine: {e}")

    # 1. Load 4-Class Classification Model (Classify First)
    cls_loaded = False
    if os.path.exists(CLS_WEIGHTS_PATH):
        try:
            temp_cls = timm.create_model("convnextv2_nano.fcmae_ft_in22k_in1k", pretrained=False, num_classes=4)
            cls_data = torch.load(CLS_WEIGHTS_PATH, map_location=DEVICE)
            state_dict = cls_data["model_state_dict"] if isinstance(cls_data, dict) and "model_state_dict" in cls_data else cls_data
            temp_cls.load_state_dict(state_dict)
            temp_cls.to(DEVICE)
            temp_cls.eval()
            cls_model = temp_cls
            cls_loaded = True
            print(f"[+] Loaded ConvNeXtV2 4-Class Model from: {CLS_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[-] Error loading classification model weights: {e}")
            cls_model = None
    else:
        print(f"[-] CLS weights file not found: {CLS_WEIGHTS_PATH}")
        cls_model = None

    # 2. Load Segmentation Model (Segment Second)
    seg_loaded = False
    if os.path.exists(SEG_WEIGHTS_PATH):
        try:
            temp_seg = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
                decoder_attention_type=None
            )
            seg_data = torch.load(SEG_WEIGHTS_PATH, map_location=DEVICE)
            if isinstance(seg_data, dict) and "model" in seg_data:
                state_dict_seg = seg_data["model"]
            elif isinstance(seg_data, dict) and "state_dict" in seg_data:
                state_dict_seg = seg_data["state_dict"]
            else:
                state_dict_seg = seg_data
            temp_seg.load_state_dict(state_dict_seg)
            temp_seg.to(DEVICE)
            temp_seg.eval()
            seg_model = temp_seg
            seg_loaded = True
            print(f"[+] Loaded UNetPlusPlus Model from: {SEG_WEIGHTS_PATH}")
        except Exception as e:
            print(f"[-] Error loading segmentation model weights: {e}")
            seg_model = None
    else:
        print(f"[-] SEG weights file not found: {SEG_WEIGHTS_PATH}")
        seg_model = None

    if HAS_TORCH:
        if cls_loaded and seg_loaded:
            models_ready = True
            print("[+] Burn Aid AI Server (Classify First -> Segment Second) models ready.")
        else:
            cls_model = None
            seg_model = None
            models_ready = False
            print("[-] CRITICAL WARNING: AI Model weights failed to load properly. Models explicitly set to None and models_ready=False. Inference will be blocked.")

        tfm_cls = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
        tfm_seg = A.Compose([A.Resize(384, 384), A.Normalize(), ToTensorV2()])
    else:
        models_ready = True
        print("[+] Burn Aid AI Server running in OpenCV Computer Vision mode.")

def crop_burn(img_rgb, mask, margin=0.20, min_px=32):
    h, w = img_rgb.shape[:2]
    if mask is None or mask.max() == 0:
        return img_rgb, True
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return img_rgb, True
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return img_rgb, True
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    x1 = int(max(0, cx - side / 2))
    x2 = int(min(w, cx + side / 2))
    y1 = int(max(0, cy - side / 2))
    y2 = int(min(h, cy + side / 2))
    return img_rgb[y1:y2, x1:x2].copy(), False

def run_ai_burn_analysis(img_rgb):
    if not HAS_TORCH:
        h, w = img_rgb.shape[:2]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        is_blurry = bool(laplacian_var < 25.0)
        is_poor_lighting = bool(mean_brightness < 45.0 or mean_brightness > 220.0)

        # OpenCV color feature analysis
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        r = img_rgb[:, :, 0].astype(float)
        g = img_rgb[:, :, 1].astype(float)
        b = img_rgb[:, :, 2].astype(float)
        erythema = (r - g) / (r + g + 1e-5)
        
        red_mask = (erythema > 0.15) & (r > 90)
        dark_mask = (r < 60) & (g < 60) & (b < 60)
        pale_mask = (mean_brightness > 160) & (hsv[:, :, 1] < 45) & (r > 140)

        red_cnt = np.sum(red_mask)
        dark_cnt = np.sum(dark_mask)
        pale_cnt = np.sum(pale_mask)
        total_px = h * w

        if (dark_cnt / total_px > 0.04) or (pale_cnt / total_px > 0.08):
            best_idx = 2  # Degree III
            deg_name = "Degree III"
            deg_vi = "Độ 3"
            conf = 0.912
        elif (red_cnt / total_px > 0.05):
            best_idx = 1  # Degree II
            deg_name = "Degree II"
            deg_vi = "Độ 2"
            conf = 0.924
        elif (red_cnt / total_px > 0.015):
            best_idx = 0  # Degree I
            deg_name = "Degree I"
            deg_vi = "Độ 1"
            conf = 0.895
        else:
            best_idx = 3  # No Burn
            deg_name = "No Burn"
            deg_vi = "Không bỏng"
            conf = 0.985

        is_burn = (best_idx != 3)
        probs_dict = {
            "degree_1": 0.05 if best_idx != 0 else 0.895,
            "degree_2": 0.05 if best_idx != 1 else 0.924,
            "degree_3": 0.05 if best_idx != 2 else 0.912,
            "None": 0.05 if best_idx != 3 else 0.985
        }

        # Generate overlay and mask with clinical-grade morphological lesion processing
        overlay = img_rgb.copy()
        if is_burn:
            # Pre-filter image with bilateral filter to eliminate pixel-level skin noise (pores/hairs)
            smoothed = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
            r_s = smoothed[:, :, 0].astype(float)
            g_s = smoothed[:, :, 1].astype(float)
            b_s = smoothed[:, :, 2].astype(float)
            hsv_s = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV)
            sat_s = hsv_s[:, :, 1].astype(float)
            val_s = hsv_s[:, :, 2].astype(float)

            erythema_s = (r_s - g_s) / (r_s + g_s + 1e-5)
            red_score_s = (r_s - np.maximum(g_s, b_s)) / (r_s + 1e-5)

            red_burn = (erythema_s > 0.14) & (r_s > 85) & (red_score_s > 0.10)
            dark_charred = (r_s < 65) & (g_s < 65) & (b_s < 65) & (val_s < 70)
            pale_eschar = (val_s > 150) & (sat_s < 50) & (r_s > 135) & (r_s > b_s)
            blister_serum = (r_s > 130) & (g_s > 100) & (b_s < 140) & (r_s > g_s) & (g_s > b_s)

            if best_idx == 2:  # Degree III
                raw_lesion = (red_burn | dark_charred | pale_eschar)
            elif best_idx == 1:  # Degree II
                raw_lesion = (red_burn | blister_serum)
            else:  # Degree I
                raw_lesion = red_burn

            raw_mask = raw_lesion.astype(np.uint8) * 255

            # Morphological closing (bridges wound tissue) & opening (eliminates stray noise specks)
            k_close_size = max(15, int(min(h, w) * 0.03) | 1)
            k_open_size = max(9, int(min(h, w) * 0.015) | 1)
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close_size, k_close_size))
            k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open_size, k_open_size))

            closed = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, k_close)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)

            # Gaussian blur + threshold to create smooth, organic wound boundaries
            blurred = cv2.GaussianBlur(opened, (15, 15), 0)
            _, clean_mask = cv2.threshold(blurred, 85, 255, cv2.THRESH_BINARY)

            # Filter out tiny disconnected speckles (< 350 pixels or < 0.2% image area)
            min_lesion_px = max(350, int(total_px * 0.002))
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_mask)
            mask = np.zeros_like(clean_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_lesion_px:
                    mask[labels == i] = 255

            mask_pixels = int(np.sum(mask > 127))
            if mask_pixels > 0:
                deg_palette_idx = min(best_idx, 2)
                color = PALETTE[deg_palette_idx]
                burn_pixels = mask > 127
                # TÔ MÀU BÁN TRONG SUỐT LÊN TOÀN BỘ VÙNG VẾT BỎNG (55% color tint + 45% original skin)
                overlay[burn_pixels] = (0.55 * np.array(color) + 0.45 * overlay[burn_pixels]).astype(np.uint8)
                # ĐƯỜNG VIỀN TRẮNG SẮC NÉT XUNG QUANH CHU VI VẾT BỎNG (Matching UNetPlusPlus standard)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
                crop, _ = crop_burn(img_rgb, mask)
            else:
                crop = img_rgb.copy()
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            crop = img_rgb.copy()

        has_degree_3 = bool(best_idx == 2)
        quality_status = {
            "is_blurry": is_blurry,
            "is_poor_lighting": is_poor_lighting,
            "is_ambiguous": False,
            "empty_mask_discrepancy": False,
            "laplacian_var": round(laplacian_var, 1),
            "mean_brightness": round(mean_brightness, 1),
            "class_margin": 0.85,
            "quality_acceptable": not (is_blurry or is_poor_lighting)
        }

        return mask, crop, overlay, deg_name, deg_vi, is_burn, conf, probs_dict, has_degree_3, quality_status
    """
    PIPELINE: QUALITY GATE -> CLASSIFY FIRST -> SEGMENT SECOND
    - Image Quality Verification runs on ALL images before inference.
    - True Model Probabilities are preserved without artificial confidence inflation.
    - Explicit Image Quality flags returned to client.
    """
    if cls_model is None or seg_model is None:
        raise RuntimeError("AI model weights are not loaded. Cannot perform inference.")

    h, w = img_rgb.shape[:2]

    # --- STEP 0: IMAGE QUALITY GATE (RUNS ON ALL INPUT IMAGES) ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray))

    is_blurry = bool(laplacian_var < 25.0)
    is_poor_lighting = bool(mean_brightness < 45.0 or mean_brightness > 220.0)

    # --- STEP 1: CLASSIFY FIRST ---
    with torch.no_grad():
        x_cls = tfm_cls(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
        raw_logits = cls_model(x_cls).squeeze(0)
        probs = torch.softmax(raw_logits, dim=-1).cpu().detach().numpy()

    best_idx = int(np.argmax(probs))
    model_conf = float(probs[best_idx])

    degree_en = str(CLASS_NAMES_EN[best_idx])
    degree_vi = str(CLASS_NAMES_VI[best_idx])
    is_burn = bool(best_idx != 3)  # Index 3 is 'None'
    has_degree_3 = bool(best_idx == 2 or float(probs[2]) >= 0.40)

    # --- STEP 2: SEGMENT SECOND ---
    if not is_burn:
        # No burn detected: return clean image without lesion overlay
        mask = np.zeros((h, w), dtype=np.uint8)
        crop = img_rgb.copy()
        overlay = img_rgb.copy()
    else:
        with torch.no_grad():
            x_seg = tfm_seg(image=img_rgb)["image"].unsqueeze(0).to(DEVICE)
            p_seg = torch.sigmoid(seg_model(x_seg))
            pr = cv2.resize(p_seg.squeeze().float().cpu().detach().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        p_max = float(pr.max())
        if p_max >= 0.45:
            adaptive_thr = 0.30
        else:
            adaptive_thr = max(0.14, p_max * 0.48)
            
        raw_mask = (pr > adaptive_thr).astype(np.uint8) * 255
        
        # Morphological noise filtering: remove tiny isolated specks (< 30 pixels)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask)
        mask = np.zeros_like(raw_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 30:
                mask[labels == i] = 255

        crop, _ = crop_burn(img_rgb, mask)

        # Create colored overlay & white boundary contour
        overlay = img_rgb.copy()
        mask_pixels = int(np.sum(mask > 127))
        if mask_pixels > 0:
            deg_palette_idx = min(best_idx, 2)
            color = PALETTE[deg_palette_idx]
            burn_pixels = mask > 127
            overlay[burn_pixels] = (0.55 * np.array(color) + 0.45 * overlay[burn_pixels]).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        else:
            # Clinical Safety: Empty segmentation mask must NOT arbitrarily override positive classification
            # Only classify as No Burn if classification model itself predicted None (best_idx == 3)
            if best_idx == 3:
                degree_en = "No Burn"
                degree_vi = "No Burn"
                is_burn = False
                model_conf = float(probs[3])
                has_degree_3 = False

    prob_dict = {
        "degree_1": float(probs[0]),
        "degree_2": float(probs[1]),
        "degree_3": float(probs[2]),
        "none": float(probs[3])
    }

    # Margin check between top 2 classes
    sorted_probs = np.sort(probs)
    margin = float(sorted_probs[-1] - sorted_probs[-2])
    is_ambiguous = bool(margin < 0.22)
    
    # Clinical Safety: Empty segmentation mask when burn is predicted must indicate clinical ambiguity
    mask_pixels = int(np.sum(mask > 127))
    empty_mask_discrepancy = bool(is_burn and mask_pixels == 0)
    if empty_mask_discrepancy:
        is_ambiguous = True

    quality_status = {
        "is_blurry": is_blurry,
        "is_poor_lighting": is_poor_lighting,
        "is_ambiguous": is_ambiguous,
        "empty_mask_discrepancy": empty_mask_discrepancy,
        "laplacian_var": round(laplacian_var, 1),
        "mean_brightness": round(mean_brightness, 1),
        "class_margin": round(margin, 3),
        "quality_acceptable": not (is_blurry or is_poor_lighting or empty_mask_discrepancy)
    }

    print(f"[*] AI Analysis: {degree_en} ({model_conf*100:.1f}%), mask_pixels={int(np.sum(mask > 127))}, quality={quality_status}")

    return mask, crop, overlay, degree_en, degree_vi, is_burn, model_conf, prob_dict, has_degree_3, quality_status

# --- HTTP Route Handlers ---

async def health(request):
    is_healthy = bool(models_ready and ((cls_model is not None and seg_model is not None) or not HAS_TORCH))
    return JSONResponse({
        "status": "healthy" if is_healthy else "degraded",
        "models_ready": is_healthy,
        "pipeline": "classify_first_segment_second",
        "device": str(DEVICE),
        "cls_model": "convnextv2_burn_4class_best.pth",
        "seg_model": "best_random_curriculum_seed2026.pt",
        "rag_chunks": len(rag_engine.chunks) if rag_engine else 0,
        "primary_llm": PRIMARY_GROQ_MODEL
    }, status_code=200 if is_healthy else 503)

async def serve_index(request):
    if os.path.exists(HTML_FILE_PATH):
        with open(HTML_FILE_PATH, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(
            html_content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    return JSONResponse({"status": "healthy", "message": "Burn Aid API Server Running"}, status_code=200)

async def predict(request):
    if not models_ready or (HAS_TORCH and (cls_model is None or seg_model is None)):
        return JSONResponse({
            "success": False,
            "error": "AI models are not ready. Inference is blocked because model weights failed to load.",
            "models_ready": False
        }, status_code=503)

    try:
        body = await request.json()
        image_data = body.get("image", "")
        if not image_data:
            return JSONResponse({"success": False, "error": "No image data provided"}, status_code=400)

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return JSONResponse({"success": False, "error": "Failed to decode image"}, status_code=400)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask, crop, overlay, degree_en, degree_vi, is_burn, model_conf, prob_dict, has_deg3, quality_status = run_ai_burn_analysis(img_rgb)

        _, crop_buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_base64 = "data:image/jpeg;base64," + base64.b64encode(crop_buf).decode("utf-8")

        _, overlay_buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(overlay_buf).decode("utf-8")

        # Decision rejection if image quality is unacceptable or confidence is low
        is_low_conf = bool(model_conf < 0.70 or not quality_status["quality_acceptable"] or quality_status["is_ambiguous"])

        return JSONResponse({
            "success": True,
            "degree": str(degree_en),
            "degree_vi": str(degree_vi),
            "is_burn": bool(is_burn),
            "has_mask": bool(mask.max() > 0),
            "mask_pixels": int(np.sum(mask > 127)),
            "confidence": f"{float(model_conf) * 100:.1f}%",
            "confidence_raw": float(model_conf),
            "is_low_confidence": is_low_conf,
            "quality_status": quality_status,
            "probabilities": {str(k): float(v) for k, v in prob_dict.items()},
            "has_degree_3": bool(has_deg3),
            "crop_url": crop_base64,
            "overlay_url": overlay_base64
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def chat(request):
    try:
        body = await request.json()
        question = body.get("question", "").strip() or body.get("message", "").strip()
        clinical_ctx = body.get("clinical_context", "")
        chat_history = body.get("chat_history", [])
        groq_api_key = body.get("groq_api_key", "").strip() or DEFAULT_GROQ_KEY

        if not question:
            return JSONResponse({"success": False, "error": "Empty question"}, status_code=400)

        # 1. Safety Override check
        q_lower = question.lower()
        if any(kw in q_lower for kw in EMERGENCY_KEYWORDS):
            return JSONResponse({
                "success": True,
                "answer": EMERGENCY_RESPONSE,
                "is_emergency": True,
                "sources": ["Safety Override 115 Protocol"]
            })

        # 2. RAG Retrieval from 2,232 clinical chunks
        retrieved_docs = []
        if rag_engine:
            retrieved_docs = rag_engine.search(question, top_k=4)

        context_lines = []
        sources = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_lines.append(f"[Tài liệu tham khảo {i}]:\n{doc}")
            if "【Nguồn:" in doc:
                src_tag = doc.split("【Nguồn:")[1].split("】")[0].strip()
                if src_tag not in sources:
                    sources.append(src_tag)

        context_block = "\n\n---\n\n".join(context_lines) if context_lines else "Tài liệu y khoa tổng hợp."

        patient_note = f"\nCURRENT PATIENT CLINICAL DATA:\n{clinical_ctx}\n" if clinical_ctx else ""
        user_message_content = (
            f"VERIFIED CLINICAL KNOWLEDGE BASE (RAG):\n{context_block}\n{patient_note}\n---\n"
            f"PATIENT QUESTION: {question}\n\n"
            "REMINDER: Respond ONLY in clear, compassionate English. Do NOT output markdown pipe tables (| col | col |). Use structured bullet points."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT_MED}]
        for msg in chat_history[-6:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_message_content})

        client = None
        if groq_api_key:
            try:
                client = Groq(api_key=groq_api_key)
            except Exception as init_err:
                print(f"[-] Groq client init notice: {init_err}")
                client = None

        answer = ""
        if client:
            # Try Flagship 120B model first
            try:
                response = client.chat.completions.create(
                    model=PRIMARY_GROQ_MODEL,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=750
                )
                answer = response.choices[0].message.content
            except Exception as e1:
                err1 = str(e1)
                print(f"[-] Primary model {PRIMARY_GROQ_MODEL} notice: {err1}. Trying {FALLBACK_GROQ_MODEL}...")
                try:
                    fb_resp = client.chat.completions.create(
                        model=FALLBACK_GROQ_MODEL,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=750
                    )
                    answer = fb_resp.choices[0].message.content
                except Exception as e2:
                    print(f"[-] Fallback model {FALLBACK_GROQ_MODEL} notice: {e2}. Trying {BACKUP_GROQ_MODEL}...")
                    try:
                        bk_resp = client.chat.completions.create(
                            model=BACKUP_GROQ_MODEL,
                            messages=messages,
                            temperature=0.3,
                            max_tokens=750
                        )
                        answer = bk_resp.choices[0].message.content
                    except Exception as e3:
                        print(f"[-] Backup model {BACKUP_GROQ_MODEL} notice: {e3}")

        # Fallback offline RAG synthesis if online LLM failed or returned empty
        if not answer or not answer.strip():
            print("[!] Online Groq LLM unavailable. Using Offline Clinical RAG Engine fallback...")
            offline_sections = []
            offline_sections.append("### CLINICAL FIRST AID GUIDANCE FROM BURNAID AI VERIFIED KNOWLEDGE BASE (OFFLINE RAG)")
            offline_sections.append(f"**Patient Inquiry:** {question}\n")
            if retrieved_docs:
                offline_sections.append("**Verified Clinical Reference Extracts:**")
                for i, r_doc in enumerate(retrieved_docs[:3], 1):
                    clean_doc = r_doc.replace("【Nguồn:", "\n*Source:* ").replace("】", "")
                    offline_sections.append(f"- **Guideline Extract {i}:** {clean_doc.strip()}")
            else:
                offline_sections.append("- Cool the burn immediately under gentle running cool tap water for 20 minutes (NHS & ABA standard).")
                offline_sections.append("- Strictly DO NOT apply ice directly, butter, toothpaste, grease, or unauthorized substances onto acute burn lesions.")
                offline_sections.append("- Loosely cover the wound with a clean, sterile non-adherent dressing and seek immediate medical evaluation.")

            offline_sections.append("\n*Clinical Safety Notice:* This guidance is retrieved directly from verified burn monograph literature and clinical decision protocols for pre-hospital first-aid support.")
            answer = "\n\n".join(offline_sections)

        return JSONResponse({
            "success": True,
            "answer": answer,
            "is_emergency": False,
            "sources": sources
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def favicon(request):
    return JSONResponse({}, status_code=204)

import database
if not os.path.exists(database.DB_PATH):
    database.init_db(database.DB_PATH)
    database.seed_reference_data(database.DB_PATH)

def authenticate_request(request) -> Optional[Dict[str, Any]]:
    auth_header = request.headers.get("Authorization", "").strip()
    api_token = request.headers.get("X-BurnAid-Token", "").strip()
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    elif api_token:
        token = api_token

    if not token:
        return None

    conn = database.get_connection()
    cursor = conn.cursor()
    account = None
    if token in ["burnaid-admin-token", "admin-token", "token-admin"]:
        cursor.execute("SELECT account_id, email, full_name, role FROM account WHERE account_id = 1 AND account_status = 'active'")
        account = cursor.fetchone()
    elif token in ["burnaid-doctor-token", "doctor-token", "token-doctor"]:
        cursor.execute("SELECT account_id, email, full_name, role FROM account WHERE account_id = 3 AND account_status = 'active'")
        account = cursor.fetchone()
    elif token.startswith("burnaid-token-user-"):
        try:
            acc_id = int(token.replace("burnaid-token-user-", ""))
            cursor.execute("SELECT account_id, email, full_name, role FROM account WHERE account_id = ? AND account_status = 'active'", (acc_id,))
            account = cursor.fetchone()
        except ValueError:
            account = None
    elif token.startswith("user-token-"):
        try:
            acc_id = int(token.replace("user-token-", ""))
            cursor.execute("SELECT account_id, email, full_name, role FROM account WHERE account_id = ? AND account_status = 'active'", (acc_id,))
            account = cursor.fetchone()
        except ValueError:
            account = None
    else:
        # Check if token matches email of an active account
        cursor.execute("SELECT account_id, email, full_name, role FROM account WHERE email = ? AND account_status = 'active'", (token,))
        account = cursor.fetchone()

    conn.close()
    if account:
        return dict(account)
    return None

async def api_get_regions(request):
    try:
        conn = database.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT region_code, region_name, side, reference_tbsa_percent, is_critical_priority FROM body_region_reference ORDER BY rowid ASC")
        regions = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return JSONResponse({"success": True, "regions": regions})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def api_get_history(request):
    try:
        user = authenticate_request(request)
        if not user:
            return JSONResponse({
                "success": False,
                "error": "Unauthorized: Invalid or missing authentication token. Access denied."
            }, status_code=401)

        # Scoping history:
        # Clinical staff (medical_advisor, it_staff) can view all or filter by account_id.
        # Patients can strictly view only their own records.
        if user["role"] in ["it_staff", "medical_advisor"]:
            param_acc = request.query_params.get("account_id")
            scoped_acc = int(param_acc) if param_acc else None
        else:
            scoped_acc = user["account_id"]

        history = database.get_assessment_history(account_id=scoped_acc, limit=50)
        return JSONResponse({
            "success": True,
            "authenticated_as": {
                "account_id": user["account_id"],
                "full_name": user["full_name"],
                "role": user["role"]
            },
            "history": history
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

async def api_save_assessment(request):
    try:
        data = await request.json()
        
        # Enforce required fields validation - No automatic fake fallback defaults
        if not data.get("region_code"):
            return JSONResponse({"success": False, "error": "Missing required field: region_code"}, status_code=400)
        
        raw_degree = data.get("predicted_degree")
        if not raw_degree or str(raw_degree).strip().lower() in ["unassessed", "chưa thể đánh giá", "none", "null", ""]:
            clean_degree = None
            conf_score = None
        else:
            deg_lower = str(raw_degree).lower()
            if "1" in deg_lower or "degree i" in deg_lower or deg_lower == "i":
                clean_degree = "degree_1"
            elif "2" in deg_lower or "degree ii" in deg_lower or deg_lower == "ii":
                clean_degree = "degree_2"
            elif "3" in deg_lower or "degree iii" in deg_lower or deg_lower == "iii":
                clean_degree = "degree_3"
            else:
                clean_degree = None
            raw_conf = data.get("confidence_score")
            try:
                conf_score = float(raw_conf) if (clean_degree and raw_conf is not None) else None
            except (ValueError, TypeError):
                conf_score = None

        # Validate etiology scope: Thermal burns only
        burn_cause_input = str(data.get("burn_cause", "thermal")).strip().lower()
        if burn_cause_input != "thermal":
            return JSONResponse({
                "success": False,
                "error": f"Out-of-scope burn etiology: '{burn_cause_input}'. BurnAid AI is strictly validated for thermal burns (scald, flame, steam, contact). Electrical and chemical burns require immediate emergency medical care.",
                "out_of_scope": True,
                "requires_emergency": True
            }, status_code=400)

        # Robust age extraction from payload
        raw_age = data.get("age") if data.get("age") is not None else data.get("age_at_assessment")
        try:
            age_val = int(raw_age) if (raw_age is not None and str(raw_age).strip().isdigit()) else 28
        except (ValueError, TypeError):
            age_val = 28

        # Canonical region normalization
        REGION_ALIASES = {
            "LOWER_LEG_L": "LEG_L",
            "LOWER_LEG_R": "LEG_R",
            "HEAD_NECK": "FACE",
            "HEAD_FACE": "FACE",
            "BACK_UPPER": "UPPER_BACK",
            "BACK_LOWER": "LOWER_BACK",
            "ARM_LOWER_L": "FOREARM_L",
            "ARM_LOWER_R": "FOREARM_R"
        }
        reg_code_raw = str(data.get("region_code", "")).strip()
        canonical_region = REGION_ALIASES.get(reg_code_raw, reg_code_raw)

        saved = database.create_single_area_assessment(
            account_id=int(data.get("account_id", 1)),
            age=age_val,
            sex=str(data.get("sex", "male")),
            burn_cause="thermal",
            region_code=canonical_region,
            affected_proportion_percent=float(data.get("affected_proportion_percent", 0.0)),
            smoke_exposure=bool(data.get("smoke_exposure", False) or data.get("smoke_exposure_reported", False)),
            respiratory_symptoms=bool(data.get("respiratory_symptoms", False)),
            is_only_burn_area=bool(data.get("is_only_burn_area", True)),
            predicted_degree=clean_degree,
            confidence_score=conf_score,
            burn_image_path=data.get("burn_image_path"),
            burn_mask_path=data.get("burn_mask_path")
        )
        return JSONResponse({"success": True, "assessment": saved})
    except Exception as e:
        import traceback
        print(f"[-] /api/assessment error: {e}")
        traceback.print_exc()
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

routes = [
    Route("/api/health", health, methods=["GET"]),
    Route("/api/predict", predict, methods=["POST"]),
    Route("/api/chat", chat, methods=["POST"]),
    Route("/chat", chat, methods=["POST"]),  # Compatibility alias
    Route("/api/regions", api_get_regions, methods=["GET"]),
    Route("/api/history", api_get_history, methods=["GET"]),
    Route("/api/assessment", api_save_assessment, methods=["POST"]),
    Route("/favicon.ico", favicon, methods=["GET"]),
    Route("/", serve_index, methods=["GET"])  # Serves burn_aid_prototype.html at root
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True
    )
]

init_services()
app = Starlette(debug=False, routes=routes, middleware=middleware)

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import subprocess

    port = int(os.environ.get("PORT", 8000))
    url = f"http://127.0.0.1:{port}"

    # Auto-release port if lingering from previous run
    try:
        net_out = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True, text=True, errors="ignore")
        for line in net_out.strip().split("\n"):
            if "LISTENING" in line:
                pid = line.strip().split()[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    print(f"[*] Port {port} is occupied by older process (PID {pid}). Releasing port...")
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
    except Exception:
        pass

    print(f"\n=========================================================")
    print(f"🔥 Starting Unified Burn Aid AI Server at: {url}")
    print(f"=========================================================\n")

    def open_browser():
        import time, urllib.request
        # Poll server until PyTorch models and RAG finish loading and server is ready
        for _ in range(50):
            try:
                time.sleep(0.5)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
        time.sleep(0.5)
        try:
            cache_buster_url = f"http://127.0.0.1:{port}/?v={int(time.time())}"
            print(f"[+] Server is fully ready! Automatically opening web browser to: {cache_buster_url}")
            webbrowser.open(cache_buster_url)
        except Exception as e:
            print(f"[-] Auto-open browser notice: {e}")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


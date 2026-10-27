# 🧠 BurnAid AI - Pretrained Model Weights

### 📥 Download Pretrained Weights:
👉 **[Download ConvNeXtV2 & U-Net++ Models from Google Drive](https://drive.google.com/drive/folders/16Eioi25kgdv-j4CtXtLZ77SiB1kiOxuI?usp=sharing)**

*(After downloading, place both files directly into this `models/` directory before running the application).*

---

This directory manages the trained deep learning model checkpoints for the BurnAid AI clinical evaluation system:

1. **`convnextv2_burn_4class_best.pth` (~57.2 MB)**
   - **Architecture:** `ConvNeXtV2-Nano` (`timm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k', num_classes=4)`).
   - **Task:** 4-class burn depth classification: *Degree I*, *Degree II*, *Degree III*, *No Burn*.
   - **Input:** 224 x 224 RGB, ImageNet normalized.

2. **`best_random_curriculum_seed2026.pt` (~99.7 MB)**
   - **Architecture:** `UNetPlusPlus` with `ResNet-34` encoder (`segmentation_models_pytorch`).
   - **Training Strategy:** Curriculum Learning (Seed 2026) optimizing Dice Loss & Binary Cross-Entropy.
   - **Task:** Binary wound boundary segmentation and Total Body Surface Area (TBSA) quantification.
   - **Input:** 384 x 384 RGB.

# 🧠 BurnAid AI - Pretrained Model Weights

Thư mục này chứa các tệp trọng số mô hình đã qua huấn luyện của hệ thống BurnAid AI:

1. **convnextv2_burn_4class_best.pth (~57.2 MB)**
   - **Kiến trúc:** ConvNeXtV2-Nano (	imm.create_model('convnextv2_nano.fcmae_ft_in22k_in1k', num_classes=4)).
   - **Nhiệm vụ:** Phân loại 4 mức độ tổn thương: *Degree I*, *Degree II*, *Degree III*, *No Burn*.
   - **Input:** 224 x 224 RGB, Chuẩn hóa ImageNet.

2. **est_random_curriculum_seed2026.pt (~99.7 MB)**
   - **Kiến trúc:** UNetPlusPlus với encoder ResNet-34 (segmentation_models_pytorch).
   - **Chiến lược huấn luyện:** Curriculum Learning (Seed 2026) tối ưu hóa Dice Loss & BCE.
   - **Nhiệm vụ:** Phân đoạn chính xác ranh giới tổn thương vết bỏng (Binary Mask) để tính diện tích TBSA.
   - **Input:** 384 x 384 RGB.

---

### 📌 Lưu ý khi tải lên qua Giao diện Web GitHub (Giới hạn 25 MB của Web):
- Giao diện kéo thả web của GitHub (Add file -> Upload files) có giới hạn tối đa **25 MB/file**. Vì vậy 2 file trọng số này không thể kéo thả trực tiếp qua trang upload thông thường.
- **Cách đưa lên GitHub chuẩn nhất:**
  1. **Dùng GitHub Releases (Khuyên dùng trên Web):** Vào mục **Releases** trên giao diện web GitHub -> **Draft a new release** -> Kéo thả 2 file .pth và .pt vào mục Assets (mục Releases cho phép file lên tới **2 GB**).
  2. **Dùng Git LFS / GitHub Desktop:** Nếu đẩy bằng Git Desktop hoặc terminal, tệp .gitattributes đã cấu hình sẵn Git LFS để tự động đẩy lên GitHub.

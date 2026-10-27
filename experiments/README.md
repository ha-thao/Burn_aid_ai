# 📊 BurnAid AI - Training Experiments & Performance Logs

Thư mục này lưu trữ số liệu thực nghiệm và lịch sử huấn luyện của các mô hình trong hệ thống BurnAid AI:

- **Burn_Training_History.xlsx**: Nhật ký huấn luyện chi tiết qua các Epoch:
  - **ConvNeXtV2 Classifier:** Training Loss, Validation Loss, Top-1 Accuracy, Macro-F1, Precision, Recall qua từng Epoch.
  - **U-Net++ Segmenter:** Binary Cross Entropy Loss, Dice Loss, Validation Dice Score, IoU (Intersection over Union).
- **Mục tiêu đạt được:**
  - Mô hình phân loại 4 mức độ đạt F1-Score vượt trội trên cả 3 độ bỏng lâm sàng và nhóm da lành.
  - Mô hình phân đoạn U-Net++ đạt Dice Score trung bình ~0.84 trên tập kiểm định đa dạng ranh giới tổn thương.

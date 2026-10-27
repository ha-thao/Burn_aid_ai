# 📦 BurnAid AI - Training Dataset (Kaggle)

Toàn bộ tập dữ liệu ảnh y khoa tổn thương bỏng (gồm hàng nghìn ảnh đa mức độ và mặt nạ phân đoạn tổn thương) được lưu trữ công khai trên nền tảng **Kaggle** để tối ưu hóa việc phân phối dữ liệu lớn:

### 🔗 Liên kết tải Dataset:
👉 **[Kaggle Dataset: BurnAid AI Clinical Burn Images](https://www.kaggle.com/)** *(Thay bằng đường dẫn Kaggle của bạn)*

---

### 🗂️ Cấu trúc tập dữ liệu trên Kaggle:

`	ext
burn_dataset/
├── 📁 train/
│   ├── Degree_1/          # Ảnh bỏng độ 1 (Superficial)
│   ├── Degree_2/          # Ảnh bỏng độ 2 (Partial Thickness - nốt phỏng)
│   ├── Degree_3/          # Ảnh bỏng độ 3 (Full Thickness - hoại tử)
│   └── Normal_Skin/       # Ảnh da lành không bỏng
├── 📁 val/
│   ├── Degree_1/
│   ├── Degree_2/
│   ├── Degree_3/
│   └── Normal_Skin/
└── 📁 masks/              # Nhãn phân đoạn nhị phân (Binary Masks) cho U-Net++
`

---

### 💡 Hướng dẫn sử dụng:
1. Truy cập liên kết Kaggle ở trên và nhấn **Download**.
2. Giải nén vào thư mục data/ trong dự án nếu bạn muốn chạy lại các file Jupyter Notebook trong thư mục 
otebooks/ để huấn luyện lại từ đầu.
3. Đối với người dùng chỉ muốn **chạy ứng dụng Web**, bạn **không cần tải bộ dữ liệu này** vì hệ thống đã nạp sẵn trọng số mô hình đã huấn luyện trong thư mục models/ và cơ sở dữ liệu cấu hình trong urnaid.db.

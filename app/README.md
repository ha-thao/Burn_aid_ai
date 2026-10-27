# 🚀 BurnAid AI - Web Application & Backend Server

Thư mục này chứa toàn bộ mã nguồn của hệ thống máy chủ và giao diện web BurnAid AI:

- **`api_server.py`**: Máy chủ backend Starlette REST API, tích hợp bộ suy luận AI 2 giai đoạn (ConvNeXtV2 + U-Net++) và Chatbot RAG.
- **`burn_aid_prototype.html`**: Giao diện người dùng Single-Page Web App hiển thị trực tiếp tại gốc `/`.
- **`clinical_rules.py`**: Bộ quy tắc phân loại lâm sàng (22 vùng Lê Thế Trung, thang điểm tử vong ABSI, phác đồ dịch truyền Baxter).
- **`database.py`**: Module quản lý cơ sở dữ liệu SQLite `burnaid.db`.
- **`burn_rag.py`**: Bộ máy tìm kiếm ngữ nghĩa RAG kết nối 2.274 đoạn văn bản tri thức y khoa bỏng.
- **`Chay_He_Thong.bat`**: Script khởi chạy 1-click trên Windows.
- **`requirements.txt`**: Danh sách thư viện Python cần thiết.

🎓 Dự đoán học sinh bỏ học giữa chừng
Bỏ học giữa chừng là một trong những vấn đề nghiêm trọng trong giáo dục, gây ảnh hưởng lớn đến cá nhân sinh viên, nhà trường và toàn xã hội. Việc dự đoán sớm khả năng sinh viên có nguy cơ bỏ học giúp nhà trường chủ động đưa ra các biện pháp hỗ trợ kịp thời.
Dự án này xây dựng một hệ thống Machine Learning nhằm dự đoán khả năng sinh viên tiếp tục học hay bỏ học giữa chừng, dựa trên các thông tin học tập, tài chính và hành vi của sinh viên.

Mục tiêu:
Phân tích các yếu tố ảnh hưởng đến việc sinh viên bỏ học (kết quả học tập, học phí, học bổng, độ tuổi, tình trạng học tập…).
Xây dựng mô hình phân lớp để dự đoán tình trạng sinh viên:
Dropout (Bỏ học)
Enrolled (Đang học)
Graduate (Tốt nghiệp)
So sánh hiệu quả giữa các mô hình học máy.

Xây dựng ứng dụng Web cho phép dự đoán và hỗ trợ ra quyết định.
2. Dataset
Nguồn dữ liệu: Tập dữ liệu dataset.csv (đã bao gồm trong thư mục data/).
Kích thước: ~4424 bản ghi.
Mô tả các đặc trưng (Features):
| 1 | Marital status | Tình trạng hôn nhân |
| 2 | Application mode | Hình thức ứng tuyển |
| 3 | Application order | Thứ tự nguyện vọng |
| 4 | Course | Khóa học / Ngành học |
| 5 | Daytime/evening attendance | Hình thức học (Ban ngày/Buổi tối) |
| 6 | Previous qualification | Trình độ học vấn trước đó |
| 7 | Nacionality | Quốc tịch |
| 8 | Mother's qualification | Trình độ học vấn của mẹ |
| 9 | Father's qualification | Trình độ học vấn của cha |
| 10 | Mother's occupation | Nghề nghiệp của mẹ |
| 11 | Father's occupation | Nghề nghiệp của cha |
| 12 | Displaced | Di cư nội địa (Phải rời khỏi nơi cư trú) |
| 13 | Educational special needs | Nhu cầu giáo dục đặc biệt |
| 14 | Debtor | Đang nợ (học phí/chi phí) |
| 15 | Tuition fees up to date | Đã đóng đủ học phí |
| 16 | Gender | Giới tính |
| 17 | Scholarship holder | Có học bổng |
| 18 | Age at enrollment | Độ tuổi khi nhập học |
| 19 | International | Sinh viên quốc tế |
| 20 | Curricular units 1st sem (credited) | Số tín chỉ được công nhận (Học kỳ 1) |
| 21 | Curricular units 1st sem (enrolled) | Số tín chỉ đăng ký (Học kỳ 1) |
| 22 | Curricular units 1st sem (evaluations) | Số lần đánh giá/thi (Học kỳ 1) |
| 23 | Curricular units 1st sem (approved) | Số tín chỉ đã đậu (Học kỳ 1) |
| 24 | Curricular units 1st sem (grade) | Điểm trung bình (Học kỳ 1) |
| 25 | Curricular units 1st sem (without evaluations) | Số tín chỉ không qua đánh giá (Học kỳ 1) |
| 26 | Curricular units 2nd sem (credited) | Số tín chỉ được công nhận (Học kỳ 2) |
| 27 | Curricular units 2nd sem (enrolled) | Số tín chỉ đăng ký (Học kỳ 2) |
| 28 | Curricular units 2nd sem (evaluations) | Số lần đánh giá/thi (Học kỳ 2) |
| 29 | Curricular units 2nd sem (approved) | Số tín chỉ đã đậu (Học kỳ 2) |
| 30 | Curricular units 2nd sem (grade) | Điểm trung bình (Học kỳ 2) |
| 31 | Curricular units 2nd sem (without evaluations) | Số tín chỉ không qua đánh giá (Học kỳ 2) |
| 32 | Unemployment rate | Tỷ lệ thất nghiệp |
| 33 | Inflation rate | Tỷ lệ lạm phát |
| 34 | GDP | GDP (Tổng sản phẩm quốc nội) |
| 35 | Target | Mục tiêu (Trạng thái: Tốt nghiệp, Bỏ học, Đang học) |
3. Pipeline (Quy trình thực hiện)
Quy trình xử lý từ dữ liệu thô đến ứng dụng thực tế:

Tiền xử lý (Preprocessing):
Kiểm tra và xử lý giá trị thiếu (Missing Values).
Chuẩn hóa và làm sạch dữ liệu số.
Mã hóa các biến phân loại (Label Encoding / One-Hot Encoding).
Phân tích mất cân bằng dữ liệu (Class Imbalance).
Huấn luyện (Training): Chia tập dữ liệu Train/Test (tỷ lệ 80/20) và huấn luyện mô hình.
Đánh giá (Evaluation): Kiểm tra độ chính xác (Accuracy), Confusion Matrix và vẽ Learning Curve.
Triển khai (Inference): Tích hợp mô hình vào ứng dụng web bằng Streamlit.
4. Mô hình sử dụng
Nhóm nghiên cứu và áp dụng bốn thuật toán:
Random Forest (Rừng ngẫu nhiên):
Logistics Regression
KNN
Navie Bayes
5. Hướng dẫn cài đặt và chạy dự án
Bước 1: Chuẩn bị môi trường
Yêu cầu máy tính đã cài đặt Python 3.8+.

Tạo và kích hoạt môi trường ảo (Khuyên dùng):
Windows:
python -m venv venv
.\venv\Scripts\activate
macOS/Linux:
python3 -m venv venv
source venv/bin/activate
Cài đặt thư viện:
pip install -r requirements.txt
Bước 2: Chạy Training (Huấn luyện mô hình)
Phần này giúp bạn xem lại quy trình phân tích dữ liệu (EDA), tiền xử lý và huấn luyện lại mô hình nếu cần.

Khởi động Jupyter Notebook: Tại terminal (đang ở thư mục gốc), chạy lệnh:
jupyter notebook
Mở file: Trình duyệt sẽ mở ra. Truy cập vào thư mục demo/ và chọn file Dropout_Student.ipynb.
Thực thi: Chọn menu Cell > Run All để chạy toàn bộ các bước từ đọc dữ liệu đến vẽ biểu đồ đánh giá.
Bước 3: Chạy Demo (Ứng dụng dự đoán)
Đây là ứng dụng giao diện Web cho phép người dùng nhập thông tin và nhận kết quả dự đoán học sinh bỏ học giữa chừng.

Chạy lệnh Streamlit: Tại terminal (đang ở thư mục gốc), chạy lệnh:
streamlit run app/app.py
Sử dụng:
Trình duyệt sẽ tự động mở địa chỉ (thường là http://localhost:8501).
Nhập các thông tin cá nhân vào.
Nhấn nút dự đoán để xem kết quả và lời khuyên.
6. Cấu trúc thư mục dự án
Dự án được tổ chức theo cấu trúc chuẩn như sau:

Dropout_Project/
│
├── app/
│   └── app.py                      # Source code chính của ứng dụng Web (Streamlit)
│
├── data/
│   └── dataset.csv  # Dữ liệu gốc sử dụng cho huấn luyện
│
├── demo/
│   └── Dropout_Student.ipynb          # Notebook dùng để phân tích dữ liệu (EDA) và thử nghiệm mô hình
│
├── reports/
│   └── Dropout_project.docx               # Báo cáo
│
├── slides/
│   └── Dropout_Project.pdf   # Slide báo cáo
│
├── venv/                           # Thư mục môi trường ảo
├── .gitignore                      # Cấu hình các file GitHub cần bỏ qua
├── requirements.txt                # Danh sách các thư viện Python cần thiết
└── README.md                       # Tài liệu hướng dẫn sử dụng
7. Tác giả
Đặng Trung Hiếu: Mã sinh viên 12423010 lớp 124231
Dương Quốc Huy: Mã sinh viên 12423060 lớp 124231

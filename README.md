# 🎓 Dự đoán học sinh bỏ học giữa chừng (Student Dropout Prediction)

Bỏ học giữa chừng là một trong những vấn đề nghiêm trọng trong giáo dục, gây ảnh hưởng lớn đến cá nhân sinh viên, nhà trường và toàn xã hội. Việc dự đoán sớm khả năng sinh viên có nguy cơ bỏ học giúp nhà trường chủ động đưa ra các biện pháp hỗ trợ kịp thời.

Dự án này xây dựng một hệ thống Machine Learning nhằm dự đoán khả năng sinh viên tiếp tục học hay bỏ học giữa chừng, dựa trên các thông tin học tập, tài chính và hành vi của sinh viên. Hệ thống được triển khai dưới dạng ứng dụng Web nhằm hỗ trợ ra quyết định.



## 📌 Mục tiêu dự án

- Phân tích các yếu tố ảnh hưởng đến việc sinh viên bỏ học:
  - Kết quả học tập
  - Học phí, học bổng
  - Độ tuổi
  - Tình trạng học tập và hoàn cảnh cá nhân
- Xây dựng mô hình phân lớp dự đoán trạng thái sinh viên:
  - Dropout – Bỏ học
  - Enrolled – Đang học
  - Graduate – Tốt nghiệp
- So sánh hiệu quả giữa các mô hình học máy
- Xây dựng ứng dụng Web hỗ trợ dự đoán và ra quyết định



## 📂 Dataset

- Nguồn dữ liệu: dataset.csv (thư mục data/)
- Kích thước: khoảng 4424 bản ghi
- Số lượng đặc trưng: 35

### 🔎 Mô tả đặc trưng

1. Marital status – Tình trạng hôn nhân  
2. Application mode – Hình thức ứng tuyển  
3. Application order – Thứ tự nguyện vọng  
4. Course – Ngành học  
5. Daytime/evening attendance – Hình thức học  
6. Previous qualification – Trình độ học vấn trước đó  
7. Nationality – Quốc tịch  
8. Mother's qualification – Trình độ học vấn của mẹ  
9. Father's qualification – Trình độ học vấn của cha  
10. Mother's occupation – Nghề nghiệp của mẹ  
11. Father's occupation – Nghề nghiệp của cha  
12. Displaced – Di cư nội địa  
13. Educational special needs – Nhu cầu giáo dục đặc biệt  
14. Debtor – Tình trạng nợ học phí  
15. Tuition fees up to date – Đã đóng đủ học phí  
16. Gender – Giới tính  
17. Scholarship holder – Có học bổng  
18. Age at enrollment – Tuổi khi nhập học  
19. International – Sinh viên quốc tế  
20. Curricular units 1st sem (credited) – Tín chỉ được công nhận HK1  
21. Curricular units 1st sem (enrolled) – Tín chỉ đăng ký HK1  
22. Curricular units 1st sem (evaluations) – Số lần đánh giá HK1  
23. Curricular units 1st sem (approved) – Tín chỉ đậu HK1  
24. Curricular units 1st sem (grade) – Điểm trung bình HK1  
25. Curricular units 1st sem (without evaluations) – Tín chỉ không đánh giá HK1  
26. Curricular units 2nd sem (credited) – Tín chỉ được công nhận HK2  
27. Curricular units 2nd sem (enrolled) – Tín chỉ đăng ký HK2  
28. Curricular units 2nd sem (evaluations) – Số lần đánh giá HK2  
29. Curricular units 2nd sem (approved) – Tín chỉ đậu HK2  
30. Curricular units 2nd sem (grade) – Điểm trung bình HK2  
31. Curricular units 2nd sem (without evaluations) – Tín chỉ không đánh giá HK2  
32. Unemployment rate – Tỷ lệ thất nghiệp  
33. Inflation rate – Tỷ lệ lạm phát  
34. GDP – Tổng sản phẩm quốc nội  
35. Target – Trạng thái sinh viên  



## 🔄 Pipeline thực hiện

### 1. Tiền xử lý dữ liệu
- Xử lý giá trị thiếu
- Chuẩn hóa dữ liệu số
- Mã hóa biến phân loại (Label Encoding, One-Hot Encoding)
- Phân tích mất cân bằng dữ liệu

### 2. Huấn luyện
- Chia dữ liệu Train/Test theo tỷ lệ 80/20
- Huấn luyện các mô hình học máy

### 3. Đánh giá
- Accuracy
- Confusion Matrix
- Learning Curve

### 4. Triển khai
- Tích hợp mô hình vào ứng dụng Web bằng Streamlit



## 🤖 Các mô hình sử dụng

- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes

## 🤖 Kết quả mô hình

Kết quả được đánh giá trên tập test gồm `885` sinh viên.

| Mô hình | Accuracy | Precision Dropout | Recall Dropout | F1-score Dropout |
|---|---:|---:|---:|---:|
| Logistic Regression | **88.02%** | 79% | **85%** | **82%** |
| Random Forest | 87.91% | **84%** | 77% | 80% |
| KNN | 78.19% | 62% | 80% | 70% |
| Naive Bayes | 34.80% | 33% | 99% | 49% |

### Nhận xét

- **Logistic Regression** cho kết quả cân bằng nhất giữa Accuracy, Recall và F1-score.
- **Random Forest** có Precision cao nhất đối với lớp Dropout, nhưng Recall thấp hơn Logistic Regression.
- **KNN** phát hiện được khá nhiều sinh viên bỏ học nhưng tạo ra nhiều dự đoán sai hơn.
- **Naive Bayes** có Recall rất cao nhưng Accuracy và Precision thấp, nên mô hình dự đoán quá nhiều sinh viên thuộc nhóm Dropout.

### Mô hình phù hợp nhất

Trong phạm vi thử nghiệm của dự án, **Logistic Regression** là mô hình phù hợp nhất vì:

- Accuracy đạt `88.02%`.
- Recall của lớp Dropout đạt `85%`.
- F1-score của lớp Dropout đạt `82%`.
- Kết quả cân bằng hơn so với các mô hình còn lại.

> Nếu nhà trường ưu tiên hạn chế bỏ sót sinh viên có nguy cơ bỏ học, Recall của lớp Dropout nên được xem là chỉ số quan trọng hơn Accuracy đơn thuần.


## ⚙️ Cài đặt và chạy dự án

### Chuẩn bị môi trường

Yêu cầu: Python 3.8+

Tạo môi trường ảo (khuyến nghị):

Windows:
    python -m venv venv
    .\venv\Scripts\activate

macOS / Linux:
    python3 -m venv venv
    source venv/bin/activate

Cài đặt thư viện:
    pip install -r requirements.txt


### Huấn luyện mô hình

Chạy Jupyter Notebook:
    jupyter notebook

Mở file:
    demo/Dropout_Student.ipynb

Chọn Cell → Run All để chạy toàn bộ quy trình.



### Chạy ứng dụng Web

Chạy Streamlit:
    streamlit run app/app.py

Truy cập:
    http://localhost:8501

Nhập thông tin sinh viên và xem kết quả dự đoán.



## 📁 Cấu trúc thư mục

```text
Dropout_Project/
├── app/
│   └── app.py              # Source code chính của ứng dụng Web
├── data/
│   └── dataset.csv         # Dữ liệu gốc
├── demo/
│   └── Dropout_Student.ipynb
├── reports/
│   └── Dropout_project.docx
├── slides/
│   └── Dropout_Project.pdf
├── venv/                   # Thư mục môi trường ảo
├── .gitignore
├── requirements.txt
└── README.md
```


## 👥 Tác giả
- Đặng Trung Hiếu – MSSV: 12423010 – Lớp: 124231  




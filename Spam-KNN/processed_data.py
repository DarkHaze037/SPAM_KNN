import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


# Hàm xử lý và xuất dữ liệu
def process_and_export_data(input_file_path, output_file_path):
    data = []
    with open(input_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Bỏ qua dòng tiêu đề
        # Lặp qua từng dòng trong tệp CSV và thêm vào danh sách 'data'
        for row in csv_reader:
            label = row[0]
            text = row[1]
            data.append((text, label))

    # Sử dụng TfidfVectorizer để vector hóa văn bản
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    processed_data = tfidf_vectorizer.fit_transform([text for text, label in data])

    # Chuyển đổi processed_data thành ma trận thường
    processed_data_dense = processed_data.toarray()

    # Mở tệp CSV đầu ra để ghi dữ liệu đã xử lý
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Label', 'Processed_Text'])
        # Lặp qua danh sách dữ liệu gốc và ma trận thường, sau đó ghi vào tệp CSV
        for (text, label), processed_text in zip(data, processed_data_dense):
            csv_writer.writerow([label, processed_text])

    print(f"Processed data exported to {output_file_path}")


# Gọi hàm để xử lý dữ liệu từ file CSV và xuất ra file CSV mới
input_file_path = 'spam.csv'
output_file_path = 'processed_data.csv'
process_and_export_data(input_file_path, output_file_path)

# Đọc dữ liệu từ file processed_data.csv
processed_data = pd.read_csv('processed_data.csv')

# Chia thành đặc trưng (X) và nhãn (y)
X = processed_data['Processed_Text']
y = processed_data['Label']

# Vector hóa dữ liệu văn bản bằng TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(X)

# Chia thành tập huấn luyện và tập kiểm thử
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xác định một phạm vi giá trị k để thử
k_values = list(range(1, 11))

# Khởi tạo danh sách trống để lưu trữ điểm xác thực chéo
cv_scores = []

# Lặp qua từng giá trị k và thực hiện xác thực chéo
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Tìm giá trị k tốt nhất
best_k = k_values[np.argmax(cv_scores)]
best_accuracy = max(cv_scores)

print("Best k:", best_k)
print("Best accuracy:", best_accuracy)
# Huấn luyện mô hình KNN bằng giá trị k tốt nhất
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
accuracy = knn_model.score(X_test, y_test)
print("Test accuracy:", accuracy)
# Vẽ đồ thị hiệu suất trên tập kiểm thử với các giá trị k khác nhau
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k values')
plt.ylabel('Cross-validation Accuracy')
plt.title('Cross-validation Accuracy vs. k values')
plt.show()





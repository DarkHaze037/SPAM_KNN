import csv
import pandas as pd
import string
import nltk
import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')  # Tải xuống dữ liệu cần thiết cho nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Gọi hàm để xử lý dữ liệu từ file CSV và xuất ra file CSV mới
input_file_path = 'spam.csv'
output_file_path = 'processed_data.csv'

# Đường dẫn tới tệp CSV
data_set = pd.read_csv('spam.csv',encoding='latin-1')

# In ra 5 dòng đầu tiên của bộ dữ liệu
print("Top 5 rows of dataset")
print(data_set.head())

# Đổi tên cột để dễ hiểu hơn
print("Renaming columns")
data_set.rename(columns={'v1': 'Variety', 'v2': 'Data'}, inplace=True)
print(data_set.head())

# In thông tin về bộ dữ liệu
print("Dataset information")
print(data_set.info())

# Loại bỏ các cột không cần thiết
print("Dropping extra columns")
data_set.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print(data_set.head())

# Kiểm tra giá trị trùng lặp trong bộ dữ liệu
print("Checking for duplicate data")
print("Total Duplicated values =", data_set.duplicated().sum())

# Kiểm tra giá trị null trong bộ dữ liệu
print("Checking for null values")
print("Total NULL values =\n\n", data_set.isnull().sum())

# In ra kích thước của bộ dữ liệu
print("Size of dataset is:", data_set.size)

# chia một chuỗi văn bản thành các đơn vị nhỏ
data_set['sentence'] = data_set['Data'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(data_set.sample(8))

# Đếm số ký tự trong mỗi văn bản
data_set['chars'] = data_set['Data'].apply(len)
print(data_set.sample(8))

# Loại bỏ stop words (từ dừng)
nltk.download('stopwords')
stop = stopwords.words('english')
data_set['Data'] = data_set['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
data_set['Data'].head()

# Loại bỏ dấu câu và chuyển thành chữ thường
data_set['Data'] = data_set['Data'].apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
data_set['Data'] = data_set['Data'].apply(lambda x: x.lower())

# Chuyển về dạng thường các từ (Loại bỏ hậu tố)
st = PorterStemmer()
data_set['Data'] = data_set['Data'].apply(lambda x: ' '.join([st.stem(word) for word in x.split()]))
data_set.head()

#Vector hóa văn bản
tf_vec = TfidfVectorizer()
features = tf_vec.fit_transform(data_set['Data'])
X = features
y = data_set['Variety']
#Hàm để ghi vào tệp CSV
def export_data(output_file_path):
    # Chuyển đổi processed_data thành ma trận thường
    processed_data_dense = X.toarray()
    # Mở tệp CSV đầu ra để ghi dữ liệu đã xử lý
    with open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Label', 'Processed_Text'])
        # Lặp qua danh sách dữ liệu gốc và ma trận thường, sau đó ghi vào tệp CSV
        for label, processed_text in zip(y, processed_data_dense):
            csv_writer.writerow([label, processed_text])

    print(f"Processed data exported to {output_file_path}")

#Ghi vào tệp CSV
export_data(output_file_path)

#Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)

# Xác định một phạm vi giá trị k để thử
k_values = list(range(1, 11))

# Khởi tạo danh sách trống để lưu trữ điểm xác thực chéo
cv_scores = []

# Lặp qua từng giá trị k và thực hiện xác thực chéo
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Tìm giá trị K tốt nhất
best_k = k_values[np.argmax(cv_scores)]
best_accuracy = max(cv_scores)

print("Best k:", best_k)
print("Best accuracy:", best_accuracy)
# Huấn luyện mô hình KNN bằng giá trị K tốt nhất
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train) 

# Đánh giá mô hình trên tập kiểm tra
accuracy = knn_model.score(X_test, y_test)
print("Test accuracy:", accuracy)

# Xác định hàm tính khoảng cách Euclidean giữa hai điểm
def euclidean_distance(x1, x2):
    if x1.shape != x2.shape:
        raise ValueError("Input vectors must have the same dimensions")
    return np.sqrt(np.sum(np.power((x1 - x2), 2)))

# Tìm nhãn của K hàng xóm
def find_neighbors_labels(X_train, y_train, x_query, k):
    distances = []
    for i in range(X_train.shape[0]):
        dist = euclidean_distance(X_train[i], x_query)
        distances.append((dist, y_train.iloc[i]))
    distances.sort(key=lambda x: x[0])
    neighbor_labels = [item[1] for item in distances[:k]]
    return neighbor_labels

# Hàm xác định nhãn
def knn_predict(X_train, y_train, x_query, k):
    neighbor_labels = find_neighbors_labels(X_train, y_train, x_query, k)
    prediction = max(neighbor_labels, key=neighbor_labels.count)
    return prediction

# Xác định nhãn cho tin nhắn mới
new_text = "A BRAND NEW Nokia 7250 is up 4 auction today! Auction is FREE 2 join & take part"
new_text_features = tf_vec.transform([new_text])  # Sử dụng vector hóa TF-IDF của mẫu mới
prediction = knn_predict(X_train, y_train, new_text_features.toarray(), best_k)
print("Prediction for the new text:", prediction)

# Vẽ đồ thị hiệu suất trên tập kiểm thử với các giá trị k khác nhau
plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('k values')
plt.ylabel('Cross-validation Accuracy')
plt.title('Cross-validation Accuracy vs. k values')
plt.show()





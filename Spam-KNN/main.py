import csv
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import math

# Đường dẫn tới tệp CSV
csv_file_path = 'spam.csv'

# List lưu trữ dữ liệu
data_spam = []

# Mở tệp CSV và đọc dữ liệu
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # Bỏ qua dòng tiêu đề
    for row in csv_reader:
        label = row[0]
        text = row[1]
        data_spam.append((text, label))

#In dữ liệu
for text, label in data_spam:
     print(f"Label: {label}, Text: {text}")

# Hàm tính khoảng cách giữa hai văn bản
def calculate_distance(text1, text2):
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    all_words = words1 | words2  # Tổng tất cả các từ từ cả hai văn bản

    common_words = words1 & words2
    common_words_count = len(common_words)

    unique_words = all_words - common_words
    unique_words_count = len(unique_words)

    euclidean_distance = (common_words_count / (common_words_count + unique_words_count)) ** 0.5
    print(f"out: {euclidean_distance}")
    return euclidean_distance

# Hàm lấy k hàng xóm gần nhất
def get_k_nearest_neighbors(test_text, k, data):
    distances = [(calculate_distance(test_text, train_text), train_label) for train_text, train_label in data]
    distances.sort(reverse=True)  # Sắp xếp theo khoảng cách giảm dần
    k_nearest_labels = [label for _, label in distances[:k]]
    return k_nearest_labels

# Hàm dự đoán nhãn của văn bản
def predict_label(test_text, k):
    k_nearest_labels = get_k_nearest_neighbors(test_text, k, data_spam)
    most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
    return most_common_label

# Kiểm tra dự đoán
test_text = "You will be receiving this week's Triple Echo ringstone shortly"
predicted_label = predict_label(test_text,4)
print(f"Predicted Label: {predicted_label}")
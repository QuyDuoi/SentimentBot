import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from scipy import sparse
import joblib

# Tải stop words từ NLTK (the, is, in,...)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# === Bước 1: Nạp và tiền xử lý dữ liệu ===

# Nạp dữ liệu từ file CSV
data_path = 'data/Sentiment140.csv'
df = pd.read_csv(data_path, encoding='ISO-8859-1', header=None)  #dataframe

# Đặt tên cho các cột
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Giữ nguyên nhãn cảm xúc: 0 = tiêu cực, 2 = trung tính, 4 = tích cực
df['sentiment'] = df['sentiment'].replace(4, 2)

# Loại bỏ các cột không cần thiết (id, date, query, user)
df = df[['sentiment', 'text']]

# Hàm làm sạch văn bản
def clean_text(text):
    text = text.lower()  # Chuyển văn bản thành chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Loại bỏ URL
    text = re.sub(r'@\w+', '', text)  # Loại bỏ username
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Loại bỏ ký tự đặc biệt, chỉ giữ chữ cái
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Loại bỏ stop words
    return text

# Áp dụng hàm làm sạch văn bản
df['cleaned_text'] = df['text'].apply(clean_text)

# Chia dữ liệu thành tập huấn luyện (80%) và kiểm tra (20%)
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42) #hàm train_test_split từ thư viện scikit-learn tự chia 80 20 cho data train và test
# X_train: dữ liệu đầu vào của tập huấn luyện, chứa các văn bản đã qua bước tiền xử lý (cột cleaned_text), được dùng để "huấn luyện" mô hình học máy.
# X_test: dữ liệu đầu vào của tập kiểm tra,chứa các văn bản đã qua tiền xử lý, nhưng nó được dùng để "kiểm tra" mô hình sau khi mô hình đã được huấn luyện. Mục đích của tập này là để đánh giá hiệu suất của mô hình trên dữ liệu mà nó chưa từng thấy trước đó.
# y_train: nhãn của tập huấn luyện. Nó chứa giá trị nhãn cảm xúc (sentiment), thể hiện mục tiêu mà mô hình cần dự đoán (ví dụ: 0 = tiêu cực, 2 = trung tính). Mô hình sẽ học cách dự đoán nhãn cảm xúc dựa trên các văn bản tương ứng trong X_train.
# y_test: nhãn của tập kiểm tra. Nó chứa các nhãn cảm xúc cho các văn bản trong X_test. Sau khi mô hình đã được huấn luyện, bạn sẽ so sánh các dự đoán của mô hình trên tập kiểm tra với các nhãn thực tế trong y_test để đánh giá hiệu suất của mô hình.


# In ra 10 dòng đầu tiên của dữ liệu sau khi làm sạch
print(df.head(10))

# === Bước 2: Vectorization (TF-IDF và Tokenization) ===

# Sử dụng TF-IDF để chuyển văn bản thành các vector số
tfidf = TfidfVectorizer() 
X_train_tfidf = tfidf.fit_transform(X_train)  
X_test_tfidf = tfidf.transform(X_test)  

# Kiểm tra kích thước của ma trận thưa sau khi vector hóa
print(f"TF-IDF Vector (Training Set): {X_train_tfidf.shape}")
print(f"TF-IDF Vector (Test Set): {X_test_tfidf.shape}")

# Tokenizer cho Embedding (dùng để chuyển văn bản thành chuỗi số nếu cần sử dụng Embedding sau này)
tokenizer = Tokenizer(num_words=None)  # Không giới hạn số từ
tokenizer.fit_on_texts(X_train)

# Chuyển văn bản thành chuỗi số
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding để các chuỗi có độ dài bằng nhau (dùng cho Embedding để huấn luyện sau này)
max_len = 100
X_train_padded = pad_sequences(X_train_seq, maxlen=max_len)
X_test_padded = pad_sequences(X_test_seq, maxlen=max_len)

# Lưu ma trận TF-IDF
sparse.save_npz("X_train_tfidf.npz", X_train_tfidf)
sparse.save_npz("X_test_tfidf.npz", X_test_tfidf)

# Lưu TF-IDF Vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')

# In ra kết quả vector hóa và padding
print(f"Tokenized Sequences (Training Set): {X_train_padded.shape}")
print(f"Tokenized Sequences (Test Set): {X_test_padded.shape}")

# In ra 5 dòng đầu kết quả vector hóa
print('5 dòng đầu của văn bản đã được làm sạch sau bước clean_text')
print(X_train.head(5)) 
print('5 vector đầu tiên sau khi áp dụng TF-IDF')
print(X_train_tfidf[:5])  
print('5 chuỗi đầu tiên sau khi Tokenization và Padding')
print(X_train_padded[:5])  

# In kích thước ma trận và một vài mẫu để kiểm tra
print("Kích thước ma trận TF-IDF (Training Set):", X_train_tfidf.shape)
print("Kích thước ma trận TF-IDF (Test Set):", X_test_tfidf.shape)
print(X_train_tfidf[0])  # In mẫu đầu tiên

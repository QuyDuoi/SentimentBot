# Sentiment Analysis with TF-IDF and BERT

## Giới thiệu
Dự án này sử dụng mô hình phân tích cảm xúc dựa trên TF-IDF và BERT để phân loại cảm xúc của văn bản. Dữ liệu đầu vào là từ file CSV chứa các tweet có nhãn cảm xúc.

## Lưu ý:
- Để sử dụng thư viện tensorflow thì cần sử dụng Python phiên bản từ 3.8 đến 3.9.
- Do mô hình huấn luyện cần xử lý nhiều tác vụ nên ưu tiên sử dụng GPU thay vì CPU. Có thể sử dụng Google Colab (Có hỗ trợ GPU).

## Yêu cầu
Để chạy dự án này, bạn cần cài đặt các thư viện sau:
```bash
pip install pandas scikit-learn nltk tensorflow torch transformers joblib scipy

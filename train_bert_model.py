import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from joblib import dump

# Chuyển đổi thành tập dữ liệu torch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    # Tải và tiền xử lý dữ liệu
    data_path = 'data/Sentiment140.csv'
    df = pd.read_csv(data_path, encoding='ISO-8859-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df['sentiment'] = df['sentiment'].replace(4, 2)  # Chuyển nhãn tích cực
    df = df[['sentiment', 'text']]
    df['label'] = df['sentiment'].apply(lambda x: 0 if x == 0 else 1)  # nhãn nhị phân

    # Chia dữ liệu
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # Tokenize văn bản
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    val_dataset = SentimentDataset(val_encodings, val_labels)

    # Tải mô hình BERT
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Định nghĩa các chỉ số đánh giá
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Các tham số huấn luyện
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        dataloader_num_workers=4
    )

    # Huấn luyện mô hình
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Lưu mô hình
    model.save_pretrained('./sentiment_model')
    tokenizer.save_pretrained('./sentiment_model')
    print("Model and tokenizer saved to './sentiment_model'")

if __name__ == '__main__':
    main()

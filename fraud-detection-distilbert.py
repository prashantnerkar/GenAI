pip install torch transformers scikit-learn


from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import pandas as pd

# Load pre-trained DistilBERT
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Load example dataset (replace with your fintech/healthcare dataset)
# Sample structure: {'text': [description], 'label': [0 for non-fraud, 1 for fraud]}
data = pd.DataFrame({
    "text": [
        "Unauthorized transaction on your credit card",
        "Routine check-up note with no abnormalities",
        "Large withdrawal from an unusual location",
        "Patient shows irregular drug prescription history",
        "Normal transaction at a grocery store",
    ],
    "label": [1, 0, 1, 1, 0]
})

# Tokenize data
def tokenize_data(texts, labels):
    tokens = tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    tokens["labels"] = torch.tensor(labels.tolist())
    return tokens

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)
train_data = tokenize_data(train_texts, train_labels)
test_data = tokenize_data(test_texts, test_labels)

# Define a custom evaluation metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print("Test Results:", results)


# Example fintech and healthcare fraud scenarios
new_texts = [
    "Suspicious login attempt detected",
    "Patient billing shows duplicate charges",
    "Transaction flagged due to unusual merchant",
    "Routine lab test results normal",
]

# Tokenize new data
new_tokens = tokenizer(new_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Get predictions
predictions = model(**new_tokens)
predicted_labels = torch.argmax(predictions.logits, dim=1)

# Interpret results
for text, label in zip(new_texts, predicted_labels):
    print(f"Text: {text} -> Fraudulent: {bool(label)}")




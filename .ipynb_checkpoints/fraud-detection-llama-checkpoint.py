pip install torch transformers datasets peft accelerate scikit-learn

from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama-3.2-1b", use_fast=False)

# Example dataset
data = pd.DataFrame({
    "text": [
        "Unauthorized transaction on your credit card",
        "Routine check-up note with no abnormalities",
        "Large withdrawal from an unusual location",
        "Patient shows irregular drug prescription history",
        "Normal transaction at a grocery store",
    ],
    "label": [1, 0, 1, 1, 0]  # 1 for Fraud, 0 for Non-Fraud
})

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'], data['label'], test_size=0.2
)

# Tokenize text
def tokenize_texts(texts, labels):
    tokens = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    tokens["labels"] = torch.tensor(labels.tolist())
    return tokens

train_data = tokenize_texts(train_texts, train_labels)
test_data = tokenize_texts(test_texts, test_labels)

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer

# Load pre-trained LLaMA model
model = AutoModelForCausalLM.from_pretrained("meta-llama-3.2-1b", device_map="auto")

# Configure PEFT with LoRA
peft_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=5e-5,
    report_to="none"
)

# Define custom Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

# Train the model
trainer.train()


from sklearn.metrics import classification_report

# Predict on test data
test_texts_tokens = tokenizer(
    test_texts.tolist(),
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt"
).to("cuda")

outputs = peft_model.generate(**test_texts_tokens)
predictions = [int(output.argmax(dim=-1)) for output in outputs]

# Classification report
print(classification_report(test_labels, predictions))


new_texts = [
    "Suspicious login attempt detected",
    "Routine lab test results normal",
    "Transaction flagged due to unusual merchant"
]

new_tokens = tokenizer(new_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
outputs = peft_model.generate(**new_tokens)
predicted_labels = [int(output.argmax(dim=-1)) for output in outputs]

# Output results
for text, label in zip(new_texts, predicted_labels):
    print(f"Text: {text} -> Fraudulent: {bool(label)}")




import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Load data
train_df = pd.read_csv("../data/FINAL_TRAIN_DATA.csv")
train_df = train_df[['Content', 'sentiment']]
train_df['sentiment'] = train_df['sentiment'].apply(lambda x: 2 if x == "positive" else (0 if x == "negative" else 1))
X_train, X_val, y_train, y_val = train_test_split(train_df["Content"].values, train_df["sentiment"].values, test_size=0.1)

test_df = pd.read_csv("../data/FINAL_TEST_DATA.csv")
test_df = test_df[['Content', 'sentiment']]
test_df['sentiment'] = test_df['sentiment'].apply(lambda x: 2 if x == "positive" else (0 if x == "negative" else 1))
X_test, y_test = test_df["Content"].values, test_df["sentiment"].values

# Load tokenizer and model
model_path = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

# Tokenize data
X_train_tokenized = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val.tolist(), padding=True, truncation=True, max_length=512)
X_test_tokenized = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512)

# Prepare datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)
test_dataset = Dataset(X_test_tokenized, y_test)

# Training arguments
training_arguments = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    logging_dir='../logs',
    logging_steps=100,
    evaluation_strategy='steps',
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    output_dir="../output"
)

# Metrics callback
class MetricsCallback:
    def __init__(self, tokenizer, eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(labels, predictions, average='macro')
        
        accuracy = accuracy_score(labels, predictions)

        return {
            "Weighted Precision": precision_weighted,
            "Weighted Recall": recall_weighted,
            "Weighted F1": f1_weighted,
            "Macro Precision": precision_macro,
            "Macro Recall": recall_macro,
            "Macro F1": f1_macro,
            "Accuracy": accuracy
        }

from sklearn.model_selection import StratifiedKFold
from transformers import pipeline

# Use StratifiedKFold to split the data into 5 folds
NUM_FOLD = 1
skf = StratifiedKFold(n_splits=NUM_FOLD, shuffle=True, random_state=42)

# Initialize lists to store predictions and true labels for each fold
all_fold_predictions = []
all_fold_true_labels = []

# Iterate over the folds
for fold, (train_index, val_index) in enumerate(skf.split(train_df["Content"].values, train_df["sentiment"].values)):
    print(f"Fold {fold + 1}/5")
    X_train_fold, X_val_fold = train_df["Content"].values[train_index], train_df["Content"].values[val_index]
    y_train_fold, y_val_fold = train_df["sentiment"].values[train_index], train_df["sentiment"].values[val_index]

    # Tokenize data
    X_train_fold_tokenized = tokenizer(X_train_fold.tolist(), padding=True, truncation=True, max_length=512)
    X_val_fold_tokenized = tokenizer(X_val_fold.tolist(), padding=True, truncation=True, max_length=512)

    # Prepare datasets
    train_dataset_fold = Dataset(X_train_fold_tokenized, y_train_fold)
    val_dataset_fold = Dataset(X_val_fold_tokenized, y_val_fold)

    # Trainer for the fold
    trainer_fold = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset_fold,
        eval_dataset=val_dataset_fold,
        compute_metrics=MetricsCallback(tokenizer, val_dataset_fold).compute_metrics
    )

    # Fine-tune the model for the fold
    trainer_fold.train()
    test_results = trainer_fold.evaluate(test_dataset)
    print("Test Results:", test_results)

    # Save the model
    model_path = f"../roberta-5fold/checkpoint-{fold + 1}"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # # Load the best model checkpoint from this fold
    # model_checkpoint_path = f"../subtask2/checkpoint-{fold + 1}"
    # model_fold = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)

    # Tokenize test data
    X_test_tokenized = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=512)
    test_dataset = Dataset(X_test_tokenized, y_test)

    # Use the model to make predictions on the test set
    test_predictions = trainer_fold.predict(test_dataset)
    all_fold_predictions.append(test_predictions.predictions)
    all_fold_true_labels.append(y_test)

# Average the predictions from all folds
avg_predictions_prob = np.mean(all_fold_predictions, axis=0)
avg_predictions = np.argmax(avg_predictions_prob, axis=1)

# Evaluate the averaged predictions on the test set
weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_test, avg_predictions, average='weighted')
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_test, avg_predictions, average='macro')
accuracy = accuracy_score(y_test, avg_predictions)

# Print the evaluation metrics on the test set
print("\nTest Set Evaluation Metrics:")
print(f"Weighted Precision: {weighted_precision}")
print(f"Weighted Recall: {weighted_recall}")
print(f"Weighted F1: {weighted_f1}")
print(f"Macro Precision: {macro_precision}")
print(f"Macro Recall: {macro_recall}")
print(f"Macro F1: {macro_f1}")
print(f"Accuracy: {accuracy}")


test_df = pd.read_csv("../data/FINAL_TEST_DATA.csv")
test_df['prediction']=avg_predictions
test_df.to_csv(f"../data/OUTPUT_Roberta_{NUM_FOLD}fold.csv", index=False)
np.save(f"../data/average_prediction_roberta.npy", avg_predictions_prob)
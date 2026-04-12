from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
import random


DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_75agree"
MODEL_NAME = "ProsusAI/finbert"
MAX_LENGTH = 256
RANDOM_SEED = 42

BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # GPU seed


def tokenize_function(example, tokenizer):
    """
    Tokenize text with padding and truncation.
    """
    return tokenizer(
        example["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )


def evaluate_model(model, dataloader, device):
    """
    Evaluate model and return loss + metrics.
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():  # no gradients
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)  # class prediction

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, macro_f1, all_labels, all_predictions


def main():
    """
    Train and evaluate FinBERT model on dataset.
    """
    set_seed(RANDOM_SEED)

    print("Checking PyTorch / CUDA setup...")
    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
    full_dataset = dataset["train"]

    print("\nCreating train / validation / test split...")
    train_valid = full_dataset.train_test_split(
        test_size=0.2,
        seed=RANDOM_SEED,
        stratify_by_column="label"  # keep class balance
    )

    valid_test = train_valid["test"].train_test_split(
        test_size=0.5,
        seed=RANDOM_SEED,
        stratify_by_column="label"
    )

    train_dataset = train_valid["train"]
    valid_dataset = valid_test["train"]
    test_dataset = valid_test["test"]

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    valid_dataset = valid_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # rename for PyTorch model
    train_dataset = train_dataset.rename_column("label", "labels")
    valid_dataset = valid_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    columns_to_return = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns_to_return)
    valid_dataset.set_format(type="torch", columns=columns_to_return)
    test_dataset.set_format(type="torch", columns=columns_to_return)

    print("\nCreating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print("\nLoading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = -1.0
    best_model_state = None

    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()  # reset gradients

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()  # backprop
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, valid_loader, device)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val F1:     {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {
                k: v.cpu().clone()  # save best weights
                for k, v in model.state_dict().items()
            }

    print("\nLoading best model...")
    model.load_state_dict(best_model_state)
    model.to(device)

    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_model(model, test_loader, device)

    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1:  {test_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[LABEL_NAMES[i] for i in range(3)]
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    save_path = "finbert_torch_baseline"
    print(f"\nSaving model to {save_path} ...")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Saved successfully.")


if __name__ == "__main__":
    main()
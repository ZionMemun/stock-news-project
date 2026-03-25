import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW


# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "finbert_torch_baseline"   # model already trained on PhraseBank
CLEAN_DATA_PATH = "clean_dataset.csv"

MISTAKES_OUTPUT_PATH = "mistakes_on_clean_data_before_finetune.csv"
FINE_TUNED_MODEL_OUTPUT = "finbert_phrasebank_then_clean_data"

MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
RANDOM_SEED = 42

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# =========================================================
# SEED
# =========================================================
def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# DATA HELPERS
# =========================================================
def normalize_label(label):
    return str(label).strip().lower()


def load_clean_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load clean dataset with columns: sentiment, text
    """
    print(f"\nLoading clean dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = {"sentiment", "text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df["sentiment"] = df["sentiment"].apply(normalize_label)
    df["text"] = df["text"].astype(str).str.strip()

    df = df[df["sentiment"].isin(LABEL2ID.keys())].copy()
    df = df[df["text"] != ""].copy()

    df["label"] = df["sentiment"].map(LABEL2ID)

    print("Shape:", df.shape)
    print("Class distribution:")
    print(df["sentiment"].value_counts())

    return df.reset_index(drop=True)


def dataframe_to_hf_dataset(df: pd.DataFrame, tokenizer):
    """
    Convert pandas DataFrame to Hugging Face Dataset and tokenize it.
    """
    hf_dataset = Dataset.from_pandas(
        df[["text", "label"]].rename(columns={"label": "labels"})
    )

    hf_dataset = hf_dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )

    hf_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return hf_dataset


# =========================================================
# EVALUATION / PREDICTION
# =========================================================
def predict_dataframe(model, tokenizer, df: pd.DataFrame, device):
    """
    Predict labels for all rows in the dataframe.
    """
    model.eval()

    texts = df["text"].tolist()
    true_labels = df["label"].tolist()

    all_preds = []
    all_confidences = []
    all_probs = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start_idx:start_idx + BATCH_SIZE]

            encodings = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            confidences, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_confidences.extend(confidences.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    results_df = df.copy()
    results_df["true_label_id"] = true_labels
    results_df["pred_label_id"] = all_preds
    results_df["true_label"] = results_df["true_label_id"].map(ID2LABEL)
    results_df["pred_label"] = results_df["pred_label_id"].map(ID2LABEL)
    results_df["confidence"] = all_confidences
    results_df["correct"] = results_df["true_label_id"] == results_df["pred_label_id"]

    results_df["prob_negative"] = [p[0] for p in all_probs]
    results_df["prob_neutral"] = [p[1] for p in all_probs]
    results_df["prob_positive"] = [p[2] for p in all_probs]

    return results_df


def print_metrics(title: str, results_df: pd.DataFrame):
    """
    Print evaluation metrics.
    """
    y_true = results_df["true_label_id"].tolist()
    y_pred = results_df["pred_label_id"].tolist()

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=[ID2LABEL[i] for i in range(3)]
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, macro_f1


def save_mistakes(results_df: pd.DataFrame, output_path: str):
    """
    Save wrong predictions on clean data.
    """
    mistakes_df = results_df[~results_df["correct"]].copy()

    mistakes_df = mistakes_df[
        [
            "sentiment",
            "text",
            "true_label",
            "pred_label",
            "confidence",
            "prob_negative",
            "prob_neutral",
            "prob_positive",
        ]
    ]

    mistakes_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved mistakes to: {output_path}")
    print(f"Number of mistakes: {len(mistakes_df)}")


# =========================================================
# TRAINING
# =========================================================
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataloader.
    """
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, macro_f1


def fine_tune_on_clean_data(model, tokenizer, clean_df: pd.DataFrame, device):
    """
    Fine-tune the PhraseBank-trained model on clean_dataset.csv
    """
    print("\n===== SPLITTING CLEAN DATA =====")

    train_df, temp_df = train_test_split(
        clean_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=clean_df["label"]
    )

    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"]
    )

    print("Train size:", len(train_df))
    print("Validation size:", len(valid_df))
    print("Test size:", len(test_df))

    train_dataset = dataframe_to_hf_dataset(train_df, tokenizer)
    valid_dataset = dataframe_to_hf_dataset(valid_df, tokenizer)
    test_dataset = dataframe_to_hf_dataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_valid_f1 = -1.0
    best_model_state = None

    print("\n===== FINE-TUNING ON CLEAN DATA =====")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        valid_loss, valid_acc, valid_f1 = evaluate_model(model, valid_loader, device)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Valid loss: {valid_loss:.4f}")
        print(f"Valid accuracy: {valid_acc:.4f}")
        print(f"Valid macro F1: {valid_f1:.4f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print("\nLoading best model based on validation F1...")
    model.load_state_dict(best_model_state)
    model.to(device)

    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, device)

    print("\n===== FINAL TEST ON CLEAN DATA =====")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro F1: {test_f1:.4f}")

    return model


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)

    print("Checking PyTorch / CUDA setup...")
    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------------------
    # Load model that was already trained on PhraseBank
    # -----------------------------------------------------
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)

    print("Model device:", next(model.parameters()).device)

    # -----------------------------------------------------
    # Load clean dataset
    # -----------------------------------------------------
    clean_df = load_clean_dataset(CLEAN_DATA_PATH)

    # -----------------------------------------------------
    # Step 1-3: Evaluate loaded model on clean data
    # -----------------------------------------------------
    clean_results = predict_dataframe(model, tokenizer, clean_df, device)
    print_metrics("PHRASEBANK-TRAINED MODEL ON CLEAN DATA", clean_results)
    save_mistakes(clean_results, MISTAKES_OUTPUT_PATH)

    # -----------------------------------------------------
    # Step 4-5: Fine-tune on clean data and save model
    # -----------------------------------------------------
    model = fine_tune_on_clean_data(model, tokenizer, clean_df, device)

    os.makedirs(FINE_TUNED_MODEL_OUTPUT, exist_ok=True)
    model.save_pretrained(FINE_TUNED_MODEL_OUTPUT)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_OUTPUT)

    print(f"\nSaved fine-tuned model to: {FINE_TUNED_MODEL_OUTPUT}")
    print("\nDone. Stopped after step 5 as requested.")


if __name__ == "__main__":
    main()
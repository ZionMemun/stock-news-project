import os
import random
import copy
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
# CONFIG
# Model already trained on PhraseBank
MODEL_PATH = "finbert_torch_baseline"

# Our cleaned dataset
CLEAN_DATA_PATH = "clean_dataset.csv"

# Original PhraseBank dataset used for the baseline model
PHRASEBANK_DATASET_NAME = "takala/financial_phrasebank"
PHRASEBANK_DATASET_CONFIG = "sentences_75agree"

# Outputs
MISTAKES_OUTPUT_PATH = "mistakes_on_clean_data_before_finetune.csv"
FINE_TUNED_MODEL_OUTPUT = "finetuned_finbert_on_clean_data"

# Tokenization / training
MAX_LENGTH = 256
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1  # if OOM, set BATCH_SIZE=16 and this=2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 6
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2
MAX_GRAD_NORM = 1.0
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


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# DATA HELPERS
def normalize_label(label):
    """
    Normalize text labels to lowercase strings.
    """
    return str(label).strip().lower()


def load_clean_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load clean CSV with columns: sentiment, text
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
        df[["text", "label"]].rename(columns={"label": "labels"}),
        preserve_index=False
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


def load_phrasebank_test_dataset(tokenizer):
    """
    Recreate the PhraseBank split used in the original baseline training
    so we can evaluate before and after fine-tuning.

    Original setup used:
    - dataset: takala/financial_phrasebank
    - config: sentences_75agree
    - split seed: 42
    """
    print("\nLoading PhraseBank dataset for regression check...")
    dataset = load_dataset(
        PHRASEBANK_DATASET_NAME,
        PHRASEBANK_DATASET_CONFIG,
        trust_remote_code=True
    )
    full_dataset = dataset["train"]

    train_valid = full_dataset.train_test_split(
        test_size=0.2,
        seed=RANDOM_SEED,
        stratify_by_column="label"
    )

    valid_test = train_valid["test"].train_test_split(
        test_size=0.5,
        seed=RANDOM_SEED,
        stratify_by_column="label"
    )

    phrasebank_test_dataset = valid_test["test"]

    phrasebank_test_dataset = phrasebank_test_dataset.map(
        lambda batch: tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        ),
        batched=True
    )

    phrasebank_test_dataset = phrasebank_test_dataset.rename_column("label", "labels")
    phrasebank_test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    print("PhraseBank test size:", len(phrasebank_test_dataset))
    return phrasebank_test_dataset


# EVALUATION / PREDICTION
def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataloader.
    Returns loss, accuracy, macro_f1, y_true, y_pred.
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

    return avg_loss, accuracy, macro_f1, all_labels, all_predictions


def print_eval_summary(title: str, loss, acc, macro_f1, y_true=None, y_pred=None):
    """
    Print a consistent evaluation summary.
    """
    print(f"\n===== {title} =====")
    print(f"Loss:     {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    if y_true is not None and y_pred is not None:
        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=[ID2LABEL[i] for i in range(3)]
        ))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))


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


def print_metrics_from_results(title: str, results_df: pd.DataFrame):
    """
    Print metrics from prediction results dataframe.
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


# TRAINING
def fine_tune_on_clean_data(model, tokenizer, clean_df: pd.DataFrame, device):
    """
    Fine-tune the PhraseBank-trained model on clean_dataset.csv
    using train/validation/test split, scheduler, weight decay,
    gradient clipping, and early stopping.
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

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    num_update_steps_per_epoch = max(
        1, len(train_loader) // GRADIENT_ACCUMULATION_STEPS
    )
    total_training_steps = num_update_steps_per_epoch * NUM_EPOCHS
    num_warmup_steps = int(WARMUP_RATIO * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps
    )

    best_valid_f1 = -1.0
    best_model_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    print("\n===== FINE-TUNING ON CLEAN DATA =====")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Total training steps: {total_training_steps}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            total_train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS

            should_step = ((step + 1) % GRADIENT_ACCUMULATION_STEPS == 0) or ((step + 1) == len(train_loader))

            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        valid_loss, valid_acc, valid_f1, _, _ = evaluate_model(model, valid_loader, device)

        current_lr = scheduler.get_last_lr()[0]

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Valid loss: {valid_loss:.4f}")
        print(f"Valid accuracy: {valid_acc:.4f}")
        print(f"Valid macro F1: {valid_f1:.4f}")
        print(f"Current LR: {current_lr:.8f}")

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            print("New best model found.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Patience counter: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(best_model_state)
    model.to(device)

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate_model(model, test_loader, device)

    print_eval_summary(
        "FINAL TEST AFTER FINETUNING ON CLEAN DATA",
        test_loss,
        test_acc,
        test_f1,
        y_true,
        y_pred
    )

    return model


# MAIN
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

    # Load tokenizer and baseline model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)

    print("Model device:", next(model.parameters()).device)

    # Load our clean dataset
    clean_df = load_clean_dataset(CLEAN_DATA_PATH)

    # Evaluate baseline model on clean data
    clean_results = predict_dataframe(model, tokenizer, clean_df, device)
    print_metrics_from_results("PHRASEBANK-TRAINED MODEL ON CLEAN DATA", clean_results)
    save_mistakes(clean_results, MISTAKES_OUTPUT_PATH)

    # Evaluate baseline model on PhraseBank test set
    phrasebank_test_dataset = load_phrasebank_test_dataset(tokenizer)
    phrasebank_test_loader = DataLoader(
        phrasebank_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    pb_before_loss, pb_before_acc, pb_before_f1, pb_y_true_before, pb_y_pred_before = evaluate_model(
        model,
        phrasebank_test_loader,
        device
    )

    print_eval_summary(
        "PHRASEBANK TEST BEFORE FINETUNING",
        pb_before_loss,
        pb_before_acc,
        pb_before_f1,
        pb_y_true_before,
        pb_y_pred_before
    )

    # Fine-tune on clean data
    model = fine_tune_on_clean_data(model, tokenizer, clean_df, device)

    # Evaluate fine-tuned model on PhraseBank test set
    pb_after_loss, pb_after_acc, pb_after_f1, pb_y_true_after, pb_y_pred_after = evaluate_model(
        model,
        phrasebank_test_loader,
        device
    )

    print_eval_summary(
        "PHRASEBANK TEST AFTER FINETUNING",
        pb_after_loss,
        pb_after_acc,
        pb_after_f1,
        pb_y_true_after,
        pb_y_pred_after
    )

    print("\n===== PHRASEBANK REGRESSION CHECK =====")
    print(f"Accuracy before finetuning: {pb_before_acc:.4f}")
    print(f"Accuracy after finetuning:  {pb_after_acc:.4f}")
    print(f"Delta accuracy:             {pb_after_acc - pb_before_acc:+.4f}")
    print(f"Macro F1 before finetuning: {pb_before_f1:.4f}")
    print(f"Macro F1 after finetuning:  {pb_after_f1:.4f}")
    print(f"Delta Macro F1:             {pb_after_f1 - pb_before_f1:+.4f}")

    # Save fine-tuned model
    os.makedirs(FINE_TUNED_MODEL_OUTPUT, exist_ok=True)
    model.save_pretrained(FINE_TUNED_MODEL_OUTPUT)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_OUTPUT)

    print(f"\nSaved fine-tuned model to: {FINE_TUNED_MODEL_OUTPUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
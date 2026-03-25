from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd


DATASET_NAME = "takala/financial_phrasebank"
DATASET_CONFIG = "sentences_75agree"
MODEL_NAME = "ProsusAI/finbert"


def main():
    print("Loading dataset")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, trust_remote_code=True)
    full_dataset = dataset["train"]

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    lengths = []

    for example in full_dataset:
        tokens = tokenizer(
            example["sentence"],
            truncation=False,
            padding=False,
        )
        lengths.append(len(tokens["input_ids"]))

    lengths_series = pd.Series(lengths)

    print("\nToken Length Statistics: ")
    print(f"Number of samples: {len(lengths_series)}")
    print(f"Mean length: {lengths_series.mean():.2f}")
    print(f"Median length: {lengths_series.median():.2f}")
    print(f"Min length: {lengths_series.min()}")
    print(f"Max length: {lengths_series.max()}")

    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"P{p}: {lengths_series.quantile(p / 100):.2f}")

    print("\nTruncation Analysis: ")
    for max_len in [64, 128, 256, 384, 512]:
        num_truncated = (lengths_series > max_len).sum()
        pct_truncated = 100 * num_truncated / len(lengths_series)
        print(
            f"max_length={max_len}: "
            f"{num_truncated} samples truncated "
            f"({pct_truncated:.2f}%)"
        )


if __name__ == "__main__":
    main()
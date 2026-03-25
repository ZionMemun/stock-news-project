import pandas as pd


DATA_PATH = "dataset.csv"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\n===== SAMPLE ROWS =====")
    print(df.head(5))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== CLASS DISTRIBUTION =====")
    print(df["sentiment"].value_counts())

    print("\n===== TEXT LENGTH ANALYSIS =====")
    df["text_length"] = df["text"].astype(str).apply(len)

    print("\nBasic stats:")
    print(df["text_length"].describe())

    print("\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"P{p}: {df['text_length'].quantile(p/100):.2f}")

    print("\nShortest texts:")
    print(df.nsmallest(5, "text_length")[["text", "text_length"]])

    print("\nLongest texts:")
    print(df.nlargest(5, "text_length")[["text", "text_length"]])

    print("\n===== DUPLICATES =====")
    duplicates = df.duplicated(subset=["text"]).sum()
    print(f"Duplicate texts: {duplicates}")

    print("\n===== SHORT TEXT CHECK =====")
    short_texts = df[df["text_length"] < 20]
    print(f"Texts shorter than 20 chars: {len(short_texts)}")

    print("\nExamples of short texts:")
    print(short_texts.head(5)[["text", "text_length"]])

    print("\n===== CLEAN DATA SIZE ESTIMATE =====")
    df_clean = df.copy()
    df_clean = df_clean.dropna()
    df_clean = df_clean[df_clean["text"].str.len() > 20]
    df_clean = df_clean.drop_duplicates(subset=["text"])

    print("After cleaning:", df_clean.shape)

    print("\n===== CLASS DISTRIBUTION AFTER CLEANING =====")
    print(df_clean["sentiment"].value_counts())


if __name__ == "__main__":
    main()
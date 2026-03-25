import pandas as pd
import unicodedata


DATA_PATH = "dataset.csv"
OUTPUT_PATH = "clean_dataset.csv"
REMOVED_OUTPUT_PATH = "removed_rows.csv"


def normalize_spaces(text: str) -> str:
    """
    Replace multiple spaces/newlines/tabs with a single space.
    """
    return " ".join(str(text).split())


def has_control_or_private_chars(text: str) -> bool:
    """
    Return True if text contains Unicode control / private / unassigned chars.
    """
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in {"Cc", "Cf", "Cs", "Co", "Cn"} and ch not in {" ", "\t", "\n"}:
            return True
    return False


def looks_like_mojibake(text: str) -> bool:
    """
    Heuristic detection for corrupted / broken-encoding strings such as:
    '׀ׁƒ׀¿...'
    """
    if not text or text.strip() == "":
        return False

    text_len = max(len(text), 1)

    # Very common markers in broken text
    paseq_count = text.count("׀")
    replacement_char_count = text.count("\ufffd")

    # Latin-1 supplement garbage
    latin1_supp_count = sum(0x80 <= ord(ch) <= 0xFF for ch in text)

    # Too many combining marks usually indicates broken text
    combining_count = sum(unicodedata.category(ch) in {"Mn", "Mc", "Me"} for ch in text)

    # Suspicious ratios
    paseq_ratio = paseq_count / text_len
    latin1_ratio = latin1_supp_count / text_len
    combining_ratio = combining_count / text_len

    # Rules
    if replacement_char_count > 0:
        return True

    if paseq_count >= 3:
        return True

    if paseq_ratio > 0.08:
        return True

    if latin1_ratio > 0.15:
        return True

    if combining_ratio > 0.20:
        return True

    return False


def clean_text_for_analysis(text: str) -> str:
    """
    Normalize spaces and strip.
    """
    text = str(text)
    text = normalize_spaces(text)
    return text.strip()


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # ========================
    # BASIC VALIDATION
    # ========================
    required_cols = {"sentiment", "text"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print("\n===== BASIC INFO =====")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\n===== SAMPLE ROWS =====")
    print(df.head(5))

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    # ========================
    # NORMALIZE BEFORE EDA
    # ========================
    work_df = df.copy()

    # Keep original text for audit
    work_df["original_text"] = work_df["text"]

    # Clean text safely
    work_df["text"] = work_df["text"].fillna("").apply(clean_text_for_analysis)
    work_df["sentiment"] = work_df["sentiment"].astype(str).str.strip().str.lower()

    print("\n===== ORIGINAL CLASS DISTRIBUTION =====")
    print(work_df["sentiment"].value_counts(dropna=False))

    # ========================
    # TEXT LENGTH EDA
    # ========================
    work_df["text_length"] = work_df["text"].apply(len)

    print("\n===== TEXT LENGTH ANALYSIS =====")
    print(work_df["text_length"].describe())

    print("\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"P{p}: {work_df['text_length'].quantile(p / 100):.2f}")

    print("\nShortest texts:")
    print(work_df.nsmallest(5, "text_length")[["sentiment", "text", "text_length"]])

    print("\nLongest texts:")
    print(work_df.nlargest(5, "text_length")[["sentiment", "text", "text_length"]])

    duplicates = work_df.duplicated(subset=["text"]).sum()
    print("\n===== DUPLICATES =====")
    print(f"Duplicate texts: {duplicates}")

    short_texts = work_df[work_df["text_length"] < 20]
    print("\n===== SHORT TEXT CHECK =====")
    print(f"Texts shorter than 20 chars: {len(short_texts)}")
    print(short_texts.head(5)[["sentiment", "text", "text_length"]])

    # ========================
    # CORRUPTION CHECK
    # ========================
    work_df["has_control_private"] = work_df["text"].apply(has_control_or_private_chars)
    work_df["looks_mojibake"] = work_df["text"].apply(looks_like_mojibake)

    print("\n===== CORRUPTED TEXT CHECK =====")
    print("Rows with control/private characters:", int(work_df["has_control_private"].sum()))
    print("Rows that look like mojibake:", int(work_df["looks_mojibake"].sum()))

    if work_df["looks_mojibake"].sum() > 0:
        print("\nExamples of mojibake-like rows:")
        print(work_df[work_df["looks_mojibake"]].head(10)[["sentiment", "text"]])

    # ========================
    # CLEANING
    # ========================
    print("\n===== CLEANING =====")
    clean_df = work_df.copy()
    removed_parts = []

    # 1) Missing / empty sentiment
    print("\nRemoving missing/empty sentiment...")
    invalid_sentiment_mask = clean_df["sentiment"].isin(["", "nan", "none", "null"])
    removed_invalid_sentiment = clean_df[invalid_sentiment_mask].copy()
    removed_invalid_sentiment["remove_reason"] = "missing_or_invalid_sentiment"
    removed_parts.append(removed_invalid_sentiment)
    clean_df = clean_df[~invalid_sentiment_mask].copy()
    print("Shape after removing invalid sentiment:", clean_df.shape)

    # 2) Empty text
    print("\nRemoving empty text...")
    empty_text_mask = clean_df["text"].isin(["", "nan", "none", "null"])
    removed_empty_text = clean_df[empty_text_mask].copy()
    removed_empty_text["remove_reason"] = "empty_text"
    removed_parts.append(removed_empty_text)
    clean_df = clean_df[~empty_text_mask].copy()
    print("Shape after removing empty text:", clean_df.shape)

    # 3) Short texts
    print("\nRemoving short texts (<=20 chars)...")
    short_mask = clean_df["text"].str.len() <= 20
    removed_short = clean_df[short_mask].copy()
    removed_short["remove_reason"] = "short_text"
    removed_parts.append(removed_short)
    clean_df = clean_df[~short_mask].copy()
    print("Shape after removing short texts:", clean_df.shape)

    # 4) Control/private/unassigned chars
    print("\nRemoving texts with control/private/unassigned characters...")
    control_mask = clean_df["text"].apply(has_control_or_private_chars)
    removed_control = clean_df[control_mask].copy()
    removed_control["remove_reason"] = "control_or_private_unicode"
    removed_parts.append(removed_control)
    clean_df = clean_df[~control_mask].copy()
    print("Shape after removing control/private chars:", clean_df.shape)

    # 5) Mojibake / broken encoding
    print("\nRemoving mojibake / broken-encoding texts...")
    mojibake_mask = clean_df["text"].apply(looks_like_mojibake)
    removed_mojibake = clean_df[mojibake_mask].copy()
    removed_mojibake["remove_reason"] = "mojibake_or_broken_encoding"
    removed_parts.append(removed_mojibake)
    clean_df = clean_df[~mojibake_mask].copy()
    print("Shape after removing mojibake:", clean_df.shape)

    # 6) Duplicate texts
    print("\nRemoving duplicate texts...")
    duplicate_mask = clean_df.duplicated(subset=["text"], keep="first")
    removed_duplicates = clean_df[duplicate_mask].copy()
    removed_duplicates["remove_reason"] = "duplicate_text"
    removed_parts.append(removed_duplicates)
    clean_df = clean_df[~duplicate_mask].copy()
    print("Shape after removing duplicates:", clean_df.shape)

    # ========================
    # BUILD REMOVED DF
    # ========================
    removed_df = pd.concat(removed_parts, ignore_index=True)

    # ========================
    # EDA AFTER CLEANING
    # ========================
    print("\n===== CLEAN DATA INFO =====")
    print("Clean shape:", clean_df.shape)

    print("\n===== CLEAN CLASS DISTRIBUTION =====")
    print(clean_df["sentiment"].value_counts())

    clean_df["text_length"] = clean_df["text"].apply(len)

    print("\n===== CLEAN TEXT LENGTH ANALYSIS =====")
    print(clean_df["text_length"].describe())

    print("\nClean percentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"P{p}: {clean_df['text_length'].quantile(p / 100):.2f}")

    print("\nClean sample rows:")
    print(clean_df.head(5)[["sentiment", "text", "text_length"]])

    print("\n===== REMOVED ROWS SUMMARY =====")
    if len(removed_df) > 0:
        print(removed_df["remove_reason"].value_counts())
    else:
        print("No rows were removed.")

    # ========================
    # SAVE
    # ========================
    clean_df = clean_df[["sentiment", "text"]].reset_index(drop=True)
    removed_df = removed_df[["sentiment", "text", "remove_reason"]].reset_index(drop=True)

    clean_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    removed_df.to_csv(REMOVED_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved cleaned dataset to: {OUTPUT_PATH}")
    print(f"Saved removed rows to: {REMOVED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
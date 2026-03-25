from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "training" / "finetuned_finbert_on_clean_data"
MAX_LENGTH = 256

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


class SentimentPredictor:
    """
    Load the fine-tuned FinBERT model and predict sentiment
    for article title + summary text.
    """

    def __init__(self, model_path=MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def build_input_text(self, title, summary):
        """
        Build the model input text from title and summary.
        """
        title = str(title or "").strip()
        summary = str(summary or "").strip()

        if title and summary:
            return f"{title}. {summary}"

        return title or summary

    def predict(self, title, summary=""):
        """
        Predict sentiment label and confidence score.

        Returns:
            {
                "sentiment_label": "positive" / "neutral" / "negative",
                "sentiment_score": float
            }
        """
        text = self.build_input_text(title, summary)

        if not text:
            return {
                "sentiment_label": None,
                "sentiment_score": None,
            }

        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence_score = probabilities[0, predicted_class_id].item()

        return {
            "sentiment_label": ID2LABEL[predicted_class_id],
            "sentiment_score": round(confidence_score, 4),
        }


def main():
    predictor = SentimentPredictor()

    examples = [
        {
            "stock_symbol": "AAPL",
            "company_name": "Apple",
            "title": "Exclusive-EU digital rules should apply to Big Tech's smart TVs, broadcasters tell antitrust chief",
            "summary": "Google, Amazon, Apple and Samsung's smart TVs and virtual assistants should fall under the EU's toughest tech rules because of their growing market power."
        },
        {
            "stock_symbol": "MSFT",
            "company_name": "Microsoft",
            "title": "Leaders and Experts from Amazon Web Services, Google, Microsoft, NVIDIA, Meta, Dell, Applied Materials and AMD Headline Technology and Innovation Programming at CERAWeek",
            "summary": "Technology and innovation speakers will participate in the conference in Houston."
        },
        {
            "stock_symbol": "TSLA",
            "company_name": "Tesla",
            "title": "Here are Tesla's Top Competitors in 2026",
            "summary": "Tesla's competitive landscape in 2026 will be unusual."
        },
        {
            "stock_symbol": "NVDA",
            "company_name": "Nvidia",
            "title": "Prediction: Nvidia Will Reach a $10 Trillion Market Cap by 2028",
            "summary": "The chipmaker's growth will drive it to previously unseen heights."
        },
    ]

    for i, example in enumerate(examples, start=1):
        result = predictor.predict(
            title=example["title"],
            summary=example["summary"]
        )

        print(f"\nExample {i}")
        print(f"Stock: {example['stock_symbol']} ({example['company_name']})")
        print(f"Title: {example['title']}")
        print(f"Prediction: {result['sentiment_label']}")
        print(f"Confidence: {result['sentiment_score']}")


if __name__ == "__main__":
    main()
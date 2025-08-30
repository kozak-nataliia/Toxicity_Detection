import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

MODEL_ID = "Natalya11/roberta_toxicity_detection"

class ToxicModel:
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(os.environ.get("THRESHOLD", threshold))
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return 0.0, 0
        batch = self.tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        logits = self.model(**batch).logits
        if logits.shape[-1] == 1:
            prob = torch.sigmoid(logits[0, 0]).item()
        else:
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        label = int(prob >= self.threshold)
        return float(prob), label

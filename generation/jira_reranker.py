import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# === Logger ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 5):
        self.top_k = top_k
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        logger.info(f"🔁 Reranking top {len(docs)} results using {self.model_name}")
        inputs = self.tokenizer(
            [query] * len(docs),
            [doc["text"] for doc in docs],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze().tolist()

        if isinstance(scores, float):  # Handle batch size 1
            scores = [scores]

        # Add scores and sort
        for i, doc in enumerate(docs):
            doc["rerank_score"] = scores[i]

        sorted_docs = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
        logger.info("✅ Reranking complete")

        return sorted_docs[:self.top_k]

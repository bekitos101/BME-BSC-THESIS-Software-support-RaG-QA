import re
import torch
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Optional
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def clean_html(text: Optional[str]) -> str:
    if not text: return ""
    return re.sub(r'<[^>]+>', '', text).strip()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_dense_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return mean_pooling(outputs, inputs['attention_mask']).squeeze().tolist()

def generate_sparse_vector(text: str) -> Dict[str, list]:
    #ignore short tokens
    tokens = re.findall(r'\w{3,}', text.lower())  
    term_freq = Counter(tokens)
    vocab = {term: idx for idx, term in enumerate(set(tokens))}
    return {
        "indices": [vocab[term] for term in term_freq],
        "values": [float(freq) for freq in term_freq.values()]
    }
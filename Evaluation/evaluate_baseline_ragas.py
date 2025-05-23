import json
import pandas as pd
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()

INPUT_JSON = "evaluation_with_baseline_gpt.json"
OUTPUT_METRICS_JSON = "baseline_ragas_results.json"

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []
for item in data:
    if not item.get("baseline_answer") or not item.get("baseline_retrieved_chunks"):
        continue  

    records.append({
        "question": item["question"],
        "answer": item["baseline_answer"],
        "contexts": item["baseline_retrieved_chunks"], 
        "ground_truth": item.get("ground_truth", item["baseline_answer"]) 
    })

ragas_dataset = Dataset.from_list(records)

metrics = [answer_relevancy, faithfulness, context_precision, context_recall]
results = evaluate(ragas_dataset, metrics=metrics)

df = results.to_pandas()
df.to_json(OUTPUT_METRICS_JSON, orient="records", indent=2)

print("\n Average RAGAS Metrics for Baseline:")
print(df[["answer_relevancy", "faithfulness", "context_precision", "context_recall"]].mean())

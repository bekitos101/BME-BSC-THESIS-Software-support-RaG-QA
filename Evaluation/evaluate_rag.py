import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datasets import Dataset
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas import evaluate
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "..", "generation"))
sys.path.insert(0, parent_dir)

from chat_assistant import ChatAssistant

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError(" OPENAI_API_KEY not found in .env")

INPUT_FILE = "ragas_eval_comp.json"
OUTPUT_FILE = "ragas_eval_comp_metrics.json"
METRICS = [answer_relevancy, faithfulness, context_precision, context_recall]

SOURCE_MAP = {
    "User Manual": "cap_manual_v3",
    "Support Tickets": "jira_tickets_hybrid",
    "Jira Tickets": "jira_tickets_hybrid"
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

assistant = ChatAssistant(
    model_name="gpt-3.5-turbo",
    sources=list(SOURCE_MAP.values())
)

for i, item in enumerate(data):
    if "rag_answer" not in item or "rag_retrieved_chunks" not in item:
        readable_source = item["source"]
        source_key = SOURCE_MAP.get(readable_source)
        if not source_key:
            print(f"Unknown source '{readable_source}' in entry {i+1}")
            continue

        print(f"[{i+1}/{len(data)}] Generating RaG answer for: {item['question']} (source: {source_key})")
        response = assistant.ask(item["question"], source=source_key, return_chunks=True)

        if isinstance(response, dict):
            item["rag_answer"] = response.get("answer", "")
            item["rag_retrieved_chunks"] = response.get("chunks", [])
            print(" ", item["rag_answer"])
        else:
            print(" Unexpected response format.")
            item["rag_answer"] = None
            item["rag_retrieved_chunks"] = []

rag_dataset = Dataset.from_list([
    {
        "question": item["question"],
        "answer": item["rag_answer"],
        "ground_truth": item["ground_truth"],
        "contexts": item.get("rag_retrieved_chunks", [item["ground_truth"]])
    } for item in data
])

def safe_evaluate(dataset, label):
    print(f"\n🚀 Evaluating: {label}")
    try:
        results = evaluate(dataset, metrics=METRICS)
        print(f" {label} evaluation complete.")
        return results
    except Exception as e:
        print(f" Evaluation failed for {label}: {e}")
        traceback.print_exc()
        return None

rag_scores = safe_evaluate(rag_dataset, "RaG")

if rag_scores:
    for i, item in enumerate(data):
        item["rag_metrics"] = {
            "answer_relevancy": rag_scores["answer_relevancy"][i],
            "faithfulness": rag_scores["faithfulness"][i],
            "context_precision": rag_scores["context_precision"][i],
            "context_recall": rag_scores["context_recall"][i],
        }

        print(f"\nQ{i+1}: {item['question']}")
        print("  RaG Metrics:", item["rag_metrics"])

# === Save updated file ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\n Updated results saved to {OUTPUT_FILE}")

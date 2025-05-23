import json
from eval_utils import get_embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def cosine(a, b):
    return cosine_similarity([a], [b])[0][0]

def evaluate_retrieval_by_source(json_path, top_k=3, similarity_threshold=0.65, output_file="retrieval_eval_by_source.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Metrics container per source
    source_metrics = defaultdict(lambda: {
        "total": 0,
        "hit@1": 0,
        "hit@3": 0,
        "avg_sim_top1": [],
        "avg_sim_top3": []
    })

    for item in tqdm(data, desc="Evaluating retrieval"):
        query = item.get("question")
        chunks = item.get("rag_retrieved_chunks", [])
        source = item.get("source", "Unknown")

        if not query or not chunks:
            continue

        source_metrics[source]["total"] += 1
        query_emb = get_embedding(query)
        chunk_embs = [get_embedding(c) for c in chunks[:top_k]]
        sims = [cosine(query_emb, emb) for emb in chunk_embs]

        if sims:
            if sims[0] >= similarity_threshold:
                source_metrics[source]["hit@1"] += 1
            if max(sims) >= similarity_threshold:
                source_metrics[source]["hit@3"] += 1
            source_metrics[source]["avg_sim_top1"].append(sims[0])
            source_metrics[source]["avg_sim_top3"].append(np.mean(sims))

    # Prepare and print summary
    summary = {}
    print("\n Retrieval Evaluation by Source:")
    for source, metrics in source_metrics.items():
        total = metrics["total"]
        hit1 = metrics["hit@1"]
        hit3 = metrics["hit@3"]
        avg1 = np.mean(metrics["avg_sim_top1"]) if metrics["avg_sim_top1"] else 0
        avg3 = np.mean(metrics["avg_sim_top3"]) if metrics["avg_sim_top3"] else 0

        result = {
            "total": total,
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@1_ratio": hit1 / total if total else 0,
            "hit@3_ratio": hit3 / total if total else 0,
            "avg_sim_top1": round(avg1, 4),
            "avg_sim_top3": round(avg3, 4)
        }
        summary[source] = result

        print(f"\n Source: {source}")
        for k, v in result.items():
            print(f"{k}: {v}")

    # Save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n Saved retrieval metrics by source to {output_file}")
    return summary


# Run the function directly (no argparse)
if __name__ == "__main__":
    evaluate_retrieval_by_source("ragas_eval.json")

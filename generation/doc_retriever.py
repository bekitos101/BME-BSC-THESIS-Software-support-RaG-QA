# generation/retriever.py

import torch
import numpy as np
import json
import os
import logging

from qdrant_client import QdrantClient
from generation.query_embedding_utils  import get_embedding


# Setup logger 
logging.basicConfig(
    filename="retriever_debug.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load metadata once 
metadata_path = os.path.join(os.path.dirname(__file__), "data", "user_manual_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    data = json.load(f)

metadata_nodes = data["nodes"]


# Soft Hybrid Retriever
class SoftHybridRetriever:
    MIN_SIMILARITY_THRESHOLD = 0.55
    MARGIN_THRESHOLD = 0.04

    def __init__(self, collection_name="cap_manual_v3", host="localhost", port=6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.metadata = metadata_nodes

        self.section_hierarchy = {}
        self.section_data = {}
        self.section_embeddings = {}

        for node in self.metadata:
            section_id = node["id"]
            self.section_data[section_id] = node
            self.section_embeddings[section_id] = get_embedding(node["title"])
            parent = node.get("parent_id")
            if parent not in self.section_hierarchy:
                self.section_hierarchy[parent] = []
            self.section_hierarchy[parent].append(section_id)

    def retrieve(self, question, top_k=10, score_threshold=0.5):
        logger.info(f"🔍 Retrieval started for question: {question}")
        query_vector = get_embedding(question)

        if len(query_vector) != 768:
            return "Invalid query embedding", [], []

        # Step 1: Run full search (no filter)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=("default", query_vector),
            limit=top_k * 5,  # Search wide
            with_payload=True,
            with_vectors=True
        )

        if not results:
            logger.warning("❌ No search results at all.")
            return "No results", [], []

        # Step 2: Get hierarchy-based boosted IDs
        top_section = None
        related_ids = []
        boosted_ids = set()
        
        relevant_sections = self._find_relevant_sections(query_vector)
        if relevant_sections:
            top_section = relevant_sections[0]
            related_ids = self._get_related_sections(top_section["id"])
            boosted_ids = set([top_section["id"]] + related_ids)
            logger.info(f"🏷️ Boosted sections: {boosted_ids}")

        # Step 3: Filter and rerank results manually
        reranked = []
        query_emb = torch.nn.functional.normalize(torch.tensor(query_vector, dtype=torch.float), dim=0)

        for r in results:
            payload = r.payload
            text = payload.get("text", "").strip()
            section_id = str(payload.get("section_id"))

            if not text or "default" not in r.vector:
                continue

            if boosted_ids and section_id not in boosted_ids:
                continue  # Skip unboosted

            chunk_emb = torch.tensor(r.vector["default"], dtype=torch.float)
            chunk_emb = torch.nn.functional.normalize(chunk_emb, dim=0)

            sim_score = torch.nn.functional.cosine_similarity(query_emb, chunk_emb, dim=0).item()
            final_score = 0.6 * r.score + 0.4 * sim_score

            reranked.append({
                "text": text,
                "citation": f"Page {payload.get('page_start', 'N/A')} | Title: {payload.get('title', 'Unknown')}",
                "score": final_score
            })

        if not reranked:
            return "No relevant content found.", [], []

        reranked = sorted(reranked, key=lambda x: -x["score"])
        top_chunks = self._select_top_chunks(reranked)

        context = "\n\n".join([f"[{c['citation']}]\n{c['text']}" for c in top_chunks])
        citations = [c["citation"] for c in top_chunks]
        texts = [c["text"] for c in top_chunks]

        logger.info(f"✅ Selected {len(top_chunks)} chunks.")
        return context, citations, texts
    
        
    def _find_relevant_sections(self, query_vector, threshold=0.5):
        query_tensor = torch.tensor(query_vector)
        candidates = []
        for section_id, emb in self.section_embeddings.items():
            sim = torch.nn.functional.cosine_similarity(
                query_tensor, torch.tensor(emb), dim=0).item()
            if sim >= threshold:
                node = self.section_data[section_id]
                candidates.append({
                    "id": section_id,
                    "title": node["title"],
                    "score": sim,
                    "level": self._get_hierarchy_level(section_id)
                })
        return sorted(candidates, key=lambda x: (-x["score"], x["level"]))

    def _get_related_sections(self, section_id):
        parent = self.section_data[section_id].get("parent_id")
        children = self.section_hierarchy.get(section_id, [])
        return ([parent] if parent else []) + children

    def _get_hierarchy_level(self, section_id):
        level = 0
        current_id = section_id
        while current_id in self.section_data:
            parent = self.section_data[current_id].get("parent_id")
            if not parent:
                break
            level += 1
            current_id = parent
        return level

    def _rerank_and_filter_chunks(self, results, query_vector):
        query_emb = torch.tensor(query_vector, dtype=torch.float)
        query_emb = torch.nn.functional.normalize(query_emb, dim=0)

        chunks = []

        for r in results:
            payload = r.payload
            logger.debug(f"🧪 Payload text preview: {payload.get('text', '')[:80]}")
            text = str(payload.get("text", "")).strip()
            
            # Handle named vector response
            if not text or not r.vector or "default" not in r.vector:
                logger.warning(f"⚠️ Skipping result with empty text or missing vector: {payload}")
                continue

            # Extract the vector from the named dict
            vector_values = r.vector["default"]
            chunk_emb = torch.tensor(vector_values, dtype=torch.float)
            chunk_emb = torch.nn.functional.normalize(chunk_emb, dim=0)

            sim_score = torch.nn.functional.cosine_similarity(
                query_emb, chunk_emb, dim=0).item()
            
            citation = f"Page {payload.get('page_start', 'N/A')} | Title: {payload.get('title', 'Unknown Title')}"

            final_score = 0.6 * r.score + 0.4 * sim_score
            

            logger.debug(f"[{citation}] Qdrant: {r.score:.4f}, Rerank: {sim_score:.4f}, Final: {final_score:.4f}")

            chunks.append({
                "text": text,
                "citation": citation,
                "score": final_score
            })
        if not chunks:
            logger.error("No usable chunks found — retrieval returned empty or invalid text.")
            return "No relevant content found.", [], []

        # Use smart selection logic
        top_chunks = self._select_top_chunks(chunks)
        logger.info(f" Context ready: {sum(len(c['text']) for c in top_chunks)} chars, {len(top_chunks)} chunks")
        logger.info(f"Selected scores: {[round(c['score'], 4) for c in top_chunks]}")

        context = "\n\n".join([f"[{c['citation']}]\n{c['text']}" for c in top_chunks])
        citations = [c["citation"] for c in top_chunks]
        texts = [c["text"] for c in top_chunks]

        return context, citations, texts


    def _select_top_chunks(self, chunks):
        if not chunks:
            return []

        chunks = sorted(chunks, key=lambda x: -x["score"])
        top1 = chunks[0]
        score1 = top1["score"]

        if len(chunks) == 1:
            return [top1]

        score2 = chunks[1]["score"]
        margin = abs(score1 - score2)

        if score1 > 0.68:
            logger.info("Using only top chunk — very high score")
            return [top1]

        if margin <= 0.04 and len(chunks) >= 3:
            logger.info("Top 3 are close in score — using 3 chunks")
            return chunks[:3]

        logger.info("Default fallback — using top 2 chunks")
        return chunks[:2]

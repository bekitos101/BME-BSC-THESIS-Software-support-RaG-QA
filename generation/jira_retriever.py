# generation/jira_hybrid_retriever.py
from qdrant_client.models import SparseVector, Prefetch, FusionQuery, Fusion
import logging
from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector
import os
import sys 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from Indexing.Jira_indexing.indexing.utils import get_dense_embedding, generate_sparse_vector
from query_expander import QueryExpander
from typing import Optional, List, Dict
from jira_reranker import CrossEncoderReranker
logging.basicConfig(
    filename="jira_retriever.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class JiraHybridRetriever:
    def __init__(
        self,
        collection_name="jira_tickets_hybrid",
        host="localhost",
        port=6333,
        model_name="deepseek-r1:1.5b"
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.query_expander = QueryExpander(model_name=model_name)
        self.reranker = CrossEncoderReranker(top_k=5)

    def retrieve(self, question: str, top_k: int = 5, filters: Optional[models.Filter] = None):
        logger.info(f" Query received: {question}")
        # Generate dense and sparse vectors
        dense_vector = self.query_expander.expand_query_hyde(question)
        logger.info(f"Query expanded via Hyde") 
        sparse_vector = generate_sparse_vector(question)

        #Prepare prefetch for dense and sparse
        prefetch = [
            Prefetch(
                query=dense_vector,
                using="dense",  
                limit=top_k * 3
            ),
            Prefetch(
                query=SparseVector(**sparse_vector),
                using="sparse",  
                limit=top_k * 3
            )
        ]

        #Perform hybrid query using Fusion (RRF)
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=filters
        )

        logger.info(f"Qdrant returned {len(results.points)} results")
        if not results.points:
            logger.warning("No results from Qdrant hybrid search")
            return "No results found.", [], []

        #prepare docs for reranking 
        docs = []
        for hit in results.points:
            payload = hit.payload
            key = payload.get("key", "N/A")
            title = payload.get("title", "")
            desc = payload.get("description", "")
            last_comment = payload.get("last_comment", "")
            solution = payload.get("solution", "")
            text = f"{title}\n{desc}\n{last_comment}\n{solution}"
            docs.append({"text": text, "metadata": payload})

        #rerank docs 
        reranked_docs = self.reranker.rerank(question, docs)


        #  Format retrieved results
        chunks, citations, texts = [], [], []
        for doc in reranked_docs:
            payload = doc["metadata"]
            rerank_score = doc.get("rerank_score", None)
            key = payload.get("key", "N/A")
            status = payload.get("status", "N/A")
            desc = payload.get("description", "No description.")
            comments = payload.get("comment_count", 0)
            label = payload.get("labels", [])
            title = payload.get("title", "No title")
            last_comment = payload.get("last_comment", "")
            solution = payload.get("solution", "")
            link = f"https://eteamproject.internal.ericsson.com/browse/{key}"

            chunk_text = (
                f"[Ticket {key} | Title: {title} | Status: {status} | Comments: {comments} | Labels: {label} | Link: {link}]\n\n"
                f"Description:\n{desc}\n\n"
                f"Last Comment:\n{last_comment}\n\n"
                f"Solution:\n{solution}"
            )
            
        
            citation = f"Jira Ticket {key}"
            chunks.append(chunk_text)
            citations.append(citation)
            texts.append(chunk_text)

            logger.info(f" Ticket {key} | Reranked")
            logger.info(f" Ticket {key} | Reranked Score: {rerank_score}")


        logger.info(f" Final top {len(texts)} chunks selected")

        context = "\n\n".join(f"[{cit}]\n{text}" for cit, text in zip(citations, texts))
        return context, citations, texts
    
# multi_source_retriever.py
from typing import List, Tuple
import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from jira_retriever import JiraHybridRetriever
from doc_retriever import SoftHybridRetriever
from .routing_controller import RoutingController
import logging

# LOGGING SETUP 
log_path = os.path.join(os.path.dirname(__file__), "multisource_retriever.log")
logging.basicConfig(
    filename="multisource_retriever.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MultiSourceRetriever:
    def __init__(self, model_name: str = "deepseek-r1:1.5b"):
        self.routing = RoutingController()
        self.retrievers = {
            "cap_manual_v3": SoftHybridRetriever(collection_name="cap_manual_v3", host="localhost", port=6333),
            "jira_tickets_hybrid": JiraHybridRetriever(collection_name="jira_tickets_hybrid", host="localhost", port=6333, model_name=model_name),
        }
        logger.info("MultiSourceRetriever initialized with retrievers: %s", list(self.retrievers.keys()))
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[str, List[str], List[str]]:
        logger.info("Received query: %s", query)
        sources = self.routing.route(query)
        logger.info("Routing determined the sources: %s", sources)
        all_chunks, all_citations, all_texts = [], [], []

        for source in sources:
            retriever = self.retrievers.get(source)
            if retriever:
                logger.info("Using retriever for source: %s", source)
                context, citations, texts = retriever.retrieve(query, top_k=top_k)
                logger.info("Retrieved %d chunks from %s", len(texts), source)
                all_chunks.extend(context.split("\n\n"))
                all_citations.extend(citations)
                all_texts.extend(texts)
            else:
                logger.warning("No retriever found for source: %s", source)
        # Combine and return
        context = "\n\n".join(all_chunks)
        logger.info("Final combined context has %d paragraphs", len(all_chunks))
        return context, all_citations, all_texts

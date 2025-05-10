import os
import sys
import logging

# Ensure local directory is on the import path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from toc_parser import TOCParser
from docs_chunker import UserManualChunker
from doc_indexer import UserManualIndexer
from config import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def main():
    # Parse TOC and generate metadata JSON
    logger.info("Parsing TOC...")
    toc_parser = TOCParser("data/user_manual_toc.txt")
    toc_parser.parse()
    toc_parser.save_to_json("data/user_manual_metadata.json")

    #  Chunk and enrich PDF using metadata
    logger.info("Chunking and enriching PDF...")
    chunker = UserManualChunker(
        pdf_path="data/user_manual_cleaned.pdf",
        metadata_path="data/user_manual_metadata.json"
    )
    chunker.run(output_path="data/enriched_sections_clean.json")

    #  Index chunks into Qdrant
    logger.info("Embedding and indexing enriched sections...")
    indexer = UserManualIndexer(data_path="data/enriched_sections_clean.json")
    indexer.run()

    logger.info("Done — document indexing pipeline complete.")


if __name__ == "__main__":
    main()

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, VectorParams, Distance
from parsers import parse_jira_xml
from indexing.utils import get_dense_embedding, generate_sparse_vector, clean_html
import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indexing.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
import pandas as pd

EXPORT_PATH= "SearchRequest.xml" 

def prepare_document(ticket: dict) -> str:
    """Combine key fields for embedding"""
    text_parts = [
        f"Title: {ticket['title']}",
        f"Summary: {ticket['summary']}",
        f"Description: {ticket['description']}",
        f"Status: {ticket['status']}",
        f"Resolution: {ticket['resolution'] or 'Unresolved'}",
    ]

    if ticket["comments"]:
        text_parts.append("Comments:")
        text_parts.extend(
            f"- {c['author']} on {c['created']}: {c['text']}" for c in ticket["comments"]
        )

    if ticket.get("last_comment"):
        text_parts.append(f"Last Comment: {ticket['last_comment']}")

    if ticket.get("solution"):
        text_parts.append(f"Solution: {ticket['solution']}")

    return "\n".join(text_parts)


def index_tickets(xml_path: str):
    
    # Parse XML
    tickets = parse_jira_xml(xml_path)
    
    # Initialize Qdrant
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    # Create collection with hybrid support for both dense and sparse vectors
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=768, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    
    # Prepare and index points
    points = []
    for idx, ticket in enumerate(tickets):
        # Prepare text
        document_text = prepare_document(ticket)
        
        # Generate vectors
        dense_vector = get_dense_embedding(document_text)
        sparse_vector = generate_sparse_vector(document_text)
        
        # Prepare metadata
        metadata = {
            "key": ticket["key"],
            "title": ticket["title"],
            "status": ticket["status"],
            "resolution": ticket["resolution"],
            "priority": ticket["priority"],
            "created": ticket["created"],
            "updated": ticket["updated"],
            "has_attachments": len(ticket["attachments"]) > 0,
            "comment_count": len(ticket["comments"]),
            "labels": ticket["labels"],
            "description": ticket["description"],
            "last_comment": ticket.get("last_comment", ""),
            "solution": ticket.get("solution", ""),
            "link": f"https://eteamproject.internal.ericsson.com/browse/{ticket['key']}",
        }        
        points.append(
            PointStruct(
                id=idx,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector
                },
                payload=metadata
            )
        )
        
        # Batch insert every 100 tickets
        if len(points) % 100 == 0:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            print(f"Indexed {idx+1} tickets...")
    
    # Insert remaining points
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    
    print(f"Successfully indexed {len(tickets)} tickets")

if __name__ == "__main__":
    index_tickets(EXPORT_PATH)
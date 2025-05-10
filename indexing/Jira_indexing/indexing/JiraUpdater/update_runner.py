#update_runner.py
import os
import sys
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
from parsers import parse_jira_xml
from indexing.utils import get_dense_embedding, generate_sparse_vector
from indexing.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME
from JiraUpdater.rss_downloader import fetch_jira_rss
from indexing.JiraUpdater.updater_config import XML_URL, SESSION_ID, XML_FILE 
import logging

# Setup logging
logging.basicConfig(
    filename="jira_update.log",
    filemode="a", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)



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

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %z").astimezone().replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return datetime(2000, 1, 1)

def run():
    # Fetch XML from JIRA
    try:
        fetch_jira_rss(XML_URL, SESSION_ID, save_path=XML_FILE)
        logger.info("XML fetched successfully.")
    except Exception as e:
        logger.error(f"Failed to fetch XML: {e}")
        return

    # Parse XML into ticket dicts
    tickets = parse_jira_xml(XML_FILE)
    logger.info(f"Parsed {len(tickets)} tickets from XML.")


    # Load existing ticket update times
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    existing = {
        pt.payload["key"]: pt
        for pt in client.scroll(collection_name=COLLECTION_NAME, limit=100_000)[0]
        if "key" in pt.payload
    }

    # Compare and prepare updated points
    points = []
    for ticket in tickets:
        existing_point = existing.get(ticket["key"])
        current_time = parse_date(ticket["updated"])
        stored_time = parse_date(existing_point.payload.get("updated", "2000-01-01")) if existing_point else None

        if existing_point and current_time <= stored_time:
            continue  

        document_text = prepare_document(ticket)
        dense_vector = get_dense_embedding(document_text)
        sparse_vector = generate_sparse_vector(document_text)

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

        points.append(PointStruct(
            id=ticket["key"],
            vector={"dense": dense_vector, "sparse": sparse_vector},
            payload=metadata
        ))

    # Upsert updated tickets
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Upserted {len(points)} updated tickets.")
    else:
        print("No new or updated tickets found.")

if __name__ == "__main__":
    run()

import json
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import get_embedding
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME

class UserManualIndexer:
    def __init__(self, data_path: str, vector_size: int = 768):
        self.data_path = data_path
        self.vector_size = vector_size
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.collection = COLLECTION_NAME

    def recreate_collection(self):
        print(f"Recreating collection: {self.collection}")
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
        )
        print(f" Collection `{self.collection}` created.")

    def load_sections(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)['nodes']

    def build_points(self, sections):
        points = []
        for node in sections:
            vector = get_embedding(node['text'])
            payload = {
                "id": node.get("id"),
                "section_id": node.get("id"),
                "title": node.get("title"),
                "type": node.get("type"),
                "page_start": node.get("page_start"),
                "parent_id": node.get("parent_id"),
                "children_ids": node.get("children_ids", []),
                "text": node.get("text"),
            }
            points.append(PointStruct(id=str(uuid4()), vector=vector, payload=payload))
        return points

    def upsert_points(self, points):
        self.client.upsert(collection_name=self.collection, points=points)
        print(f" Indexed {len(points)} sections into `{self.collection}`.")

    def run(self):
        self.recreate_collection()
        sections = self.load_sections()
        points = self.build_points(sections)
        self.upsert_points(points)
        

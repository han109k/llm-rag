import os

import requests
from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")
collection_name = "univerlist_rag"

snapshot_name = os.path.basename("./snapshots/univerlist_rag.snapshot")
print("snapshot_name:", snapshot_name)

try:
    response = requests.post(
        "http://localhost:6333/collections/univerlist_rag/snapshots/upload?priority=snapshot",
        headers={
            "api-key": "",
        },
        files={
            "snapshot": (
                snapshot_name,
                open("./snapshots/univerlist_rag.snapshot", "rb"),
            )
        },
        timeout=30,
    )
    print("Snapshot upload response:", response.json())
except Exception as e:
    print("Snapshot upload failed:", str(e))

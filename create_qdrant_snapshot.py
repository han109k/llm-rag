import os

import requests
from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")
collection_name = "univerlist_rag"

snapshot_info = client.create_snapshot(collection_name=collection_name)

snapshot_url = f"http://localhost:6333/collections/{collection_name}/snapshots/{snapshot_info.name}"

# Create a directory to store snapshots
os.makedirs("snapshots", exist_ok=True)

snapshot_name = os.path.basename(snapshot_url)
local_snapshot_path = os.path.join("snapshots", snapshot_name)

response = requests.get(snapshot_url, headers={"api-key": ""}, timeout=30)
with open(local_snapshot_path, "wb") as f:
    response.raise_for_status()
    f.write(response.content)

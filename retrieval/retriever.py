from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

embedder = SentenceTransformer(EMBEDDING_MODEL)

def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_vector = embedder.encode(query).tolist()

    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in result["matches"]]
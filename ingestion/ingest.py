from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *

# ----------------------------------
# Initialize Pinecone (NEW SDK)
# ----------------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
try:
    existing_indexes = pc.list_indexes()
    index_names = [idx.name for idx in existing_indexes.indexes]
    
    if PINECONE_INDEX not in index_names:
        print(f"Creating index {PINECONE_INDEX}...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=PINECONE_ENV
            )
        )
        print(f"✅ Index {PINECONE_INDEX} created successfully")
except Exception as e:
    print(f"❌ Index creation error - {type(e).__name__}: {e}")
    sys.exit(1)

index = pc.Index(PINECONE_INDEX)

# ----------------------------------
# Embeddings
# ----------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

def embed(text: str):
    return embeddings.embed_query(text)

# ----------------------------------
# Load CSV
# ----------------------------------
import os
csv_path = os.path.join(os.path.dirname(__file__), "..", "indian_legal_dataset.csv")
df = pd.read_csv(csv_path)

# ----------------------------------
# Prepare vectors
# ----------------------------------
vectors = []

for i, row in df.iterrows():
    vectors.append({
        "id": f"doc-{i}",
        "values": embed(row["Content"]),
        "metadata": {
            "category": row["Category"],
            "subcategory": row["Subcategory"],
            "role": row["Role"],
            "text": row["Content"]
        }
    })

# ----------------------------------
# Upsert (batch-safe)
# ----------------------------------
index.upsert(vectors=vectors)

print(f"✅ Successfully ingested {len(vectors)} legal records into Pinecone")
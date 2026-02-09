from groq import Groq
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import *

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query: str, context_chunks: list[str]) -> str:
    context = "\n".join(context_chunks)

    prompt = f"""
You are a factual AI assistant.
Answer ONLY using the context provided.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content
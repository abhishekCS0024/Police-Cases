from retrieval.retriever import retrieve
from generation.generator import generate_answer

def rag(query: str):
    context = retrieve(query)
    return generate_answer(query, context)
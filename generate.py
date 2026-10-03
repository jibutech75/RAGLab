# generate.py
import ollama

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n---\n".join(context_chunks)
    #return f"""You are an assistant answering questions about an invoice document.
#Use ONLY the context below to answer. If the answer isn't in the context, say so.
    return f"""You are a helpful assistant. Use the context below if it's relevant to the question.
If the context doesn't contain the answer, use your own general knowledge instead, and mention that the answer isn't from the document.

Context:
{context}

Question: {query}

Answer:"""

def generate_answer(query: str, context_chunks: list[str]) -> str:
    prompt = build_prompt(query, context_chunks)
    response = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]



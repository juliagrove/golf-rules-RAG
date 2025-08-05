import json
import os

import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FAISS_FOLDER, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-2.0-flash")
embedder = SentenceTransformer(EMBEDDING_MODEL)


def load_index(faiss_dir=FAISS_FOLDER):
    faiss_file = os.path.join(faiss_dir, "faiss_index.idx")
    id_map_file = os.path.join(faiss_dir, "id_map.json")
    index = faiss.read_index(faiss_file)

    with open(id_map_file, "r") as file:
        id_map = json.load(file)

    return index, id_map


def retrieve_chunks(question, faiss_index, id_map, top_k=3):
    # converts the users question to an embedding vector
    query_emb = embedder.encode([question]).astype("float32")

    # returns similarity scores(not used) and a 2D array of the top matching vector id, shape: [# of questions, top_k]
    _, top_ids = faiss_index.search(query_emb, top_k)

    # gets the actual matching chunks of text in the id map using the corresponding vector ids
    return [id_map[str(i)] for i in top_ids[0]]


def prompt_llm(chunks, question):
    # reading json in data/chunked_output/
    context = "\n".join([chunk.get("text") for chunk in chunks])

    prompt = f"""
        You are an expert in the rules of golf. Answer the question using
        ONLY the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Instructions: 
            - Base your answer only on the provided context
            - If the context doesn't fully answer the question, respond
            with "The provided rules do not specify this"
            - Be clear and concise
    """

    response = llm.generate_content(prompt)
    return response.text

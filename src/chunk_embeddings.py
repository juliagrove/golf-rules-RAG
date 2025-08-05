import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKED_FOLDER,
    EMBEDDING_MODEL,
    FAISS_FOLDER,
    SCRAPED_FOLDER,
)


embedder = SentenceTransformer(EMBEDDING_MODEL)


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def save_chunks(input=SCRAPED_FOLDER, output_folder=CHUNKED_FOLDER):

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input):
        file_name = os.path.splitext(file)[0]
        file_path = os.path.join(input, file)

        with open(file_path) as f:
            pages = json.load(f)

        full_doc = " ".join(page.get("text") for page in pages)
        chunks = create_chunks(full_doc)
        output = os.path.join(output_folder, f"{file_name}_chunks.jsonl")

        with open(output, "w") as f:
            for i, chunk in enumerate(chunks):
                json.dump({"id": f"{file_name}_chunk_{i}", "text": chunk}, f)
                f.write("\n")  # .jsonl file
        print("Chunked and saved")


def build_index(
    chunk_folder=CHUNKED_FOLDER, faiss_output=FAISS_FOLDER, create_chunks=False
):

    faiss_file = os.path.join(faiss_output, "faiss_index.idx")
    id_map_file = os.path.join(faiss_output, "id_map.json")

    # True when changing chunk size, chunk overlap or initial file creation
    if create_chunks:
        save_chunks()

    chunks = []

    os.makedirs(faiss_output, exist_ok=True)

    for file in os.listdir(chunk_folder):
        with open(os.path.join(chunk_folder, file), "r") as f:
            for line in f:
                chunks.append(json.loads(line))

    texts = [chunk.get("text") for chunk in chunks]
    ids = [chunk.get("id") for chunk in chunks]

    embeddings = np.array(embedder.encode(texts)).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, faiss_file)
    with open(id_map_file, "w") as file:
        json.dump(
            {i: {"id": ids[i], "text": texts[i]} for i in range(len(texts))}, file
        )
    print(f"FAISS index and ID map saved to {faiss_output}")


if __name__ == "__main__":
    build_index(create_chunks=True)

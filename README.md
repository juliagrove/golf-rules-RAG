# Golf Rules RAG Pipeline

This project demonstrates the use of **Retrieval Augmented Generation (RAG)** to answer questions about the rules of golf. The pipeline retrieves the USGA official rules and uses a Large Language Model (LLM) to generate natural language answers.

---

## Workflow

1. **Data Acquisition**  
   - Collected and scraped USGA Golf Rules

2. **Chunking**  
   - Split JSON text into chunks for efficient retrieval

3. **Embedding & Indexing**  
   - Used vector embeddings and stored them in **FAISS index**.

4. **Retrieval**  
   - Queried the FAISS index to retrieve the most relevant context for each user question.

5. **LLM Generation**  
   - Sent retrieved context to **Gemini LLM API** to produce a natural language response.

---

## Instructions to Run the RAG Pipeline

1. **Set up your Environment**
   - Create and activate the conda env: 

   ```bash
   cd to/project/folder

   conda env create -f environment.yml
   
   conda activate golf-rag
   ```
   - Obtain a Gemini API key
   - Create a `.env` file in the root directory with: `GEMINI_API_KEY = "YOUR-API-KEY"`

2. **Download the PDF**
   - Download the [official USGA rules of golf](https://www.usga.org/content/dam/usga/images/rules/rules-modernization/golf-new-rules/Rules%20of%20Golf%20for%202019%20(Final).pdf) pdf and place it in `data/pdf`
3. **Scrape the PDF**
   - Run `scrape.py` to obtain the text from the PDF

4. **Create the Chunks and Embeddings**
   - Run `chunk_embeddings.py` to create the FAISS Index and metadata

5. **Run Pipeline**
   - Run `main.py` and enter a question

---

## Example Usage

```bash
python src/scrape.py
python src/chunk_embeddings.py
python src/main.py
```

**Input:**  
```
Enter a question: What happens if you swing and miss off the tee?
```

**Output:**  
```
Answer: If the player’s ball in play is in the teeing area after a stroke (such as a teed ball after a stroke that missed the ball) the player may lift or move the ball without penalty, and play that ball or another ball from anywhere in the teeing area from a tee or the ground.
```

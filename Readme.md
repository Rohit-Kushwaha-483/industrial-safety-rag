# Industrial Safety RAG Flask App

This project implements a Retrieval-Augmented Generation (RAG) application using LangChain with:

- PDF document ingestion
- Vector embeddings via HuggingFace
- Similarity search with Chroma vector store
- Question answering via Google Gemini LLM
- Reranking using Cohere API
- Flask web app with a simple HTML frontend

## Usage

1. Load your PDFs into `industrial-safety-pdfs/`
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for APIs in `.env`
4. Run the Flask app: `python app.py`
5. Visit `http://localhost:5000` to ask questions about the documents.

---

## API Keys Required

- Google API Key (`GOOGLE_API_KEY`)
- Cohere API Key (`COHERE_API_KEY`)

---
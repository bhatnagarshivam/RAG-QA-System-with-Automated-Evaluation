# PDF RAG Assistant

A Streamlit RAG app for PDF question answering. It uses LangChain `PyPDFLoader`, page-level Tesseract OCR for scanned pages, `RecursiveCharacterTextSplitter`, `sentence-transformers/all-MiniLM-L6-v2` embeddings, Chroma, and GPT (`gpt-4o-mini`) for both reranking and final answers.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Install Tesseract OCR on your operating system and make sure `tesseract` is on `PATH`. Add an API key in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

Then run from the repository root:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, then select the repository, branch, and `app.py`.
3. In the app's **Settings → Secrets**, add `OPENAI_API_KEY = "your-openai-api-key"`.
4. Deploy. Community Cloud installs Python packages from `requirements.txt` and Tesseract from `packages.txt`.

The Chroma database is stored in `chroma_db/` on the app's local filesystem. Community Cloud storage is ephemeral, so indexed PDFs and vectors can disappear when the app restarts. Users should re-upload and re-index documents after a restart; use hosted persistent storage if the knowledge base must survive restarts or be shared across users.

## Notes

- GPT is used as an LLM-based reranker over the top vector-search candidates and to generate grounded responses.
- Uploaded documents and questions are sent to the configured OpenAI API when answering; avoid uploading sensitive material unless your data policy permits it.
- The first run downloads the MiniLM model. CPU inference and OCR can be slow for large PDFs on the free Community Cloud tier.
"# rag" 

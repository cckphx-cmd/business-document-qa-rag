# Business Document Q&A — Evaluation System

This evaluation proves your RAG pipeline works on real business documents (contracts, reports, proposals) and gives simple, quantitative metrics you can show in your Sunday presentation.

## What it Does

- Loads or builds a FAISS vector index from your documents  
  - By default, it tries to **load** `./faiss_index`  
  - If missing, it **builds** an index from `./docs/*.txt`  
  - If `./docs` is empty, it seeds two tiny demo business docs so the system always runs end-to-end
- Asks **20 test questions** from `evaluation_questions.json` (10 factual, 5 analytical, 5 summary)
- Runs retrieval-augmented Q&A using `gpt-3.5-turbo`
- Records for each question:
  - **Response time (s)**
  - **Answer length (words)**
  - **Keyword match ratio** (fraction of expected keywords found in the answer)
- Saves results to `results.csv` and prints a summary
- Optional: displays an interactive table if you launch it with Streamlit

---

## Quick Start (Copy/Paste)

From the project root:

```bash
# 1) Install dependencies
pip install -q langchain_openai langchain_community faiss-cpu python-dotenv streamlit

# 2) Add your OpenAI key
echo "OPENAI_API_KEY=sk-yourkey" > .env

# 3) (Optional) Put your business docs here as plain text:
mkdir -p docs
# Example: drop .txt files into ./docs

# 4) Run the evaluation (CLI)
python evaluate.py

# 5) Or run with a simple UI (Streamlit)
streamlit run evaluate.py

"""
EVALUATION RUNNER FOR BUSINESS DOCUMENT Q&A (RAG) SYSTEM
- Vector store: FAISS
- Model: OpenAI GPT-3.5-turbo via langchain_openai
- Libraries allowed: langchain_openai, langchain_community, faiss-cpu, python-dotenv, streamlit

WHAT THIS SCRIPT DOES
1) Loads or builds a FAISS vector store:
   - Tries to load from ./faiss_index
   - If not found, it auto-builds a tiny demo index from ./docs/*.txt
     (If no files found, it seeds two tiny built-in business docs so it always runs.)

2) Loads 20 evaluation questions from ./evaluation_questions.json

3) Runs retrieval-augmented Q&A for each question and records:
   - Response time (seconds) per question
   - Response length (words)
   - Keyword match score = fraction of expected keywords found in the answer (case-insensitive, stem-ish)

4) Saves a detailed CSV: ./results.csv
   Columns: id, type, question, answer, response_time_s, answer_word_count, keyword_match_ratio

5) Prints a short summary. If launched with Streamlit, also shows an interactive table.

USAGE (terminal):
  python evaluate.py
OR with Streamlit UI:
  streamlit run evaluate.py

REQUIREMENTS:
  pip install langchain_openai langchain_community faiss-cpu python-dotenv streamlit
  Create .env with: OPENAI_API_KEY=sk-...
"""

import os
import json
import time
import csv
from typing import List, Dict, Any

# Allowed libraries (and their LangChain core dependency)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional: Streamlit (allowed)
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()
# ----------------------------
# CONFIGURATION (edit if needed)
# ----------------------------
VECTORSTORE_DIR = "./faiss_index"   # Where FAISS index will be saved/loaded
DOCS_DIR = "./docs"                 # If index missing, *.txt files here are ingested
QUESTIONS_PATH = "./evaluation_questions.json"
RESULTS_CSV = "./results.csv"

# Retrieval parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4  # number of chunks to retrieve


def ensure_api_key() -> None:
    """Load .env and ensure OPENAI_API_KEY is available."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Create a .env file in this folder with:\n"
        )


def seed_demo_texts_if_needed() -> List[str]:
    """
    If ./docs has no .txt files, provide two small in-memory demo documents
    so the pipeline can still run end-to-end.
    """
    texts = []
    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)

    txt_files = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".txt")]
    if txt_files:
        return texts  # no need to seed; user already has files

    # Seed two tiny business-like docs to guarantee a working demo
    demo1 = (
        "MASTER SERVICES AGREEMENT (MSA)\n"
        "Effective Date: January 1, 2024\n"
        "Parties: Acme Corp (Vendor) and Example LLC (Client)\n"
        "Scope of Work: Vendor will provide data analytics services and monthly reports.\n"
        "Payment Terms: Net 30 days from invoice date. Late payments incur 1.5% monthly interest.\n"
        "Acceptance Criteria: Reports must include KPIs and pass accuracy checks.\n"
        "Termination: Either party may terminate with 30 days written notice.\n"
        "Governing Law: State of Arizona.\n"
        "Confidentiality: Both parties agree to keep information confidential."
    )
    demo2 = (
        "STATEMENT OF WORK (SOW)\n"
        "Project Timeline: Phase 1 (Jan-Feb), Phase 2 (Mar-Apr), Final Delivery: April 30, 2024.\n"
        "Deliverables: KPI dashboard, monthly summary, and executive presentation.\n"
        "Budget: Total value $50,000 (fixed fee). Invoices issued monthly.\n"
        "Risks: Data quality and stakeholder availability; Mitigation: weekly check-ins and data validation.\n"
        "Signatures: Client COO and Vendor Account Director.\n"
        "Acceptance Process: Client review within 5 business days, then sign-off via email."
    )

    # Write seed files into ./docs so users can inspect/edit later
    with open(os.path.join(DOCS_DIR, "demo_msa.txt"), "w", encoding="utf-8") as f:
        f.write(demo1)
    with open(os.path.join(DOCS_DIR, "demo_sow.txt"), "w", encoding="utf-8") as f:
        f.write(demo2)

    texts.extend([demo1, demo2])
    return texts


def build_or_load_vectorstore() -> FAISS:
    """
    Load FAISS from VECTORSTORE_DIR if present.
    Otherwise, build from DOCS_DIR/*.txt (or seeded demo texts) and save to disk.
    """
    embeddings = OpenAIEmbeddings()

    # Try to load an existing index first
    if os.path.isdir(VECTORSTORE_DIR):
        try:
            vs = FAISS.load_local(VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs
        except Exception:
            # Fall through to rebuild if loading fails
            pass

    # Build: gather texts from ./docs/*.txt or seed demo files
    seed_demo_texts_if_needed()
    loaders = []
    txt_files = []
    for name in os.listdir(DOCS_DIR):
        if name.lower().endswith(".txt"):
            path = os.path.join(DOCS_DIR, name)
            txt_files.append(path)
            loaders.append(TextLoader(path, encoding="utf-8"))

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        # Should not happen because seed_demo_texts_if_needed writes two demo files,
        # but keep a guard to be safe.
        raise RuntimeError("No documents found to build the index.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    return vs


def load_questions(path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Invalid evaluation_questions.json format.")
    return data["questions"]


def run_rag(llm: ChatOpenAI, vs: FAISS, query: str, top_k: int = TOP_K) -> str:
    """
    Very simple RAG: retrieve top_k chunks, then ask the LLM to answer
    *only* using the provided context.
    """
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)

    context = "\n\n".join([d.page_content for d in docs]) if docs else "No context found."

    system_prompt = (
        "You are a helpful business document assistant. Answer the user question "
        "ONLY using the context below. If the answer is not in the context, say "
        "'I cannot find that in the provided documents.' Keep answers concise.\n\n"
        f"CONTEXT:\n{context}"
    )

    # ChatOpenAI expects messages list; we provide system+user
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    response = llm.invoke(messages)
    # response is an AIMessage; its .content has the text
    return getattr(response, "content", "").strip()


def word_count(text: str) -> int:
    return len(text.split())


def keyword_match_ratio(answer: str, expected_keywords: List[str]) -> float:
    """
    Simple keyword hit ratio: case-insensitive; basic 'stem-ish' match by lowering and checking substrings.
    Returns hits / total_keywords (0 if none).
    """
    if not expected_keywords:
        return 0.0
    ans = answer.lower()
    hits = 0
    for kw in expected_keywords:
        if kw.lower() in ans:
            hits += 1
    return hits / len(expected_keywords)


def evaluate_all() -> List[Dict[str, Any]]:
    """
    Main evaluation loop:
    - loads/creates vector store
    - loads questions
    - queries the RAG pipeline
    - records metrics
    - writes CSV
    - returns rows for optional Streamlit view
    """
    ensure_api_key()
    questions = load_questions(QUESTIONS_PATH)
    vs = build_or_load_vectorstore()

    # Model: gpt-3.5-turbo (fast, cheap)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    rows: List[Dict[str, Any]] = []
    for q in questions:
        qid = q.get("id")
        qtype = q.get("type", "")
        question = q.get("question", "")
        expected = q.get("expected_answer_keywords", [])

        start = time.perf_counter()
        answer = run_rag(llm, vs, question, top_k=TOP_K)
        elapsed = time.perf_counter() - start

        row = {
            "id": qid,
            "type": qtype,
            "question": question,
            "answer": answer,
            "response_time_s": round(elapsed, 3),
            "answer_word_count": word_count(answer),
            "keyword_match_ratio": round(keyword_match_ratio(answer, expected), 3),
        }
        rows.append(row)

    # Save CSV
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "type",
                "question",
                "answer",
                "response_time_s",
                "answer_word_count",
                "keyword_match_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute simple aggregates for console/Streamlit summary."""
    if not rows:
        return {
            "avg_response_time_s": 0.0,
            "avg_answer_words": 0.0,
            "avg_keyword_match_ratio": 0.0,
        }

    n = len(rows)
    avg_time = sum(r["response_time_s"] for r in rows) / n
    avg_words = sum(r["answer_word_count"] for r in rows) / n
    avg_kw = sum(r["keyword_match_ratio"] for r in rows) / n

    return {
        "avg_response_time_s": round(avg_time, 3),
        "avg_answer_words": round(avg_words, 1),
        "avg_keyword_match_ratio": round(avg_kw, 3),
    }


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":
    # Detect if running via Streamlit: st.runtime.exists() is available in recent versions.
    running_streamlit = False
    try:
        running_streamlit = st.runtime.exists()
    except Exception:
        running_streamlit = False

    rows = evaluate_all()
    summary = summarize(rows)

    if running_streamlit:
        st.title("Business Document Q&A — Evaluation Results")
        st.write("Saved CSV:", RESULTS_CSV)
        st.dataframe(rows)
        st.subheader("Summary")
        st.json(summary)
    else:
        print("\n=== Evaluation Complete ===")
        print(f"Results saved to: {RESULTS_CSV}")
        print("Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

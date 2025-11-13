Slide 1: Title & Introduction

Content:

Project Title

Your Name & Program

Tech Stack Icons: Python, LangChain, FAISS, GPT-3.5, Streamlit

One-line project summary

Talking Points:

“Good afternoon, my name is [Name]. Today I’ll present my capstone project: an AI-powered system that answers questions from business documents using Retrieval Augmented Generation.”

Time: 1 min
Visuals: Clean title slide with tech logos in a horizontal row.

Slide 2: Problem Statement

Content:

“Businesses struggle to extract key insights from large document sets.”

“Manual search is time-consuming and error-prone.”

Example: contracts, reports, proposals.

Talking Points:

“Imagine searching through hundreds of pages to find a single clause or financial figure. This system automates that process.”

Time: 1 min
Visuals: Graphic showing a person overwhelmed by documents vs. AI extracting answers.

Slide 3: Project Goal & Objectives

Content:

Main Goal: “Create an intelligent document Q&A system using RAG.”

Objectives:

Automate document understanding

Provide accurate, explainable answers

Evaluate retrieval and response quality

Talking Points:

“The objective isn’t just to build a chatbot—it’s to build a reliable retrieval pipeline optimized for factual business use.”

Time: 1 min
Visuals: Three goal icons with short phrases.

Slide 4: System Overview

Content:

High-level flow: User → Query → Retriever → LLM → Answer

RAG Concept Summary

Talking Points:

“RAG stands for Retrieval Augmented Generation. It combines a retriever that fetches relevant text chunks with a generator that produces natural answers.”

Time: 1.5 min
Visuals: Overview flow diagram.

Slide 5: Technical Architecture

Content:

Components:

Document Loader

Text Splitter

Embedding Model

Vector Store (FAISS)

Retriever + GPT-3.5

Streamlit Frontend

Talking Points:

“The system pipelines documents through these stages—from raw PDF to interactive Q&A.”

Time: 2 min
Visuals: End-to-end architecture diagram with labeled arrows.

Slide 6: Data Processing Pipeline

Content:

Step 1: PDF parsing

Step 2: Chunking (500 tokens)

Step 3: Embedding generation

Step 4: Vector storage (FAISS)

Talking Points:

“Chunks of around 500 tokens provided the best trade-off between context coverage and retrieval accuracy.”

Time: 1.5 min
Visuals: Pipeline diagram or flow arrows.

Slide 7: Model Selection & Reasoning

Content:

GPT-3.5 chosen for cost-performance balance

Alternative models (GPT-4, LLaMA-2) considered

Trade-offs shown in table

Talking Points:

“GPT-3.5 offered strong accuracy at one-fifth the cost of GPT-4 for short business questions.”

Time: 1 min
Visuals: Comparison table (GPT-3.5 vs GPT-4 vs open-source).

Slide 8: Vector Database Choice

Content:

Comparison: FAISS vs ChromaDB vs Pinecone

Decision: FAISS (offline, open-source, simple integration)

Talking Points:

“FAISS was ideal for local evaluation—lightweight, free, and fast.”

Time: 1 min
Visuals: Bar chart comparing query latency or cost.

Slide 9: Streamlit Interface (Frontend)

Content:

Screenshot of Streamlit app

Key UI features: document upload, question input, answer output

Talking Points:

“The interface was designed for non-technical users—drag, ask, and get an answer.”

Time: 1 min
Visuals: Screenshot or mockup.

Slide 10: Evaluation Methodology

Content:

20-question evaluation (10 factual, 5 analytical, 5 summary)

Metrics:

Precision, Recall, F1-score

Answer relevance (0–5 scale)

Talking Points:

“Each question had expected keywords and was graded automatically.”

Time: 1.5 min
Visuals: Evaluation workflow chart or table.

Slide 11: Results & Metrics

Content:

Charts: Retrieval accuracy, response relevance

Example: Precision = 0.82, Recall = 0.77, F1 = 0.79

Summary of top insights

Talking Points:

“The system achieved nearly 80% F1, showing strong retrieval performance for factual queries.”

Time: 1.5 min
Visuals: Bar/line charts with performance metrics.

Slide 12: Live Demo

Content:

Short intro to what you’ll show

Key document: business contract or annual report

Talking Points:

“Let’s see the system in action: I’ll upload a sample contract and ask a few questions.”

Time: 2 min
Visuals: None—switch to Streamlit demo.

Slide 13: Limitations & Future Work

Content:

Limitations:

Context window truncation

No multi-turn memory

Limited evaluation set

Future:

Add re-ranking

Multi-document chaining

GPT-4 or fine-tuning options

Talking Points:

“The next step is scaling the system to multi-document and multilingual contexts.”

Time: 1 min
Visuals: Split slide: “Limitations” left, “Future Work” right.

Slide 14: Technical Justification Summary

Content:

Decision matrix with chosen options highlighted

Summary of trade-offs and cost analysis

Talking Points:

“Every technical decision balanced performance, cost, and feasibility within student constraints.”

Time: 1 min
Visuals: Matrix table (criteria × options).

Slide 15: Conclusion & Q&A

Content:

“Built an end-to-end RAG pipeline.”

“Demonstrated retrieval accuracy and cost-effective deployment.”

“Open for questions.”

Talking Points:

“Thank you! This project shows that RAG can make business document understanding accessible and efficient.”

Time: 1 min
Visuals: Minimal thank-you slide with your contact info.
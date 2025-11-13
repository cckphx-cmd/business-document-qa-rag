# 🎯 Business Intelligence Suite
### Multi-Document RAG System with Dual LLM Architecture

An AI-powered document analysis system that enables semantic search across multiple business documents simultaneously, with accurate source attribution and dual answer modes.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Key Features

### Core Capabilities
- **Multi-Document Search**: Query across multiple PDFs simultaneously
- **Persistent Storage**: ChromaDB vector database with automatic persistence
- **Dual LLM Pipeline**: Toggle between factual (precise) and conversational (friendly) answer modes
- **Smart Citations**: Automatic source attribution with document names and page numbers
- **Professional UI**: Keyboard shortcuts, Enter-to-search, real-time document management

### Advanced Features
- **Cross-Document Analysis**: Compare information across multiple documents
- **Document Library Management**: Upload, view, and delete documents with ease
- **Export Functionality**: Download conversation history for records
- **Stats Dashboard**: Real-time metrics and performance indicators
- **Theme System**: Customizable CSS themes

---

## 📊 Performance Metrics

Evaluated with 20 diverse business questions:

| Metric | Result |
|--------|--------|
| **Average Response Time** | 1.24 seconds |
| **Accuracy (Keyword Match)** | 72.1% |
| **Perfect Answers** | 5/20 (100% match) |
| **Completion Rate** | 100% |

See [EVALUATION_SUMMARY.md](EVALUATION_SUMMARY.md) for detailed results.

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────┐
│            Streamlit Frontend                    │
│  (Professional UI with keyboard shortcuts)       │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│         LangChain RAG Pipeline                   │
│  • Query Processing                              │
│  • Document Retrieval (k=5)                      │
│  • Context Assembly                              │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
┌──────▼─────┐        ┌───────▼────────┐
│  ChromaDB  │        │   Dual LLM     │
│  (Vector   │        │   Pipeline     │
│  Database) │        │                │
│            │        │  • Factual     │
│  OpenAI    │        │  • Conversational│
│  Embeddings│        │                │
└────────────┘        └────────────────┘
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Professional UI)
- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB (Persistent)
- **Embeddings**: OpenAI text-embedding-ada-002
- **LLMs**: OpenAI GPT-3.5-turbo (dual pipeline)
- **Document Processing**: PyPDF, RecursiveCharacterTextSplitter

---

## 🚦 Getting Started

### Prerequisites
```bash
Python 3.9+
OpenAI API Key
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/cckphx-cmd/business-document-qa-rag.git
cd business-document-qa-rag
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

5. **Run the application**
```bash
streamlit run app_multidoc_pro.py
```

6. **Open browser**
```
http://localhost:8501
```

---

## 📖 Usage

### Upload Documents
1. Click "Browse files" in the upload section
2. Select one or multiple PDF files (Cmd/Ctrl + Click for multiple)
3. Wait for processing (progress bar shows status)
4. Documents appear in the sidebar library

### Ask Questions
1. Type your question in the search box
2. Press **Enter** or click "Search All Docs"
3. View results with source citations
4. Toggle between factual and conversational modes in Settings

### Example Queries
```
- "What are the payment terms across all contracts?"
- "Compare termination clauses in these documents"
- "Which document mentions Q4 revenue?"
- "Summarize the key risks identified"
- "What is the highest revenue number in all documents?"
```

---

## 🎨 Features in Detail

### Dual LLM Pipeline
- **Factual Mode**: Precise, professional answers with exact citations
- **Conversational Mode**: Warm, approachable rewrites maintaining accuracy
- **Side-by-Side**: Compare both styles simultaneously

### Document Management
- View all uploaded documents in sidebar
- See upload date and metadata
- Delete individual documents
- Persistent storage (survives restart)

### Keyboard Shortcuts
- **Enter**: Submit search
- **Triple-click**: Select all text in search box
- **Delete/Backspace**: Clear selected text

---

## 📁 Project Structure
```
business-document-qa-rag/
├── app_multidoc_pro.py      # Main application (professional version)
├── app.py                    # Single-doc version
├── evaluate.py               # Evaluation script
├── evaluation_questions.json # Test questions
├── results.csv              # Evaluation results
├── EVALUATION_SUMMARY.md    # Detailed metrics
├── themes/                  # CSS themes
│   └── current.css
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (not tracked)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🧪 Evaluation

Run the evaluation suite:
```bash
python evaluate.py
```

This tests the system with 20 diverse questions and generates:
- Response time metrics
- Accuracy measurements
- Detailed CSV results

---

## 🎓 Use Cases

- **Contract Analysis**: Search and compare terms across multiple contracts
- **Report Synthesis**: Extract insights from multiple business reports
- **Due Diligence**: Rapid document review for M&A or investments
- **Compliance**: Check policy adherence across document sets
- **Research**: Academic or market research across multiple sources

---

## 🔮 Future Enhancements

- [ ] Document comparison view (side-by-side)
- [ ] Advanced analytics dashboard
- [ ] Export to PDF with citations
- [ ] API endpoint for integration
- [ ] Support for Word docs, Excel, etc.
- [ ] Brand voice assistance feature
- [ ] Multi-language support

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Author

**CCK85**
- GitHub: [@cckphx-cmd](https://github.com/cckphx-cmd)
- Project: Capstone 2025

---

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [OpenAI](https://openai.com/)
- UI with [Streamlit](https://streamlit.io/)
- Vector DB by [ChromaDB](https://www.trychroma.com/)

---

## 📝 Notes

This system demonstrates practical application of:
- Retrieval Augmented Generation (RAG)
- Vector databases and embeddings
- Multi-document semantic search
- Dual LLM architectures
- Production-ready UI/UX design

**Perfect for showcasing in portfolios and technical interviews!**


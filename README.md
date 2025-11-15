# 🎯 Business Intelligence Suite
### Multi-Document RAG System with Dual LLM Architecture

An AI-powered document analysis system that enables semantic search across multiple business documents simultaneously, with accurate source attribution and dual answer modes.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🚀 Key Features

### Multi-Document Q&A (Tab 1)
- **Multi-Document Search**: Query across multiple PDFs simultaneously
- **Persistent Storage**: ChromaDB vector database with automatic persistence
- **Dual LLM Pipeline**: Toggle between factual (precise) and conversational (friendly) answer modes
- **Smart Citations**: Automatic source attribution with document names and page numbers
- **Cross-Document Analysis**: Compare information across multiple documents
- **Document Library Management**: Upload, view, and delete documents with ease

### Brand Voice Assistant (Tab 2) 🆕
- **Brand Guide Upload**: Upload company style guides, brand guidelines, or tone-of-voice documents
- **Context-Aware Transformation**: Uses your actual brand guidelines via ChromaDB retrieval
- **Multiple Message Types**: Email, Slack, social media, customer support, and more
- **Tone Control**: Professional, friendly, technical, empathetic, urgent, casual
- **Before/After Comparison**: Visual side-by-side transformation results
- **Guidelines Transparency**: See which brand guidelines were used in transformation

### Professional UI/UX
- **Keyboard Shortcuts**: Enter-to-search, intuitive navigation
- **Export Functionality**: Download conversation history for records
- **Stats Dashboard**: Real-time metrics and performance indicators
- **Theme System**: Customizable CSS themes
- **Responsive Design**: Works on desktop and tablet

---

## 🎬 Demo

### Multi-Document Q&A
Upload multiple business documents and ask questions like:
- *"Compare payment terms across all contracts"*
- *"What is the highest revenue mentioned?"*
- *"Which documents discuss termination clauses?"*

Get instant answers with accurate source citations from across your entire document library.

### Brand Voice Assistant
Transform informal drafts into polished, on-brand communications:
- Upload your company's style guide
- Paste any draft message
- Receive a professional, brand-consistent version instantly

Perfect for marketing teams, customer support, and anyone who needs to maintain consistent brand voice.

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
┌─────────────────────────────────────────────────────────┐
│              Streamlit Frontend                          │
│  • Multi-Document Q&A (Tab 1)                           │
│  • Brand Voice Assistant (Tab 2)                        │
│  • Settings & Configuration (Tab 3)                     │
└──────────────────┬──────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐    ┌──────▼─────────┐
│   Document     │    │  Brand Voice   │
│   Q&A System   │    │  Transformer   │
└───────┬────────┘    └──────┬─────────┘
        │                     │
┌───────▼──────────────────────▼─────────┐
│         ChromaDB Vector Store           │
│  • business_documents (Q&A)             │
│  • brand_guidelines (Voice Transform)   │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼─────┐      ┌───────▼────────┐
│   OpenAI    │      │   Dual LLM     │
│  Embeddings │      │   Pipeline     │
│             │      │                │
│ ada-002     │      │  • Factual     │
│             │      │  • Conversational│
└─────────────┘      └────────────────┘
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

### Tab 1: Multi-Document Q&A

#### Upload Documents
1. Click "Browse files" in the upload section
2. Select one or multiple PDF files (Cmd/Ctrl + Click for multiple)
3. Wait for processing (progress bar shows status)
4. Documents appear in the sidebar library

#### Ask Questions
1. Type your question in the search box
2. Press **Enter** or click "Search All Docs"
3. View results with source citations
4. Toggle between factual and conversational modes in Settings

#### Example Queries
```
- "What are the payment terms across all contracts?"
- "Compare termination clauses in these documents"
- "Which document mentions Q4 revenue?"
- "Summarize the key risks identified"
- "What is the highest revenue number in all documents?"
```

---

### Tab 2: Brand Voice Assistant 🆕

#### Upload Brand Guidelines (Optional)
1. Go to Tab 2: Brand Voice Assistant
2. Upload your company's brand guide or style guide (PDF)
3. System extracts voice patterns and stores in ChromaDB
4. Guidelines used automatically for all transformations

#### Transform Messages
1. Select message type (Email, Slack, Social Media, etc.)
2. Choose desired tone (Professional, Friendly, Technical, etc.)
3. Enter your draft message
4. Click "Transform Message"
5. View before/after comparison
6. Copy the transformed message

#### Example Transformations
```
Before: "Hey, server's down. Working on it."

After (Professional):
"We're currently experiencing a service interruption and our team 
is actively working to restore functionality. We'll provide updates 
every 30 minutes and appreciate your patience."

---

Before: "Can't give you a refund. Policy says no."

After (Empathetic):
"I understand your frustration and want to help find the best 
solution. While our standard policy has specific timeframes, let me 
review your account to explore what options might be available."
```

---

## 🎨 Features in Detail

### Multi-Document Q&A

#### Dual LLM Pipeline
- **Factual Mode**: Precise, professional answers with exact citations
- **Conversational Mode**: Warm, approachable rewrites maintaining accuracy
- **Side-by-Side**: Compare both styles simultaneously

#### Document Management
- View all uploaded documents in sidebar
- See upload date and metadata
- Delete individual documents
- Persistent storage (survives restart)

---

### Brand Voice Assistant 🆕

#### Document-Based Learning
- Upload company brand guidelines (PDF)
- ChromaDB extracts and stores voice patterns
- Retrieval-augmented transformation uses actual guidelines
- Transparency: See which guidelines influenced each transformation

#### Multiple Contexts
- **Message Types**: Email, Slack, social media, support, press releases
- **Tone Options**: Professional, friendly, technical, empathetic, urgent, casual
- **Smart Adaptation**: Combines message type + tone + brand guidelines

#### Transformation Quality
- Preserves all key information
- Maintains professional standards
- Applies brand-specific language
- Formats appropriately for channel
- Before/after comparison for review

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

## 💼 Use Cases

### Multi-Document Q&A
- **Contract Analysis**: Search and compare terms across multiple contracts
- **Report Synthesis**: Extract insights from multiple business reports
- **Due Diligence**: Rapid document review for M&A or investments
- **Compliance**: Check policy adherence across document sets
- **Research**: Academic or market research across multiple sources

### Brand Voice Assistant
- **Marketing Teams**: Ensure all communications match brand guidelines
- **Customer Support**: Transform support responses to maintain brand voice
- **Internal Communications**: Standardize messaging across departments
- **Social Media**: Brand-consistent posts and responses
- **Content Creation**: Generate on-brand marketing copy and emails
- **Onboarding**: Help new employees learn company voice quickly

---

## 🔮 Future Enhancements

### Completed ✅
- [x] Multi-document RAG with ChromaDB
- [x] Dual LLM pipeline (factual + conversational)
- [x] Brand Voice Assistant with document upload
- [x] Professional UI with keyboard shortcuts
- [x] Comprehensive evaluation metrics

### Planned 🚧
- [ ] Document comparison view (side-by-side)
- [ ] Advanced analytics dashboard
- [ ] Export to PDF with citations
- [ ] API endpoint for integration
- [ ] Support for Word docs, Excel, etc.
- [ ] Multi-language support
- [ ] Real-time collaboration features

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
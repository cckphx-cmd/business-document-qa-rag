import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st
import tempfile
from datetime import datetime

load_dotenv()

def export_conversation_history(history):
    """Export conversation history to formatted text"""
    if not history:
        return "No conversation history to export."
    
    output = "="*60 + "\n"
    output += "BUSINESS DOCUMENT Q&A - CONVERSATION EXPORT\n"
    output += "="*60 + "\n\n"
    
    for i, item in enumerate(history, 1):
        output += f"Question {i} ({item['timestamp']}):\n"
        output += f"{item['question']}\n\n"
        output += f"Answer:\n{item['answer']}\n\n"
        output += "-"*60 + "\n\n"
    
    return output

# Page config
st.set_page_config(
    page_title="Business Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .doc-info {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .question-box {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
    background-color: #e8f4f8;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
    color: #212529;
}
    }
    .source-passage {
        background-color: #fff3cd;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📄 Business Document Q&A System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload business documents and ask questions using AI-powered search</p>', unsafe_allow_html=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []

# Sidebar
with st.sidebar:
    st.header("📊 System Info")
    
    if 'vector_store' in st.session_state:
        st.success("✅ Document Loaded")
        
        # Document stats
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Chunks", st.session_state.get('chunks_count', 0))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Questions", len(st.session_state['conversation_history']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        if st.session_state.get('doc_filename'):
            st.markdown(f"**File:** {st.session_state['doc_filename']}")
        if st.session_state.get('doc_pages'):
            st.markdown(f"**Pages:** {st.session_state['doc_pages']}")
        
        st.divider()
        
        if st.button("🔄 Clear Document", use_container_width=True):
            for key in ['vector_store', 'chunks_count', 'doc_filename', 'doc_pages', 'conversation_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
            # Export conversation if exists
        if st.session_state.get('conversation_history'):
            st.divider()
            export_text = export_conversation_history(st.session_state['conversation_history'])
            st.download_button(
                label="📥 Download Q&A History",
                data=export_text,
                file_name=f"qa_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True,
                type="secondary",
                key="download_bottom"
            
            )   

    else:
        st.info("👆 Upload a document to begin")
    if st.session_state['conversation_history']:
            # Export conversation
            export_text = export_conversation_history(st.session_state['conversation_history'])
            
            st.download_button(
                label="📥 Download Q&A History",
                data=export_text,
                file_name=f"qa_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.divider()
    st.caption("Built with LangChain, FAISS, and OpenAI")
    st.caption("Capstone Project 2025")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a business document (contract, report, proposal, etc.)"
    )

with col2:
    st.subheader("ℹ️ Supported Documents")
    st.markdown("""
    - Contracts & Agreements
    - Business Reports
    - Proposals & RFPs
    - Financial Statements
    - Corporate Documents
    """)

# Process uploaded file
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    # Only process if new file
    if st.session_state.get('doc_filename') != uploaded_file.name:
        with st.spinner("🔄 Processing document... This may take a moment."):
            try:
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                if not documents or len(documents) == 0:
                    st.error("❌ Could not extract text from PDF. Please ensure it's a text-based PDF.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    if len(chunks) == 0:
                        st.error("❌ Could not create text chunks from document.")
                    else:
                        embeddings = OpenAIEmbeddings()
                        vector_store = FAISS.from_documents(chunks, embeddings)
                        
                        # Store in session
                        st.session_state['vector_store'] = vector_store
                        st.session_state['chunks_count'] = len(chunks)
                        st.session_state['doc_filename'] = uploaded_file.name
                        st.session_state['doc_pages'] = len(documents)
                        st.session_state['conversation_history'] = []
                        
                        st.success(f"✅ Successfully processed **{uploaded_file.name}**!")
                        st.balloons()
                        
                        # Show document info
                        st.markdown(f"""
                        <div class="doc-info">
                            <strong>📄 Document Processed</strong><br>
                            <strong>Filename:</strong> {uploaded_file.name}<br>
                            <strong>Pages:</strong> {len(documents)}<br>
                            <strong>Text Chunks:</strong> {len(chunks)}<br>
                            <strong>Status:</strong> Ready for questions
                        </div>
                        """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Q&A Section
if 'vector_store' in st.session_state:
    st.divider()
    st.subheader("💬 Ask Questions")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the payment terms? Who are the parties involved?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Ask Question", use_container_width=True, type="primary")
    with col2:
        if st.session_state['conversation_history']:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state['conversation_history'] = []
                st.rerun()
    
    if ask_button and question:
        with st.spinner("🤔 Analyzing document..."):
            try:
                vector_store = st.session_state['vector_store']
                docs = vector_store.similarity_search(question, k=3)
                
                if not docs:
                    st.warning("⚠️ No relevant information found in the document.")
                    
                else:
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
                    prompt = f"""You are a professional business analyst reviewing documents. Provide accurate, structured answers based ONLY on the provided context.

CONTEXT FROM DOCUMENT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer directly and professionally
- For contracts: cite specific clauses, dates, and parties
- For financial data: include exact numbers, dates, and currency
- For agreements: specify terms, obligations, and timeframes  
- If information is not in the context, state "This information is not provided in the document"
- Use bullet points for multiple items
- Be concise but complete

ANSWER:"""

                    
                    response = llm.invoke(prompt)
                    
                    # Save to history
                    st.session_state['conversation_history'].append({
                        'question': question,
                        'answer': response.content,
                        'sources': docs,
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Display conversation history
    if st.session_state['conversation_history']:
        st.divider()
        st.subheader("📝 Conversation History")
        
        for i, item in enumerate(reversed(st.session_state['conversation_history'])):
            with st.container():
                st.markdown(f"**Q{len(st.session_state['conversation_history']) - i}:** {item['question']} *({item['timestamp']})*")
                
                st.markdown(f"""
                <div class="answer-box">
                    <strong>Answer:</strong><br>
                    {item['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander("📚 View Source Passages"):
                    for j, doc in enumerate(item['sources'], 1):
                        st.markdown(f"""
                        <div class="source-passage">
                            <strong>Passage {j}:</strong><br>
                            {doc.page_content[:400]}...
                        </div>
                        """, unsafe_allow_html=True)
                
                st.divider()

else:
    st.info("👆 **Getting Started:** Upload a PDF document above to begin asking questions!")
    
    # Show example questions
    st.subheader("💡 Example Questions You Can Ask")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **For Contracts:**
        - What are the payment terms?
        - What is the termination clause?
        - Who are the parties involved?
        - What are the deliverables?
        """)
    
    with col2:
        st.markdown("""
        **For Reports:**
        - What are the key findings?
        - Summarize the main points
        - What risks are identified?
        - What are the recommendations?
        """)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import json
import time

load_dotenv()

print("Starting evaluation...")

# Load questions
with open('evaluation_questions.json', 'r') as f:
    data = json.load(f)
    questions = data['questions'][:5]  # Test with first 5 questions only

print(f"Loaded {len(questions)} questions")

# Load a document
print("Loading test document...")
# Create simple demo document (no PDF needed)
print("Using demo document...")
from langchain_core.documents import Document

documents = [
    Document(page_content="""
Service Agreement between Acme Solutions LLC and Bright Future Enterprises.
Effective Date: January 1, 2025
Payment Terms: Net 30 days, invoiced monthly at $150/hour
Contract Duration: 12 months
Termination: Either party may terminate with 30 days written notice
Services: Consulting and software development as described in Exhibit A
Confidentiality: Both parties agree to maintain confidentiality of proprietary information
""")
]
print(f"Created demo document")
documents = [Document(page_content="This is a service agreement between Acme Corp and Client Inc. Payment terms are net 30 days. The contract starts on January 1, 2025 and runs for 12 months. Either party may terminate with 30 days notice.")]
print("Using demo document")

# Create chunks
print("Creating chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Create vector store
print("Creating embeddings... (this takes a moment)")
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store ready!")

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Test questions
results = []
print("\nTesting questions...")

for i, q in enumerate(questions, 1):
    print(f"\n{i}. {q['question']}")
    
    start_time = time.time()
    
    # Search
    docs = vector_store.similarity_search(q['question'], k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Get answer
    prompt = f"Based on this context, answer the question.\n\nContext: {context}\n\nQuestion: {q['question']}\n\nAnswer:"
    response = llm.invoke(prompt)
    
    elapsed = time.time() - start_time
    
    print(f"   Answer: {response.content[:100]}...")
    print(f"   Time: {elapsed:.2f}s")
    
    results.append({
        'question': q['question'],
        'answer': response.content,
        'time': elapsed
    })

# Summary
print("\n" + "="*50)
print("EVALUATION COMPLETE")
print(f"Questions answered: {len(results)}")
print(f"Average response time: {sum(r['time'] for r in results)/len(results):.2f}s")
print("="*50)

# Save results
with open('simple_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to simple_results.json")
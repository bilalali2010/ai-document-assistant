import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title="AI Document Q&A", page_icon="🤖", layout="wide")

st.title("📚 AI Document Q&A Bot")
st.info("🚀 Proper AI version with ChromaDB - Fast and Powerful")

def process_document(uploaded_file):
    """Process PDF with ChromaDB - with better error handling"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and split document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Check if PDF has content
        if not documents or len(documents) == 0:
            return None, "PDF appears to be empty or cannot be read"
        
        # Check if documents have content
        total_text = "".join([doc.page_content for doc in documents if doc.page_content])
        if len(total_text.strip()) < 10:  # Less than 10 characters
            return None, "PDF contains no readable text (might be image-based)"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Use ChromaDB instead of FAISS
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings
        )
        
        return vector_store, "Document processed successfully!"
        
    except Exception as e:
        return None, f"Error processing PDF: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_error" not in st.session_state:
    st.session_state.processing_error = ""

# File upload
uploaded_file = st.file_uploader("Upload PDF for AI Q&A", type=["pdf"], 
                                help="Upload a PDF with readable text (not image-only)")

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name} ({uploaded_file.size} bytes)")
    
    # Check file size
    if uploaded_file.size < 100:  # Less than 100 bytes
        st.error("❌ File is too small or empty")
    else:
        if st.session_state.vector_store is None:
            with st.spinner("🔄 Processing document with AI..."):
                vector_store, status = process_document(uploaded_file)
                if vector_store:
                    st.session_state.vector_store = vector_store
                    try:
                        st.session_state.llm = HuggingFaceHub(
                            repo_id="google/flan-t5-base",
                            model_kwargs={"temperature": 0.1, "max_length": 512}
                        )
                        st.success("✅ AI Ready! Ask questions below")
                        st.session_state.processing_error = ""
                    except Exception as e:
                        st.error(f"❌ AI model loading failed: {e}")
                else:
                    st.error(f"❌ {status}")
                    st.session_state.processing_error = status

# Q&A Interface
if st.session_state.vector_store:
    st.header("💬 Ask Questions")
    
    question = st.text_input("Your question about the document:")
    
    if question and st.button("Get AI Answer"):
        with st.spinner("🤖 AI thinking..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 2})
                )
                
                answer = qa_chain.run(question)
                st.success("**AI Answer:**")
                st.write(answer)
                
            except Exception as e:
                st.error(f"AI error: {str(e)}")

# Test with sample document
st.header("🧪 Test with Sample Content")
if st.button("Load Sample Document for Testing"):
    # Create a sample document in memory
    sample_text = """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines. 
    Machine Learning is a subset of AI that enables computers to learn from data.
    Deep Learning uses neural networks with multiple layers.
    Natural Language Processing (NLP) allows computers to understand human language.
    
    Applications include chatbots, image recognition, and recommendation systems.
    Popular frameworks are TensorFlow, PyTorch, and LangChain.
    """
    
    from langchain.schema import Document
    sample_doc = Document(page_content=sample_text)
    
    # Process sample
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents([sample_doc])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)
    
    st.session_state.vector_store = vector_store
    st.session_state.llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.1, "max_length": 512}
    )
    
    st.success("✅ Sample AI document loaded! Try asking:")
    st.code("What is Artificial Intelligence?\nWhat are AI applications?\nExplain Machine Learning")

st.sidebar.markdown("""
### 🎯 Features
- ✅ **Real AI Q&A** - ChromaDB based
- ✅ **PDF Understanding** - Proper text processing  
- ✅ **Free Models** - Hugging Face integration
- ✅ **Fast Deployment** - Lightweight dependencies

### 🔧 Technology
- **Vector DB**: ChromaDB (FAISS alternative)
- **AI Models**: Hugging Face
- **Framework**: LangChain

### 💡 Tips
- Use PDFs with **selectable text** (not image-only)
- **Smaller files** process faster
- **Clear text** works best
""")

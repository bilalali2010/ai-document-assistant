import streamlit as st
import os
import tempfile
import time

# Safe imports with compatibility
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA
    from langchain_community.llms import HuggingFaceHub
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORT_SUCCESS = False

# Configure page first
st.set_page_config(
    page_title="Free Document Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-radius: 0.5rem;
        border-left: 5px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def safe_imports():
    """Safely import required libraries with error handling"""
    try:
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        from langchain.chains import RetrievalQA
        from langchain.llms import HuggingFaceHub
        return True, None
    except ImportError as e:
        return False, f"Missing dependency: {e}"

def initialize_embeddings():
    """Initialize embeddings with fallback"""
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return embeddings, None
    except Exception as e:
        return None, f"Embeddings error: {e}"

def initialize_llm():
    """Initialize free LLM with fallback options"""
    try:
        from langchain.llms import HuggingFaceHub
        
        # Try different free models (fallback options)
        models_to_try = [
            "google/flan-t5-large",
            "google/flan-t5-base", 
            "declare-lab/flan-alpaca-base"
        ]
        
        for model_id in models_to_try:
            try:
                llm = HuggingFaceHub(
                    repo_id=model_id,
                    model_kwargs={
                        "temperature": 0.1,
                        "max_length": 512,
                        "max_new_tokens": 256
                    }
                )
                return llm, f"Using model: {model_id}"
            except:
                continue
                
        return None, "No suitable model found"
        
    except Exception as e:
        return None, f"LLM initialization error: {e}"

def process_document(uploaded_file):
    """Process uploaded document with robust error handling"""
    if not uploaded_file:
        return None, "No file provided"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load document based on file type
        file_type = uploaded_file.type
        if file_type == "application/pdf":
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
        elif file_type == "text/plain":
            from langchain.document_loaders import TextLoader
            loader = TextLoader(tmp_path)
        elif "wordprocessingml" in file_type or "msword" in file_type:
            try:
                from langchain.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(tmp_path)
            except:
                # Fallback for DOCX
                import docx2txt
                text = docx2txt.process(tmp_path)
                from langchain.docstore.document import Document
                documents = [Document(page_content=text)]
                return process_text_chunks(documents), "DOCX processed with fallback"
        else:
            return None, f"Unsupported file type: {file_type}"
        
        documents = loader.load()
        
        # Process text chunks
        return process_text_chunks(documents), "Document processed successfully"
        
    except Exception as e:
        return None, f"Error processing document: {e}"
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def process_text_chunks(documents):
    """Process document chunks into vector store"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings, error = initialize_embeddings()
    if error:
        raise Exception(f"Embeddings failed: {error}")
    
    # Create vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def main():
    """Main application function"""
    st.markdown('<div class="main-header">📚 Free Document Q&A Bot</div>', unsafe_allow_html=True)
    
    # Check imports
    imports_ok, import_error = safe_imports()
    if not imports_ok:
        st.error(f"❌ Import Error: {import_error}")
        st.info("Please check that all dependencies are installed correctly.")
        return
    
    # Info box
    st.markdown("""
    <div class="info-box">
    💡 <strong>Completely Free AI Document Assistant</strong><br>
    • 100% Free - No API costs<br>
    • Supports PDF, TXT, DOCX files<br>
    • Smart document understanding<br>
    • Deploys instantly on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Hugging Face Token (optional)
        hf_token = st.text_input(
            "Hugging Face Token (optional)",
            type="password",
            help="Get free token from huggingface.co/settings/tokens for better performance"
        )
        
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
            st.success("✅ Token set successfully")
        
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "docx"],
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
            st.write(f"Size: {uploaded_file.size / 1024:.1f} KB")
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = ""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_initialized" not in st.session_state:
        st.session_state.llm_initialized = False
    
    # Process document
    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("🔄 Processing document... This may take a minute."):
            vector_store, status_message = process_document(uploaded_file)
            
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.processing_status = status_message
                st.success("✅ Document processed successfully!")
            else:
                st.error(f"❌ {status_message}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Question input
        if st.session_state.vector_store:
            question = st.chat_input("What would you like to know about your document?")
            
            if question:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Analyzing document..."):
                        try:
                            # Initialize LLM
                            llm, model_info = initialize_llm()
                            
                            if llm:
                                # Show model info
                                st.caption(f"*{model_info}*")
                                
                                # Create QA chain
                                from langchain.chains import RetrievalQA
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=st.session_state.vector_store.as_retriever(
                                        search_kwargs={"k": 2}
                                    ),
                                    return_source_documents=False
                                )
                                
                                # Get answer
                                start_time = time.time()
                                answer = qa_chain.run(question)
                                response_time = time.time() - start_time
                                
                                # Display answer
                                st.write(answer)
                                st.caption(f"Response time: {response_time:.1f}s")
                                
                                # Add to chat history
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer
                                })
                                
                            else:
                                st.error("❌ Failed to initialize AI model. Please try again.")
                                
                        except Exception as e:
                            st.error(f"❌ Error generating response: {str(e)}")
                            st.info("💡 Try rephrasing your question or uploading a smaller document.")
        
        else:
            st.info("📝 Please upload a document to start asking questions")
    
    with col2:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** a document (PDF/TXT/DOCX)
        2. **Wait** for processing to complete
        3. **Ask questions** about your content
        4. **Get AI-powered answers**
        
        **💡 Tips:**
        - For better performance, add a Hugging Face token
        - Start with simple questions
        - Larger documents take longer to process
        - Clear conversation to start fresh
        
        **🔧 Technical Info:**
        - Embeddings: sentence-transformers
        - Vector DB: FAISS
        - LLM: Flan-T5 (free)
        """)
        
        # Controls
        st.header("🛠️ Controls")
        
        if st.button("🔄 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🗑️ Reset Document", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.messages = []
            st.session_state.processing_status = ""
            st.rerun()
        
        # Status
        st.header("📊 Status")
        if st.session_state.vector_store:
            st.success("✅ Ready for questions")
        else:
            st.warning("⏳ Waiting for document upload")

if __name__ == "__main__":
    main()

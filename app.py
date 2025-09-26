import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Free Document Q&A Bot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
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
</style>
""", unsafe_allow_html=True)

def initialize_llm():
    """Initialize free Hugging Face LLM"""
    try:
        from langchain.llms import HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.1, "max_length": 512},
            huggingfacehub_api_token=st.secrets.get("HUGGINGFACEHUB_API_TOKEN", "")
        )
        return llm
    except Exception as e:
        st.error(f"LLM initialization failed: {e}")
        return None

def process_document(uploaded_file):
    """Process uploaded document and create vector store"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load document based on file type
        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.type == "text/plain":
            loader = TextLoader(tmp_path)
        elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error("Unsupported file type")
            return None
        
        documents = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create embeddings using free model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return None
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def main():
    st.markdown('<div class="main-header">📚 Free Document Q&A Bot</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    💡 <strong>Completely Free Version</strong><br>
    • Uses Hugging Face models (no OpenAI costs)<br>
    • Supports PDF, TXT, DOCX files<br>
    • Deployable on Streamlit Cloud for free
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a document",
            type=["pdf", "txt", "docx"],
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.write(f"File size: {uploaded_file.size / 1024:.2f} KB")
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Process document when uploaded
    if uploaded_file and st.session_state.vector_store is None:
        with st.spinner("🔄 Processing document... This may take a few moments."):
            vector_store = process_document(uploaded_file)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.markdown('<div class="success-box">✅ Document processed successfully! You can now ask questions.</div>', unsafe_allow_html=True)
    
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Question input
        if st.session_state.vector_store:
            question = st.chat_input("Ask a question about your document...")
            
            if question:
                # Add user question to chat
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                
                # Generate answer
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Thinking..."):
                        try:
                            # Initialize LLM
                            llm = initialize_llm()
                            if llm:
                                # Create QA chain
                                qa_chain = RetrievalQA.from_chain_type(
                                    llm=llm,
                                    chain_type="stuff",
                                    retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": 3}),
                                    return_source_documents=True
                                )
                                
                                # Get answer
                                result = qa_chain({"query": question})
                                answer = result["result"]
                                
                                # Display answer
                                st.write(answer)
                                
                                # Show source documents
                                with st.expander("📚 Source Documents"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.write(f"**Source {i+1}:** {doc.page_content[:200]}...")
                                
                                st.session_state.messages.append({"role": "assistant", "content": answer})
                            else:
                                st.error("Failed to initialize language model")
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")
        else:
            st.info("📝 Please upload a document to start asking questions")
    
    with col2:
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Upload** a PDF, TXT, or DOCX file
        2. **Wait** for document processing
        3. **Ask questions** about your document
        4. **Get AI-powered answers**
        
        **Supported Files:**
        - PDF documents
        - Text files (.txt)
        - Word documents (.docx)
        
        **Free Models Used:**
        - Embeddings: sentence-transformers/all-MiniLM-L6-v2
        - LLM: google/flan-t5-large
        """)
        
        # Reset button
        if st.button("🔄 Reset Conversation"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.rerun()

if __name__ == "__main__":
    main()

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
    """Process PDF with ChromaDB"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load and split document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
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
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store, "Document processed successfully!"
        
    except Exception as e:
        return None, f"Error: {e}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# File upload
uploaded_file = st.file_uploader("Upload PDF for AI Q&A", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    
    if "vector_store" not in st.session_state:
        with st.spinner("🔄 Processing document with AI..."):
            vector_store, status = process_document(uploaded_file)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.llm = HuggingFaceHub(
                    repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.1, "max_length": 512}
                )
                st.success("✅ AI Ready! Ask questions below")
            else:
                st.error(f"❌ {status}")

# Q&A Interface
if "vector_store" in st.session_state:
    st.header("💬 Ask Questions")
    
    question = st.text_input("Your question about the document:")
    
    if question and st.button("Get AI Answer"):
        with st.spinner("🤖 AI thinking..."):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=st.session_state.llm,
                    chain_type="stuff",
                    retriever=st.session_state.vector_store.as_retriever()
                )
                
                answer = qa_chain.run(question)
                st.success("**AI Answer:**")
                st.write(answer)
                
            except Exception as e:
                st.error(f"AI error: {e}")

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
""")

import os
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.embeddings import EmbeddingsManager
from src.vectorstore import VectorStoreManager
from src.rag_pipeline import LLMService, RAGPipeline

# Page setup
st.set_page_config(
    page_title="RAG-Based AI Assistant",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Retrieval-Augmented Generation (RAG) Chatbot")
st.caption("Powered by LangChain, ChromaDB Vector Store, and Google Gemini 3.8 Flash")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = 0

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager(collection_name="streamlit_rag_demo")

if "emb_manager" not in st.session_state:
    st.session_state.emb_manager = EmbeddingsManager()

# Default Gemini configuration
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-3.8-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

# Clean & Minimalist Sidebar
with st.sidebar:
    st.header("📄 Knowledge Base")

    default_paper_name = "NIPS-2017-attention-is-all-you-need-Paper.pdf"
    has_local_paper = os.path.exists(default_paper_name)

    if has_local_paper:
        st.info(f"Paper: `{default_paper_name}`")
        if st.button("⚡ Load Paper PDF", use_container_width=True):
            with st.spinner("Indexing document pages into ChromaDB..."):
                docs = load_pdf(default_paper_name)
                chunks = split_documents(
                    docs,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    source_name=default_paper_name
                )
                texts = [c["chunk"] for c in chunks]
                embeddings = st.session_state.emb_manager.embed_documents(texts)
                count = st.session_state.vector_store.add_chunks(chunks, embeddings)
                st.session_state.indexed_docs = count
                st.success(f"Indexed {count} chunks successfully!")

    uploaded_file = st.file_uploader("Upload Another PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("📥 Index Uploaded PDF", use_container_width=True):
            with st.spinner("Processing uploaded PDF..."):
                docs = load_pdf(uploaded_file)
                chunks = split_documents(
                    docs,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    source_name=uploaded_file.name
                )
                texts = [c["chunk"] for c in chunks]
                embeddings = st.session_state.emb_manager.embed_documents(texts)
                count = st.session_state.vector_store.add_chunks(chunks, embeddings)
                st.session_state.indexed_docs += count
                st.success(f"Indexed {count} chunks from `{uploaded_file.name}`!")

    st.divider()
    st.markdown(f"**Database:** ChromaDB (`{st.session_state.indexed_docs}` chunks indexed)")
    st.markdown(f"**LLM:** `{MODEL_NAME}`")

    if st.session_state.indexed_docs > 0:
        if st.button("🗑️ Clear Chat & Knowledge Base", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.indexed_docs = 0
            st.session_state.messages = []
            st.rerun()

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {idx+1}** (Document: `{src['source']}`, Page: `{src['page']}`):")
                    st.caption(src["text"])

# Handle user query
user_query = st.chat_input("Ask a question about your indexed documents...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    llm_service = LLMService(
        provider="gemini",
        model_name=MODEL_NAME,
        api_key=GEMINI_KEY
    )
    pipeline = RAGPipeline(
        vector_store=st.session_state.vector_store,
        embeddings_manager=st.session_state.emb_manager,
        llm_service=llm_service
    )

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and synthesizing answer..."):
            response = pipeline.answer_query(user_query, top_k=TOP_K)
            st.markdown(response["answer"])

            if response["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, src in enumerate(response["sources"]):
                        st.markdown(f"**Source {idx+1}** (Document: `{src['source']}`, Page: `{src['page']}`):")
                        st.caption(src["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"]
    })

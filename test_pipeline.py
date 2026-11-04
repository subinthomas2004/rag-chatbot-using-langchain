"""
Verification script for complete RAG pipeline.
"""

from src.document_loader import load_text
from src.text_splitter import split_documents
from src.embeddings import EmbeddingsManager
from src.vectorstore import VectorStoreManager
from src.rag_pipeline import LLMService, RAGPipeline

def run_test():
    print("Step 1: Loading document...")
    docs = load_text("sample_docs/transformer_paper_summary.txt")
    print(f"Loaded {len(docs)} document page(s). Length: {len(docs[0]['text'])} chars")

    print("\nStep 2: Splitting text into chunks...")
    chunks = split_documents(docs, chunk_size=300, overlap=50, source_name="transformer_summary.txt")
    print(f"Generated {len(chunks)} chunks.")
    print("Sample Chunk 1:", chunks[0]["chunk"])

    print("\nStep 3: Generating embeddings...")
    emb_manager = EmbeddingsManager(provider="offline")
    texts = [c["chunk"] for c in chunks]
    embeddings = emb_manager.embed_documents(texts)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}.")

    print("\nStep 4: Indexing in ChromaDB vector store...")
    vector_store = VectorStoreManager(collection_name="test_verification_store")
    vector_store.clear()
    count = vector_store.add_chunks(chunks, embeddings)
    print(f"Indexed {count} chunks into ChromaDB.")

    print("\nStep 5: Testing RAG Pipeline Retrieval & Generation...")
    llm_service = LLMService(provider="offline")
    pipeline = RAGPipeline(vector_store, emb_manager, llm_service)

    question = "What is the transformer architecture and its encoder-decoder structure?"
    result = pipeline.answer_query(question, top_k=2)

    print("\n--- RAG Results ---")
    print("Question:", question)
    print("Retrieved Sources Count:", len(result["sources"]))
    for i, s in enumerate(result["sources"]):
        print(f"\n[Source {i+1} - Page {s['page']}]:")
        print(s["text"][:150], "...")
    print("\nSynthesized Answer:")
    print(result["answer"][:300], "...")
    print("\nPipeline Verification Passed Successfully!")

if __name__ == "__main__":
    run_test()

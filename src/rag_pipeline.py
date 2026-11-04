"""
LLM and RAG Pipeline service integrating LangChain PromptTemplate,
Multi-provider LLMs (Groq, OpenAI, Google Gemini, Anthropic Claude, or Offline Mock),
and Context Augmentation.
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate


RAG_SYSTEM_PROMPT = """You are an intelligent, helpful RAG (Retrieval-Augmented Generation) assistant.
Your task is to answer the user question accurately based STRICTLY on the provided context passages below.
If the answer cannot be found in the context, clearly state that the provided documents do not contain sufficient information to answer the question. Do not hallucinate or make assumptions beyond the text.

Context passages:
---------------------
{context}
---------------------

User Question: {question}

Detailed Answer:"""


class LLMService:
    def __init__(
        self,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self._init_client()

    def _init_client(self):
        if self.provider == "gemini" and self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            chosen_model = self.model_name or "gemini-3.8-flash"
            self.client = genai.GenerativeModel(chosen_model)
        elif self.provider == "groq" and self.api_key:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            self.model_name = self.model_name or "llama-3.3-70b-versatile"
        elif self.provider == "openai" and self.api_key:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = self.model_name or "gpt-4o-mini"
        elif self.provider == "claude" and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model_name = self.model_name or "claude-3-haiku-20240307"
            except ImportError:
                self.client = None
        else:
            self.client = None

    def generate(self, prompt: str) -> str:
        if self.provider == "gemini" and self.client:
            import google.generativeai as genai
            # List of fallback model names in case the selected version is decommissioned or renamed
            fallback_models = [
                self.model_name,
                "gemini-3.8-flash",
                "gemini-3.8-flash-lite",
                "gemini-3.7-flash",
                "gemini-3.5-flash",
                "gemini-flash-latest",
                "gemini-1.5-flash-latest"
            ]
            seen = set()
            models_to_try = [m for m in fallback_models if m and not (m in seen or seen.add(m))]

            last_err = None
            for m_name in models_to_try:
                try:
                    active_model = genai.GenerativeModel(m_name)
                    response = active_model.generate_content(
                        prompt,
                        generation_config={"temperature": self.temperature}
                    )
                    if hasattr(response, "text") and response.text:
                        return response.text.strip()
                except Exception as err:
                    last_err = err
                    continue

            return (
                f"⚠️ Gemini API Notice: Could not complete generation ({last_err}).\n\n"
                f"Synthesized from ChromaDB context:\n"
                f"{self._extract_fallback_context(prompt)}"
            )

        if self.provider == "groq" and self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    temperature=self.temperature
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                return f"⚠️ Groq Error: {e}"

        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"⚠️ OpenAI Error: {e}"

        # Deterministic context synthesizer
        return (
            "[Demo / Offline Synthesis]\n\n"
            f"{self._extract_fallback_context(prompt)}"
        )

    def _extract_fallback_context(self, prompt: str) -> str:
        try:
            context_part = prompt.split("Context passages:")[-1].split("User Question:")[0].strip()
            return f"Answer based on retrieved passages:\n\n{context_part[:800]}..."
        except Exception:
            return "Unable to parse retrieved passages."


class RAGPipeline:
    def __init__(
        self,
        vector_store,
        embeddings_manager,
        llm_service: LLMService
    ):
        self.vector_store = vector_store
        self.embeddings_manager = embeddings_manager
        self.llm_service = llm_service
        self.prompt_template = PromptTemplate(
            template=RAG_SYSTEM_PROMPT,
            input_variables=["context", "question"]
        )

    def answer_query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Executes end-to-end RAG pipeline:
        1. Encodes query into dense vector
        2. Retrieves top-k relevant chunks from vector store
        3. Formats prompt with retrieved context
        4. Queries LLM for generation
        5. Returns answer along with source attribution
        """
        query_embedding = self.embeddings_manager.embed_query(question)
        retrieved_docs = self.vector_store.retrieve(query_embedding, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found in the vector store. Please upload or index a document first.",
                "sources": []
            }

        context_blocks = []
        for i, doc in enumerate(retrieved_docs):
            context_blocks.append(
                f"[Source {i+1} | Document: {doc['source']} | Page: {doc['page']}]:\n{doc['text']}"
            )
        context_str = "\n\n".join(context_blocks)

        formatted_prompt = self.prompt_template.format(
            context=context_str,
            question=question
        )

        answer = self.llm_service.generate(formatted_prompt)

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "prompt": formatted_prompt
        }

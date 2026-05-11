"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with FAISS
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all 50 questions → generates ≥ 50 LangSmith traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# 1. Load env and set LangSmith BEFORE importing LangChain
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# 2. LLM and Embeddings (Using Gemini via OpenAI Compatibility)
gemini_key = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", 
    google_api_key=gemini_key
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    google_api_key=gemini_key
)

# 3. Build Vectorstore
def build_vectorstore():
    text = Path("data/knowledge_base.txt").read_text(errors='ignore')
    # If the file is empty, put some dummy text so it doesn't crash
    if not text.strip(): text = "Dummy knowledge base text."
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 4. Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])

# 5. Build RAG Chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever

# 6. Traced Query Function
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    return chain.invoke(question)

# 7. Sample Questions
SAMPLE_QUESTIONS =[
    "What are the three main types of machine learning?", "What is overfitting in machine learning?",
    "Explain the bias-variance tradeoff.", "How does regularization prevent overfitting?",
    "What is cross-validation?", "What is backpropagation?",
    "What are Convolutional Neural Networks primarily used for?", "How do LSTM networks address the vanishing gradient problem?",
    "What activation functions are commonly used in neural networks?", "What is the role of pooling layers in CNNs?",
    "What is the transformer architecture?", "What are word embeddings?",
    "What is transfer learning in NLP?", "How does BERT handle language understanding?",
    "What is self-attention in transformers?", "What is GPT and how is it trained?",
    "What is instruction tuning?", "What is RLHF?", "What is chain-of-thought prompting?",
    "What is the context length of GPT-4?", "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG pipeline?", "What is dense retrieval?",
    "Why is chunking strategy important in RAG?", "What advanced RAG techniques exist beyond basic retrieval?",
    "What are vector databases used for?", "What is FAISS?",
    "How do text embeddings capture semantic meaning?", "What is HNSW?",
    "What is hybrid search in vector databases?", "What is LangChain?",
    "What is LangChain Expression Language (LCEL)?", "What is LangGraph?",
    "What memory types does LangChain support?", "What are LangChain retrievers?",
    "What is LangSmith?", "What information do LangSmith traces capture?",
    "What is the LangSmith Prompt Hub?", "How does LangSmith help monitor production LLM applications?",
    "What are LangSmith datasets used for?", "What is RAGAS?",
    "How does RAGAS compute faithfulness?", "What is answer relevancy in RAGAS?",
    "What is context recall in RAGAS?", "What inputs does RAGAS evaluation require?",
    "What is Guardrails AI?", "What is PII and why is it important to detect in LLM responses?",
    "What does structured output validation ensure?", "What is Constitutional AI?",
    "What are common AI safety concerns with LLMs?",
]

# 8. Main
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline (Gemini Edition)")
    print("=" * 60)

    vectorstore = build_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        try:
            answer = ask(chain, question)
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
            print(f"       A: {answer[:100]}...\n")
        except Exception as e:
            print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] ERROR: {e}")
        
        # RATE LIMIT PROTECTOR: 15 RPM max = 1 call every 4 seconds.
        # Sleeping 4.5 seconds to be safe.
        time.sleep(4.5)

    print(f"✅ Traces sent to LangSmith project '{os.environ.get('LANGCHAIN_PROJECT')}'")
    print("   Open https://smith.langchain.com to view traces.")

if __name__ == "__main__":
    main()
"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all 50 questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import time
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# 1. Environment / imports
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langsmith import Client, traceable

# ── 2. LLM / Embeddings ──
gemini_key = os.environ.get("GEMINI_API_KEY")
ls_client = Client() # LangSmith Client

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=gemini_key)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=gemini_key)

# Prompt Hub names (Must be unique - append your name/ID)
PROMPT_V1_NAME = "rag-prompt-v1-longtran" 
PROMPT_V2_NAME = "rag-prompt-v2-longtran"

# Local definitions (used as fallbacks)
SYSTEM_V1 = "You are a concise assistant. Answer in 1-2 sentences using ONLY the context: {context}"
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = "You are an expert AI tutor. Provide a structured, well-organized answer using this context: {context}"
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])


# ── 3. Push prompts to LangSmith Prompt Hub ──
def push_prompts_to_hub(client):
    """Upload both prompt versions to LangSmith Prompt Hub."""
    print("Pushing prompts to LangSmith Hub...")
    try:
        client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1: Concise")
        client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2: Structured")
        print("✅ Pushed prompts to Hub.")
    except Exception as e:
        print(f"ℹ️  Note: {e} (Likely already exist)")


# ── 4. Pull prompts from Prompt Hub (THE PART I MISSED) ──
def pull_prompts_from_hub(client):
    """Download both prompt versions. Fall back to local if Hub fails."""
    prompts = {}

    # Pull V1
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for V1")

    # Pull V2
    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for V2")

    return prompts


# ── 5. A/B routing — deterministic hash ──
def get_prompt_version(request_id: str) -> str:
    """Route request based on MD5 hash of request_id."""
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B query function ──
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, prompt, question: str, version_key: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    
    return {"question": question, "answer": answer, "version": version_key}


# ── 7. Main ──
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    # Infrastructure setup
    push_prompts_to_hub(ls_client)
    prompts = pull_prompts_from_hub(ls_client)

    # Setup Knowledge Base
    text = Path("data/knowledge_base.txt").read_text(errors='ignore')
    chunks = text.split("\n")
    vectorstore = FAISS.from_texts([c for c in chunks if c.strip()], embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Pull the questions from step 1
    # FIX: Dynamically import from a file starting with a number
    import importlib.util
    spec = importlib.util.spec_from_file_location("task1", "01_langsmith_rag_pipeline.py")
    task1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task1)
    SAMPLE_QUESTIONS = task1.SAMPLE_QUESTIONS

    print(f"\nRouting {len(SAMPLE_QUESTIONS)} questions...")
    v1_count = 0
    v2_count = 0

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        prompt = prompts[version_key]

        try:
            res = ask_ab(retriever, prompt, question, version_key)
            tag = "V1" if version_key == PROMPT_V1_NAME else "V2"
            if tag == "V1": v1_count += 1 
            else: v2_count += 1
            
            print(f"[{i+1:02d}/50] [prompt-{tag}] {question[:55]}...")
        except Exception as e:
            print(f"[{i+1:02d}/50] ERROR: {e}")

        time.sleep(5) # Respect your 15 RPM limit

    print(f"\n✅ Routing Summary: V1={v1_count}, V2={v2_count}")

if __name__ == "__main__":
    main()
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
import sys
import hashlib
from pathlib import Path

# ── 1. Environment / imports ────────────────────────────────────────────────
# TODO: load .env and set LangSmith env vars (same as step 1)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"]    = "<your-langsmith-api-key>"
# os.environ["LANGCHAIN_PROJECT"]    = "<your-project-name>"

# TODO: import required libraries
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
# TODO: write PROMPT_V1 — concise, 2-4 sentence answers
# SYSTEM_V1 = (
#     "You are a helpful AI assistant. "
#     "Answer the user's question using ONLY the provided context. "
#     "Keep your answer concise (2-4 sentences). "
#     "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
#     "Context:\n{context}"
# )
# PROMPT_V1 = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_V1),
#     ("human",  "{question}"),
# ])

# TODO: write PROMPT_V2 — structured, expert 3-5 sentence answers
# SYSTEM_V2 = (
#     "You are an expert AI tutor. Provide a structured, accurate answer.\n\n"
#     "Instructions:\n"
#     "1. Read the context carefully.\n"
#     "2. Identify the key facts relevant to the question.\n"
#     "3. Write a clear, well-organized answer (3-5 sentences).\n"
#     "4. State explicitly if the context lacks sufficient information.\n\n"
#     "Context:\n{context}"
# )
# PROMPT_V2 = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_V2),
#     ("human",  "{question}"),
# ])

# Prompt Hub names (change these to your own unique names)
PROMPT_V1_NAME = "my-rag-prompt-v1"   # TODO: choose a unique name
PROMPT_V2_NAME = "my-rag-prompt-v2"   # TODO: choose a unique name


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """
    Upload both prompt versions to LangSmith Prompt Hub.

    Use: client.push_prompt(name, object=template, description="...")
    The 'object' argument must be a ChatPromptTemplate instance.
    """
    # TODO: push PROMPT_V1
    # try:
    #     url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
    #     print(f"✅ Pushed V1 → {url}")
    # except Exception as e:
    #     print(f"⚠️  V1: {e}")

    # TODO: push PROMPT_V2
    # try:
    #     url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
    #     print(f"✅ Pushed V2 → {url}")
    # except Exception as e:
    #     print(f"⚠️  V2: {e}")

    pass  # remove this line when done


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """
    Download both prompt versions from LangSmith Prompt Hub.
    Fall back to local templates if Hub is unavailable.

    Use: client.pull_prompt(name) → returns a ChatPromptTemplate
    """
    prompts = {}

    # TODO: pull PROMPT_V1_NAME, fall back to local PROMPT_V1 on error
    # try:
    #     prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
    #     print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    # except Exception:
    #     prompts[PROMPT_V1_NAME] = PROMPT_V1
    #     print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    # TODO: pull PROMPT_V2_NAME, fall back to local PROMPT_V2 on error
    # try:
    #     prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
    #     print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    # except Exception:
    #     prompts[PROMPT_V2_NAME] = PROMPT_V2
    #     print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.

    Rules:
      even hash → PROMPT_V1_NAME
      odd  hash → PROMPT_V2_NAME

    This is DETERMINISTIC: same request_id always maps to the same version.
    """
    # TODO: compute MD5 hash of request_id, convert to integer
    # hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    # TODO: return V1 name if even, V2 name if odd
    # return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

    pass  # remove this line when done


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    # TODO: copy your build_vectorstore() implementation from step 1
    pass


# ── 7. Traced A/B query function ────────────────────────────────────────────
# TODO: add @traceable decorator with name="ab-rag-query" and tags=["ab-test"]
# @traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Run the RAG chain using the given prompt version.
    Returns a dict: {"question": ..., "answer": ..., "version": ...}

    Steps:
      a) Retrieve top-3 docs with retriever.invoke(question)
      b) Join their page_content into a single context string
      c) Run (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Return the result dict
    """
    # TODO: retrieve docs
    # docs = retriever.invoke(question)
    # context = "\n\n".join(doc.page_content for doc in docs)

    # TODO: run the chain
    # answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    # TODO: return result
    # return {"question": question, "answer": answer, "version": version}

    pass  # remove this line when done


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    # TODO: create LangSmith client
    # client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    # TODO: push both prompts
    # push_prompts_to_hub(client)

    # TODO: pull both prompts from Hub
    # prompts = pull_prompts_from_hub(client)

    # TODO: build vectorstore, retriever, and LLM
    # vectorstore = build_vectorstore()
    # retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    # llm         = ChatOpenAI(...)

    # TODO: loop over all 50 questions with A/B routing
    # from 01_langsmith_rag_pipeline import SAMPLE_QUESTIONS
    # for i, question in enumerate(SAMPLE_QUESTIONS):
    #     request_id  = f"req-{i:04d}"
    #     version_key = get_prompt_version(request_id)
    #     version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
    #     prompt      = prompts[version_key]
    #
    #     result = ask_ab(retriever, llm, prompt, question, version_tag)
    #     print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    # TODO: print routing summary (how many went to V1 vs V2)

    pass  # remove this line when done


if __name__ == "__main__":
    main()

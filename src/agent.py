"""
agent.py

Policy-Aware Customer Support Agent — entry point.

Flow:
    1. Parse all FAQ PDFs in data/ into Q&A chunks (runtime parsing, no
       pre-processing step required).
    2. Embed every chunk's question once at startup (retrieval corpus).
    3. Loop: read a customer question, retrieve the closest FAQ chunk(s),
       ask the LLM to answer grounded in that content, decide confidence
       and respond/escalate, print the structured JSON result.

Run:
    export OPENAI_API_KEY=sk-...
    python src/agent.py
"""

from __future__ import annotations

import json
import os
import sys

from openai import OpenAI

from document_parser import load_all_documents
from retrieval import FAQRetriever

CHAT_MODEL = "gpt-4o-mini"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# Similarity thresholds on retrieval confidence (cosine similarity, 0-1).
# These are starting values, not tuned against real traffic — see README
HIGH_SIMILARITY = 0.55
LOW_SIMILARITY = 0.40

SYSTEM_PROMPT = """You are a customer support assistant for a term life insurance provider.
You must answer ONLY using the policy context provided below. Do not use outside knowledge,
and do not guess or invent policy details that are not present in the context.
If part of the customer's question is not addressed by the specific facts in the context,
explicitly say that part isn't covered rather than inferring an answer from general knowledge.

Respond with a JSON object with exactly these keys:
- "answer": a natural, concise answer to the customer's question, grounded strictly in the
  provided context. If the context does not actually answer the question, say so plainly
  instead of guessing.
- "context_sufficient": true if the provided context genuinely answers the customer's
  question, false if it does not.
"""

def build_user_prompt(question: str, context_blocks: list[str]) -> str:
    context_text = "\n\n---\n\n".join(context_blocks)
    return (
        f"Policy context:\n{context_text}\n\n"
        f"Customer question: {question}\n\n"
        "Respond with the JSON object described in the system prompt."
    )


def classify_confidence(top_similarity: float, context_sufficient: bool) -> str:
    """Combine the retrieval score with the LLM's own sufficiency judgment."""
    if not context_sufficient or top_similarity < LOW_SIMILARITY:
        return "low"
    if top_similarity < HIGH_SIMILARITY:
        return "medium"
    return "high"


def answer_question(client: OpenAI, retriever: FAQRetriever, question: str) -> dict:
    """Run the full retrieve -> generate -> package pipeline for one customer question."""

    # --- Failure case 1: empty / whitespace-only input ---
    if not question or not question.strip():
        return {
            "query": question,
            "category": "N/A",
            "answer": "No question was provided.",
            "source": "N/A",
            "confidence": "low",
            "action": "escalate",
        }

    matches = retriever.top_matches(question, k=6)
    top_match = matches[0]

    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_user_prompt(question, [m.chunk.as_context() for m in matches]),
                },
            ],
        )
        raw = completion.choices[0].message.content
        parsed = json.loads(raw)
        answer_text = parsed.get("answer", "").strip()
        context_sufficient = bool(parsed.get("context_sufficient", False))

    # --- Failure case 2: LLM call fails or returns unparseable output ---
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this is a top-level safety net
        return {
            "query": question,
            "category": top_match.chunk.category,
            "answer": "We're unable to generate an answer right now. Your question has been escalated to a support agent.",
            "source": [f"{m.chunk.source_file} :: {m.chunk.question}" for m in matches],
            "confidence": "low",
            "action": "escalate",
            "_error": str(exc),  # kept for debugging;
        }

    confidence = classify_confidence(top_match.similarity, context_sufficient)
    action = "respond" if confidence in ("high", "medium") else "escalate"

    if action == "escalate" and not answer_text:
        answer_text = "This question needs a human agent to review — it isn't clearly covered by our policy FAQ."

    return {
        "query": question,
        "category": top_match.chunk.category,
        "answer": answer_text,
        "source": [f"{m.chunk.source_file} :: {m.chunk.question}" for m in matches],
        "confidence": confidence,
        "action": action,
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set the OPENAI_API_KEY environment variable before running.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print("Loading and parsing policy FAQ documents...")
    chunks = load_all_documents(DATA_DIR)
    print(f"Loaded {len(chunks)} FAQ entries. Building embeddings index (one-time cost)...")
    retriever = FAQRetriever(client, chunks)
    print("Ready. Type a customer question, or 'exit' to quit.\n")

    while True:
        try:
            question = input("Customer question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        result = answer_question(client, retriever, question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()

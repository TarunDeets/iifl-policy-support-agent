# Policy-Aware Customer Support Agent

A small prototype that answers customer questions about term life insurance,
grounded strictly in a set of policy FAQ documents, with automatic
escalation when the available information isn't enough.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python src/agent.py
```

Place any number of FAQ PDFs (in a "N. Section Heading" / "Q: ... A: ..."
format) in `data/` — they're parsed at runtime, no pre-processing step
needed. Type a question at the prompt; type `exit` to quit.

## 1. How does it work?

At startup, all PDFs in `data/` are parsed into individual Q&A chunks,
each tagged with its FAQ section as a category (85 chunks across 9
sections in this submission). Each chunk's question is embedded once
(`text-embedding-3-small`) into an in-memory index — no vector database.
For each customer question, the same embedding is computed and compared
by cosine similarity against every chunk to retrieve the top 6 matches.
Those matches are passed as grounding context to `gpt-4o-mini`, which
answers strictly from that context and reports whether the context was
actually sufficient. That signal, combined with the retrieval similarity
score, determines the `confidence` and `respond`/`escalate` decision in
the structured JSON output.

## 2. Why this model/approach?

Embeddings handle paraphrasing far better than keyword matching — customers
don't phrase questions the way an FAQ does — while staying simple enough to
build and reason about in a few hours: no vector DB, just numpy cosine
similarity over ~100 short vectors. `gpt-4o-mini` was chosen for
generation because it's fast, cheap, and more than capable for short,
grounded Q&A; embedding quality matters far more than generation model
size here.

## 3. What would I improve before production?

- Retrieval is embeddings-only; adding keyword/exact-match matching
  (e.g. BM25) alongside embeddings would catch cases where semantic
  similarity misses specific terms, section numbers, or exact figures
  that a customer's question happens to share with the FAQ text.
- Compound questions are scored all-or-nothing: if one part of a
  multi-part question lacks grounding, the whole answer is escalated even
  when other parts were well-supported. Splitting into sub-claims scored
  independently would be more precise.
- No caching of embeddings across runs — the FAQ corpus is re-embedded
  on every startup.
- No logging/observability layer to review escalated or low-confidence
  queries over time and catch systematic retrieval gaps.
- No automated test suite — validation so far is manual, question-by-question.

## 4. A security/governance concern in financial services

Customer questions can contain sensitive personal and financial details —
health conditions, income, age, policy numbers — as seen in this
prototype's own test questions. Sending that data to a third-party LLM API
raises data-residency, retention, and PII-handling concerns that matter
a lot more in financial services than in a general chatbot. Before
production, this would need redaction of sensitive fields where possible,
a data-processing agreement with the LLM provider, and clear logging/audit
controls over what's sent externally and why.

## 5. AI coding tools used

Claude was used throughout: to design the retrieval/generation architecture,
write and iterate on all three Python modules, and debug real issues found
by hand-testing sample questions. All fixes were reviewed and tested against real parsed data before being applied.

## Example

**Input:**
```
I moved to Dubai for work — does my Indian term policy still protect my family if something happens to me there?
```

**Output:**
```json
{
  "query": "I moved to Dubai for work — does my Indian term policy still protect my family if something happens to me there?",
  "category": "NRI Term Insurance",
  "answer": "Yes, your Indian term policy can still provide coverage for your family if something happens to you in Dubai, but you may need to provide specific documents such as a foreign death certificate and other records required by the insurer.",
  "source": [
    "term_insurance_faq_sections_7_to_9_detailed.pdf :: Is Indian term insurance valid abroad?",
    "term_insurance_faq_sections_4_to_6_detailed.pdf :: Does term insurance cover death abroad?",
    "term_insurance_faq_sections_7_to_9_detailed.pdf :: Can NRIs buy term insurance in India?"
  ],
  "confidence": "high",
  "action": "respond"
}
```
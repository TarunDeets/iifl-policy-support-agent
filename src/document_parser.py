"""
document_parser.py

Extracts structured Q&A chunks from the term-insurance FAQ PDFs.

Each PDF is organized as a set of numbered sections (e.g. "1. Tax Benefits"),
and each section contains a series of "Q: ... / A: ..." pairs, possibly
wrapped across multiple lines. This module turns that raw text into a flat
list of chunks, one per Q&A pair, each tagged with the section it came from
and the source file it was extracted from.

No pre-processing step is required: PDFs are parsed at runtime every time
the agent starts up.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

from pypdf import PdfReader

# Matches numbered section headers like "1. Tax Benefits" or "10. Policy Management"
SECTION_HEADER_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-Z][A-Za-z0-9 &/'\-]+)\s*$")

# Matches a footer/running-header line repeated on every PDF page, e.g.

FOOTER_RE = re.compile(r"^.*Term Insurance FAQ.*Page\s*\d+\s*$")


@dataclass
class FAQChunk:
    """One retrievable unit: a single FAQ question/answer pair."""

    question: str
    answer: str
    category: str
    source_file: str

    def as_context(self) -> str:
        """How this chunk is presented to the LLM as grounding context."""
        return f"Category: {self.category}\nQ: {self.question}\nA: {self.answer}"


def _extract_raw_text(pdf_path: str) -> str:
    """Pull raw text out of a PDF, page by page, preserving reading order."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def _clean_lines(raw_text: str) -> list[str]:
    """Drop repeated running headers/footers and form-feed noise, keep line breaks."""
    lines = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if FOOTER_RE.match(stripped):
            continue
        lines.append(stripped)
    return lines


def parse_pdf(pdf_path: str) -> list[FAQChunk]:
    """
    Parse a single FAQ PDF into a list of FAQChunk objects.

    Walks the cleaned lines top to bottom, tracking the current section
    ("category") and accumulating multi-line question/answer text until the
    next "Q:", the next "A:", a new section header, or end of file.
    """
    source_file = os.path.basename(pdf_path)
    lines = _clean_lines(_extract_raw_text(pdf_path))

    chunks: list[FAQChunk] = []
    current_category = "General"
    current_question: str | None = None
    current_answer_parts: list[str] = []

    def flush():
        """Save the in-progress Q/A pair (if any) as a chunk."""
        if current_question is not None and current_answer_parts:
            chunks.append(
                FAQChunk(
                    question=current_question.strip(),
                    answer=" ".join(current_answer_parts).strip(),
                    category=current_category,
                    source_file=source_file,
                )
            )

    for line in lines:
        header_match = SECTION_HEADER_RE.match(line)
        if header_match:
            flush()
            current_question = None
            current_answer_parts = []
            current_category = header_match.group(2).strip()
            continue

        if line.startswith("Q:"):
            flush()
            current_question = line[2:].strip()
            current_answer_parts = []
            continue

        if line.startswith("A:"):
            current_answer_parts = [line[2:].strip()]
            continue

        # Continuation line: belongs to whichever of Q/A we're currently building.
        if current_answer_parts:
            current_answer_parts.append(line)
        elif current_question is not None:
            current_question += " " + line

    flush()
    return chunks


def load_all_documents(data_dir: str) -> list[FAQChunk]:
    """
    Parse every PDF found in `data_dir` into one combined pool of chunks.

    This is the "retrieval corpus": all Q&A pairs from all policy/FAQ
    documents, regardless of which file or section they came from. Category
    and source file are preserved on each chunk for the structured output.
    """
    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF documents found in '{data_dir}'. "
            "Place the policy/FAQ PDFs there before running the agent."
        )

    all_chunks: list[FAQChunk] = []
    for path in pdf_paths:
        all_chunks.extend(parse_pdf(path))

    if not all_chunks:
        raise ValueError(
            "Parsed 0 Q&A pairs from the provided PDFs. "
            "Check that the documents follow the expected 'Q: ... / A: ...' format."
        )

    return all_chunks


if __name__ == "__main__":
    # Quick manual check: run `python src/document_parser.py` to sanity-check
    # extraction against whatever PDFs are currently in data/.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chunks = load_all_documents(os.path.join(here, "data"))
    print(f"Parsed {len(chunks)} Q&A chunks from {len(set(c.source_file for c in chunks))} document(s).\n")
    for c in chunks[:3]:
        print(f"[{c.source_file}] ({c.category}) Q: {c.question}")
        print(f"  A: {c.answer[:100]}...\n")

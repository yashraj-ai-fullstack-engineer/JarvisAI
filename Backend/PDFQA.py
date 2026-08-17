"""Single-shot PDF question answering backed by Supabase pgvector."""

from __future__ import annotations

import hashlib
import io
import math
import re
import uuid
from dataclasses import dataclass
from typing import Any

import requests

from Backend.LLMProvider import (
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    EmbeddingUnavailable,
    LocalLLMUnavailable,
    generate_text,
    get_config,
)


MAX_PDF_PAGES = 30
MAX_PDF_BYTES = 5 * 1024 * 1024
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_WORDS = 45
RETRIEVAL_TOP_K = 8
PDF_EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
PDF_EMBEDDING_DIMENSIONS = 2048
PDF_EMBEDDING_TIMEOUT_SECONDS = 60

SUPABASE_URL = get_config("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = get_config("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_SECRET_KEY = get_config("SUPABASE_SECRET_KEY", "")
SUPABASE_ANON_KEY = get_config("SUPABASE_ANON_KEY", "")
SUPABASE_PUBLISHABLE_KEY = get_config("SUPABASE_PUBLISHABLE_KEY", "")


class PDFQAError(RuntimeError):
    """Raised when a PDF cannot be processed or answered safely."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int
    text: str


def answer_transient_pdf_question(
    *,
    pdf_bytes: bytes,
    filename: str,
    question: str,
) -> dict[str, Any]:
    """Answer from an upload without writing the file or its chunks to Supabase."""
    clean_question = " ".join(question.split())
    if not clean_question:
        raise PDFQAError("Ask a question about the PDF before sending.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PDFQAError(f"PDF is too large. The current limit is {MAX_PDF_BYTES // (1024 * 1024)} MB.")

    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pdf_pages(pages)
    if not chunks:
        raise PDFQAError("I could not find readable text in this PDF. Scanned PDFs need OCR support first.")

    embeddings = embed_pdf_texts([chunk["text"] for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise PDFQAError("The embedding service returned an incomplete result.")
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    query_embedding = embed_pdf_texts([clean_question])[0]
    matches = rank_transient_chunks(query_embedding, chunks)
    if not matches:
        raise PDFQAError("No matching PDF context was found after indexing the document.")

    answer = generate_grounded_answer(clean_question, filename, matches)
    return {
        "answer": answer,
        "document": {
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "saved": False,
        },
        "citations": citation_summary(matches),
    }


def remember_pdf_document(
    *,
    pdf_bytes: bytes,
    filename: str,
    question: str,
    user_id: str,
) -> dict[str, Any]:
    """Persist a document only after the user explicitly uses /remember."""
    clean_question = " ".join(question.split()) or f"Summarize {filename}."
    if not user_id:
        raise PDFQAError("Sign in before saving a document to your document memory.")
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PDFQAError(f"PDF is too large. The current limit is {MAX_PDF_BYTES // (1024 * 1024)} MB.")

    pages = extract_pdf_pages(pdf_bytes)
    chunks = chunk_pdf_pages(pages)
    if not chunks:
        raise PDFQAError("I could not find readable text in this PDF. Scanned PDFs need OCR support first.")
    embeddings = embed_pdf_texts([chunk["text"] for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise PDFQAError("The embedding service returned an incomplete result.")
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    document = create_supabase_document(
        user_id=user_id,
        session_id=user_id,
        filename=filename,
        pdf_bytes=pdf_bytes,
        page_count=len(pages),
    )
    insert_supabase_chunks(document["id"], user_id, user_id, chunks)
    query_embedding = embed_pdf_texts([clean_question])[0]
    matches = match_supabase_chunks(
        document_id=document["id"],
        session_id=user_id,
        query_embedding=query_embedding,
        match_count=min(RETRIEVAL_TOP_K, max(1, len(chunks))),
    )
    if not matches:
        raise PDFQAError("The document was saved, but no matching context was found for that question.")
    for match in matches:
        match["filename"] = filename
    answer = generate_grounded_answer(clean_question, filename, matches)
    return {
        "answer": answer,
        "document": {
            "id": document["id"],
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "saved": True,
        },
        "citations": citation_summary(matches),
    }


def answer_saved_document_question(*, user_id: str, question: str) -> dict[str, Any]:
    """Answer a /doc query from every document explicitly saved by this user."""
    clean_question = " ".join(question.split())
    if not clean_question:
        raise PDFQAError("Write a question after /doc.")
    if not user_id:
        raise PDFQAError("Sign in before searching your saved documents.")
    query_embedding = embed_pdf_texts([clean_question])[0]
    matches = match_saved_document_chunks(
        user_id=user_id,
        query_embedding=query_embedding,
        match_count=RETRIEVAL_TOP_K,
    )
    if not matches:
        raise PDFQAError("No saved document matched that question. Upload one with /remember first.")
    answer = generate_grounded_answer(clean_question, "your saved documents", matches)
    return {
        "answer": answer,
        "document": {"filename": "Saved document library", "page_count": 0, "chunk_count": len(matches), "saved": True},
        "citations": citation_summary(matches),
    }


def rank_transient_chunks(query_embedding: list[float], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank temporary chunks in memory so normal uploads never reach Supabase."""
    query_norm = math.sqrt(sum(value * value for value in query_embedding))
    if not query_norm:
        return []
    matches = []
    for chunk in chunks:
        embedding = chunk.get("embedding") or []
        dot_product = sum(a * b for a, b in zip(query_embedding, embedding))
        embedding_norm = math.sqrt(sum(value * value for value in embedding))
        if not embedding_norm:
            continue
        matches.append({
            "chunk_index": chunk["chunk_index"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_text": chunk["text"],
            "metadata": {"title": chunk["title"]},
            "similarity": dot_product / (query_norm * embedding_norm),
        })
    return sorted(matches, key=lambda item: float(item["similarity"]), reverse=True)[:min(RETRIEVAL_TOP_K, len(matches))]


def extract_pdf_pages(pdf_bytes: bytes) -> list[ExtractedPage]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFQAError("Install pypdf first: python -m pip install -r requirements.txt") from exc

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        raise PDFQAError("That file could not be opened as a valid PDF.") from exc

    if reader.is_encrypted:
        raise PDFQAError("Password-protected PDFs are not supported yet.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise PDFQAError(f"Please upload a PDF with {MAX_PDF_PAGES} pages or fewer.")
    if not reader.pages:
        raise PDFQAError("The PDF has no pages.")

    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_pdf_text(page.extract_text() or "")
        if text:
            pages.append(ExtractedPage(page_number=page_number, text=text))

    if not pages:
        raise PDFQAError("I could not find readable text in this PDF. Scanned PDFs need OCR support first.")
    return pages


def clean_pdf_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pdf_pages(pages: list[ExtractedPage]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for page in pages:
        words = page.text.split()
        start = 0
        page_chunk_index = 1
        while start < len(words):
            end = start
            char_count = 0
            while end < len(words):
                next_count = char_count + len(words[end]) + (1 if char_count else 0)
                if next_count > CHUNK_MAX_CHARS and end > start:
                    break
                char_count = next_count
                end += 1

            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append({
                    "chunk_index": len(chunks),
                    "page_start": page.page_number,
                    "page_end": page.page_number,
                    "title": f"Page {page.page_number}, chunk {page_chunk_index}",
                    "text": chunk_text,
                })
                page_chunk_index += 1

            if end >= len(words):
                break
            start = max(start + 1, end - CHUNK_OVERLAP_WORDS)
    return chunks


def embed_pdf_texts(texts: list[str]) -> list[list[float]]:
    api_key = get_config("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        raise EmbeddingUnavailable("OPENROUTER_API_KEY is required for PDF embeddings.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = get_config("OPENROUTER_HTTP_REFERER", OPENROUTER_HTTP_REFERER)
    title = get_config("OPENROUTER_APP_TITLE", OPENROUTER_APP_TITLE)
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    payload = {
        "model": PDF_EMBEDDING_MODEL,
        "input": texts,
        "encoding_format": "float",
    }
    try:
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/embeddings",
            json=payload,
            headers=headers,
            timeout=PDF_EMBEDDING_TIMEOUT_SECONDS,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EmbeddingUnavailable(_http_error("OpenRouter embeddings", response)) from exc
        data = sorted(response.json().get("data", []), key=lambda item: item.get("index", 0))
        embeddings = [item.get("embedding") for item in data]
    except requests.RequestException as exc:
        raise EmbeddingUnavailable(f"OpenRouter embeddings are not reachable: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise EmbeddingUnavailable("OpenRouter returned an unexpected embedding response.") from exc

    if len(embeddings) != len(texts) or any(not isinstance(item, list) for item in embeddings):
        raise EmbeddingUnavailable("OpenRouter returned incomplete embeddings.")
    normalized = [[float(value) for value in embedding] for embedding in embeddings]
    bad_dimensions = [len(embedding) for embedding in normalized if len(embedding) != PDF_EMBEDDING_DIMENSIONS]
    if bad_dimensions:
        raise EmbeddingUnavailable(
            f"Expected {PDF_EMBEDDING_DIMENSIONS}-dimension embeddings, got {bad_dimensions[0]}."
        )
    return normalized


def supabase_headers(prefer: str = "") -> dict[str, str]:
    api_key = (
        get_config("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_SERVICE_ROLE_KEY)
        or get_config("SUPABASE_SECRET_KEY", SUPABASE_SECRET_KEY)
        or get_config("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)
        or get_config("SUPABASE_PUBLISHABLE_KEY", SUPABASE_PUBLISHABLE_KEY)
    )
    if not supabase_url() or not api_key:
        raise PDFQAError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_SECRET_KEY to .env, then run supabase_pdf_rag.sql."
        )
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def supabase_url() -> str:
    return get_config("SUPABASE_URL", SUPABASE_URL).rstrip("/")


def create_supabase_document(
    *,
    user_id: str,
    session_id: str,
    filename: str,
    pdf_bytes: bytes,
    page_count: int,
) -> dict[str, Any]:
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        "filename": filename[:240] or "document.pdf",
        "file_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "page_count": page_count,
        "embedding_model": PDF_EMBEDDING_MODEL,
    }
    response = requests.post(
        f"{supabase_url()}/rest/v1/pdf_documents",
        json=payload,
        headers=supabase_headers("return=representation"),
        timeout=20,
    )
    if not response.ok:
        raise PDFQAError(_http_error("Supabase document insert", response))
    rows = response.json()
    if not rows:
        raise PDFQAError("Supabase did not return the inserted PDF document.")
    return rows[0]


def insert_supabase_chunks(document_id: str, user_id: str, session_id: str, chunks: list[dict[str, Any]]) -> None:
    rows = [
        {
            "document_id": document_id,
            "user_id": user_id,
            "session_id": session_id,
            "chunk_index": chunk["chunk_index"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "chunk_text": chunk["text"],
            "embedding": chunk["embedding"],
            "metadata": {"title": chunk["title"]},
        }
        for chunk in chunks
    ]
    response = requests.post(
        f"{supabase_url()}/rest/v1/pdf_chunks",
        json=rows,
        headers=supabase_headers(),
        timeout=30,
    )
    if not response.ok:
        raise PDFQAError(_http_error("Supabase chunk insert", response))


def match_supabase_chunks(
    *,
    document_id: str,
    session_id: str,
    query_embedding: list[float],
    match_count: int,
) -> list[dict[str, Any]]:
    payload = {
        "p_document_id": document_id,
        "p_session_id": session_id,
        "p_query_embedding": query_embedding,
        "p_match_count": match_count,
    }
    response = requests.post(
        f"{supabase_url()}/rest/v1/rpc/match_pdf_chunks",
        json=payload,
        headers=supabase_headers(),
        timeout=20,
    )
    if not response.ok:
        raise PDFQAError(_http_error("Supabase vector search", response))
    rows = response.json()
    return rows if isinstance(rows, list) else []


def match_saved_document_chunks(
    *,
    user_id: str,
    query_embedding: list[float],
    match_count: int,
) -> list[dict[str, Any]]:
    response = requests.post(
        f"{supabase_url()}/rest/v1/rpc/match_saved_pdf_chunks",
        json={
            "p_user_id": user_id,
            "p_query_embedding": query_embedding,
            "p_match_count": match_count,
        },
        headers=supabase_headers(),
        timeout=20,
    )
    if not response.ok:
        raise PDFQAError(_http_error("Supabase saved-document search", response))
    rows = response.json()
    return rows if isinstance(rows, list) else []


def generate_grounded_answer(question: str, filename: str, matches: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        "[document {filename} | chunk {chunk_index} | page {page_start} | similarity {similarity}]\n{chunk_text}".format(
            filename=match.get("filename") or filename,
            chunk_index=match.get("chunk_index", ""),
            page_start=match.get("page_start", ""),
            similarity=round(float(match.get("similarity") or 0), 4),
            chunk_text=match.get("chunk_text", ""),
        )
        for match in matches
    )
    system = (
        "You answer questions using only the supplied PDF excerpts. "
        "Do not use outside knowledge, chat history, or assumptions. "
        "If the excerpts do not contain the answer, say exactly: "
        f"'I couldn't find that in {filename}.' "
        "Cite the PDF page number after each factual sentence using [page N]. "
        "Be direct and concise."
    )
    prompt = (
        f"PDF filename: {filename}\n"
        f"Question: {question}\n\n"
        f"Retrieved PDF excerpts:\n{context}\n\n"
        "Answer from these excerpts only."
    )
    try:
        return generate_text(prompt=prompt, system=system, temperature=0.05, reasoning="off").strip()
    except LocalLLMUnavailable:
        raise
    except Exception as exc:
        raise PDFQAError("The language model could not answer from the retrieved PDF context.") from exc


def citation_summary(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set()
    citations: list[dict[str, Any]] = []
    for match in matches:
        key = (int(match.get("page_start") or 0), int(match.get("chunk_index") or 0))
        if key in seen:
            continue
        seen.add(key)
        text = str(match.get("chunk_text") or "")
        citations.append({
            "filename": str(match.get("filename") or ""),
            "page": key[0],
            "chunk_index": key[1],
            "preview": text[:180] + ("..." if len(text) > 180 else ""),
        })
    return citations


def _http_error(label: str, response: requests.Response) -> str:
    detail = response.text[:500]
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("message") or payload.get("error") or payload.get("hint") or detail)
    except ValueError:
        pass
    request_id = uuid.uuid4().hex[:8]
    return f"{label} failed with HTTP {response.status_code} ({request_id}): {detail}"

"""Embedding-based RAG over the owner's resume."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from Backend.LLMProvider import (
    EMBEDDING_MODEL,
    EmbeddingUnavailable,
    embed_texts,
    generate_text,
    get_config,
)
from Backend.Paths import DATA_DIR
from Backend.PDFQA import PDF_EMBEDDING_MODEL, embed_pdf_texts, supabase_headers, supabase_url


ROOT = Path(__file__).resolve().parent
RESUME_PATH = ROOT / "Resume_Yashraj.pdf"
INDEX_PATH = DATA_DIR / "OwnerRAG" / "index.json"
CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_WORDS = 35
DEFAULT_TOP_K = 4
INDEX_VERSION = 4
OWNER_PROFILE_KEY = "nexa_owner"
OWNER_PROFILE_SUBJECT = "Yashraj Gupta"
logger = logging.getLogger("nexa.owner_rag")


class OwnerRAGError(RuntimeError):
    pass


def is_owner_question(query: str) -> bool:
    """Detect questions that should be answered from the owner's resume."""
    normalized = " ".join(query.lower().split())
    owner_markers = (
        "your owner",
        "your creator",
        "your developer",
        "who made you",
        "who built you",
        "who created you",
        "who developed you",
        "creator of you",
        "owner of you",
        "made you",
        "built you",
        "created you",
        "your master",
        "master of you",
    )
    profile_markers = (
        "owner profile",
        "owner resume",
        "owner projects",
        "owner skills",
        "owner education",
        "owner experience",
        "creator profile",
        "creator resume",
        "resume of your owner",
        "about your owner",
        "about your creator",
    )
    return (
        "yashraj" in normalized
        or any(marker in normalized for marker in owner_markers + profile_markers)
        or bool(re.search(r"\bwho is (?:the )?master\b", normalized))
    )


def is_creator_identity_question(query: str) -> bool:
    """Detect questions specifically asking who created the assistant."""
    normalized = " ".join(query.lower().split())
    detail_markers = (
        "project",
        "projects",
        "skill",
        "skills",
        "education",
        "experience",
        "work",
        "worked",
        "done",
        "built",
        "developed",
    )
    if any(marker in normalized for marker in detail_markers):
        return False
    creator_markers = (
        "who made you",
        "who built you",
        "who created you",
        "who developed you",
        "creator of you",
        "created you",
        "made you",
        "built you",
        "your creator",
        "your owner",
        "owner of you",
        "your master",
        "master of you",
    )
    return any(marker in normalized for marker in creator_markers)


def is_project_question(query: str) -> bool:
    normalized = " ".join(query.lower().split())
    return bool(re.search(r"\b(?:project|projects)\b", normalized))


def _query_sections(query: str) -> set[str]:
    """Map natural owner questions to resume sections for hybrid retrieval."""
    normalized = " ".join(query.lower().split())
    sections: set[str] = set()
    all_sections = {"Profile", "Education", "Skills", "Experience", "Projects"}

    broad_markers = (
        "everything",
        "all information",
        "all info",
        "full details",
        "complete details",
        "complete profile",
        "entire resume",
        "resume",
    )
    if any(marker in normalized for marker in broad_markers):
        return set(all_sections)

    section_markers = {
        "Profile": (
            "profile",
            "summary",
            "about",
            "who is",
            "contact",
            "email",
            "mobile",
            "phone",
            "linkedin",
            "github",
            "leetcode",
            "portfolio",
        ),
        "Education": (
            "education",
            "study",
            "studied",
            "studies",
            "college",
            "institute",
            "university",
            "school",
            "degree",
            "bachelor",
            "btech",
            "gpa",
            "percentage",
            "class 12",
            "12th",
        ),
        "Skills": (
            "skill",
            "skills",
            "tech stack",
            "technology",
            "technologies",
            "language",
            "languages",
            "backend",
            "frontend",
            "database",
            "cloud",
            "devops",
            "ai tools",
        ),
        "Experience": (
            "experience",
            "work",
            "worked",
            "job",
            "role",
            "company",
            "maersk",
            "internship",
            "fulltime",
            "associate",
            "aiml",
        ),
        "Projects": (
            "project",
            "projects",
            "built",
            "developed",
            "done",
            "jarvis-ai",
            "blog-ai",
            "airbnb",
        ),
    }

    for section, markers in section_markers.items():
        if any(marker in normalized for marker in markers):
            sections.add(section)
    return sections


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:48] or "chunk"


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pdf_pages(pdf_path: Path = RESUME_PATH) -> list[dict[str, Any]]:
    if not pdf_path.exists():
        raise OwnerRAGError(f"Resume PDF was not found at {pdf_path}.")

    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise OwnerRAGError(
            "Install pypdf first: python -m pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if text:
            pages.append({"page": page_number, "text": text})

    if not pages:
        raise OwnerRAGError("No extractable text was found in the resume PDF.")
    return pages


def _chunk_page(page_number: int, text: str) -> list[dict[str, Any]]:
    words = text.split()
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = start
        char_count = 0
        while end < len(words):
            next_word = words[end]
            next_count = char_count + len(next_word) + (1 if char_count else 0)
            if next_count > CHUNK_MAX_CHARS and end > start:
                break
            char_count = next_count
            end += 1

        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append({
                "id": f"resume-p{page_number}-c{len(chunks) + 1}",
                "page": page_number,
                "section": "General",
                "title": f"Page {page_number} chunk {len(chunks) + 1}",
                "text": chunk_text,
            })

        if end >= len(words):
            break
        start = max(start + 1, end - CHUNK_OVERLAP_WORDS)

    return chunks


def _section_between(text: str, start_heading: str, end_headings: tuple[str, ...]) -> str:
    start_match = re.search(rf"(?m)^{re.escape(start_heading)}\s*$", text)
    if not start_match:
        return ""
    start = start_match.end()
    end = len(text)
    for heading in end_headings:
        end_match = re.search(rf"(?m)^{re.escape(heading)}\s*$", text[start:])
        if end_match:
            end = min(end, start + end_match.start())
    return _clean_text(text[start:end])


def _named_chunk(page_number: int, section: str, title: str, text: str) -> dict[str, Any]:
    return {
        "id": f"resume-p{page_number}-{_slug(section)}-{_slug(title)}",
        "page": page_number,
        "section": section,
        "title": title,
        "text": _clean_text(text),
    }


def _project_chunks(page_number: int, text: str) -> list[dict[str, Any]]:
    projects_text = _section_between(text, "Projects", ())
    if not projects_text:
        return []

    lines = [line.strip() for line in projects_text.splitlines() if line.strip()]
    heading_indexes = [
        index for index, line in enumerate(lines)
        if "github" in line.lower() or "live-link" in line.lower()
    ]
    chunks: list[dict[str, Any]] = []
    for position, start in enumerate(heading_indexes):
        end = heading_indexes[position + 1] if position + 1 < len(heading_indexes) else len(lines)
        title = re.sub(r"\s*(?:Github|Live-Link|\|)+.*$", "", lines[start], flags=re.I).strip()
        title = title or lines[start]
        project_text = "\n".join(lines[start:end])
        chunks.append(_named_chunk(page_number, "Projects", title, project_text))
    return chunks


def _structured_page_chunks(page_number: int, text: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    profile = _clean_text(text.split("Education", 1)[0])
    if profile:
        chunks.append(_named_chunk(page_number, "Profile", "Owner identity and summary", profile))

    section_boundaries = {
        "Education": ("Skills", "Experience", "Projects"),
        "Skills": ("Experience", "Projects"),
        "Experience": ("Projects",),
    }
    for section, end_headings in section_boundaries.items():
        section_text = _section_between(text, section, end_headings)
        if section_text:
            for chunk in _chunk_page(page_number, section_text):
                chunk["section"] = section
                chunk["title"] = section
                chunk["id"] = f"resume-p{page_number}-{_slug(section)}-{len(chunks) + 1}"
                chunks.append(chunk)

    chunks.extend(_project_chunks(page_number, text))
    return chunks or _chunk_page(page_number, text)


def _source_mtime(pdf_path: Path = RESUME_PATH) -> float:
    return pdf_path.stat().st_mtime


def _source_sha256(pdf_path: Path = RESUME_PATH) -> str:
    if not pdf_path.exists():
        raise OwnerRAGError(f"Resume PDF was not found at {pdf_path}.")
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()


def _owner_profile_key() -> str:
    return get_config("OWNER_PROFILE_KEY", OWNER_PROFILE_KEY).strip() or OWNER_PROFILE_KEY


def _owner_profile_subject() -> str:
    return get_config("OWNER_PROFILE_SUBJECT", OWNER_PROFILE_SUBJECT).strip() or OWNER_PROFILE_SUBJECT


def _source_chunks() -> list[dict[str, Any]]:
    """Extract stable, section-aware chunks without requiring an embedding service."""
    chunks: list[dict[str, Any]] = []
    for page in _extract_pdf_pages():
        chunks.extend(_structured_page_chunks(page["page"], page["text"]))
    if not chunks:
        raise OwnerRAGError("The resume text could not be split into chunks.")
    for order, chunk in enumerate(chunks):
        chunk["order"] = order
    return chunks


def _index_is_current(index: dict[str, Any], pdf_path: Path = RESUME_PATH) -> bool:
    return (
        index.get("version") == INDEX_VERSION
        and index.get("source_path") == str(pdf_path)
        and index.get("source_mtime") == _source_mtime(pdf_path)
        and index.get("embedding_model") in {EMBEDDING_MODEL, "lexical-fallback"}
        and bool(index.get("chunks"))
    )


def _embed_in_batches(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        embeddings.extend(embed_texts(texts[start:start + batch_size]))
    return embeddings


def build_owner_index(force: bool = False) -> dict[str, Any]:
    """Build a local cache. Embeddings enhance it but are never required."""
    if INDEX_PATH.exists() and not force:
        try:
            existing = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if _index_is_current(existing):
                return existing
        except (OSError, json.JSONDecodeError):
            pass

    chunks = _source_chunks()
    embedding_model = "lexical-fallback"
    if get_config("OWNER_PROFILE_LOCAL_EMBEDDINGS", "false").strip().lower() == "true":
        try:
            embeddings = _embed_in_batches([chunk["text"] for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise OwnerRAGError("Embedding count did not match chunk count.")
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            embedding_model = EMBEDDING_MODEL
        except EmbeddingUnavailable:
            logger.info("owner_profile.local_embedding_unavailable using=lexical_fallback")

    index = {
        "version": INDEX_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_path": str(RESUME_PATH),
        "source_name": RESUME_PATH.name,
        "source_mtime": _source_mtime(),
        "embedding_model": embedding_model,
        "chunk_max_chars": CHUNK_MAX_CHARS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "chunks": chunks,
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def load_owner_index() -> dict[str, Any]:
    """Load the vector index, rebuilding it when missing or stale."""
    if INDEX_PATH.exists():
        try:
            index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            if _index_is_current(index):
                return index
        except (OSError, json.JSONDecodeError):
            pass
    return build_owner_index(force=True)


def _supabase_enabled() -> bool:
    return bool(
        supabase_url()
        and (
            get_config("SUPABASE_SERVICE_ROLE_KEY", "").strip()
            or get_config("SUPABASE_SECRET_KEY", "").strip()
        )
    )


def _supabase_error(operation: str, response: Any) -> OwnerRAGError:
    detail = ""
    try:
        payload = response.json()
        detail = str(payload.get("message") or payload.get("hint") or payload.get("code") or "")
    except Exception:
        detail = str(getattr(response, "text", "") or "")
    return OwnerRAGError(f"Supabase owner-profile {operation} failed: {detail[:300] or 'unknown error'}")


def _supabase_get_current_document() -> dict[str, Any] | None:
    import requests

    response = requests.get(
        f"{supabase_url()}/rest/v1/owner_profile_documents",
        params={
            "profile_key": f"eq.{_owner_profile_key()}",
            "select": "profile_key,source_sha256,embedding_model,source_filename,chunk_count",
            "limit": "1",
        },
        headers=supabase_headers(),
        timeout=15,
    )
    if not response.ok:
        raise _supabase_error("document lookup", response)
    rows = response.json()
    return rows[0] if isinstance(rows, list) and rows else None


def sync_owner_profile(force: bool = False) -> dict[str, Any]:
    """Persist the resume's vectors in Supabase, only when its content changes.

    This runs lazily on the first owner-profile question after a resume update;
    it never writes personal data from ordinary user chats.
    """
    import requests

    if not _supabase_enabled():
        raise OwnerRAGError(
            "Supabase owner-profile storage is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )

    source_sha256 = _source_sha256()
    current = _supabase_get_current_document()
    embedding_model = get_config("PDF_EMBEDDING_MODEL", PDF_EMBEDDING_MODEL)
    if (
        not force
        and current
        and current.get("source_sha256") == source_sha256
        and current.get("embedding_model") == embedding_model
        and int(current.get("chunk_count") or 0) > 0
    ):
        return {"ok": True, "synced": False, "source_sha256": source_sha256}

    chunks = _source_chunks()
    embeddings = embed_pdf_texts([chunk["text"] for chunk in chunks])
    if len(embeddings) != len(chunks):
        raise OwnerRAGError("Owner-profile embedding count did not match resume chunks.")

    profile_key = _owner_profile_key()
    document_payload = {
        "profile_key": profile_key,
        "subject_name": _owner_profile_subject(),
        "source_filename": RESUME_PATH.name,
        "source_sha256": source_sha256,
        "embedding_model": embedding_model,
        "chunk_count": len(chunks),
    }
    # The first sync needs a parent row for the foreign key. Later updates
    # insert new chunks before moving the document's source pointer.
    if current is None:
        initial_document = requests.post(
            f"{supabase_url()}/rest/v1/owner_profile_documents?on_conflict=profile_key",
            json=document_payload,
            headers=supabase_headers("resolution=merge-duplicates"),
            timeout=20,
        )
        if not initial_document.ok:
            raise _supabase_error("initial document upsert", initial_document)
    chunk_rows = [
        {
            "profile_key": profile_key,
            "source_sha256": source_sha256,
            "chunk_key": chunk["id"],
            "chunk_index": int(chunk["order"]),
            "page_number": int(chunk["page"]),
            "section": str(chunk.get("section") or "General"),
            "title": str(chunk.get("title") or ""),
            "chunk_text": str(chunk["text"]),
            "embedding": embedding,
            "metadata": {"source_filename": RESUME_PATH.name},
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    headers = supabase_headers("resolution=merge-duplicates")
    chunks_response = requests.post(
        f"{supabase_url()}/rest/v1/owner_profile_chunks?on_conflict=profile_key,source_sha256,chunk_key",
        json=chunk_rows,
        headers=headers,
        timeout=30,
    )
    if not chunks_response.ok:
        raise _supabase_error("chunk upsert", chunks_response)

    # The document pointer changes only after every vector is present, so a
    # concurrent read always sees a complete previous or complete new index.
    document_response = requests.post(
        f"{supabase_url()}/rest/v1/owner_profile_documents?on_conflict=profile_key",
        json=document_payload,
        headers=supabase_headers("resolution=merge-duplicates,return=representation"),
        timeout=20,
    )
    if not document_response.ok:
        raise _supabase_error("document upsert", document_response)
    return {"ok": True, "synced": True, "source_sha256": source_sha256, "chunks": len(chunks)}


def _retrieve_supabase_context(question: str, top_k: int) -> dict[str, Any]:
    import requests

    sync_owner_profile()
    query_embedding = embed_pdf_texts([question])[0]
    response = requests.post(
        f"{supabase_url()}/rest/v1/rpc/match_owner_profile_chunks",
        json={
            "p_profile_key": _owner_profile_key(),
            "p_query_embedding": query_embedding,
            "p_match_count": min(max(top_k, 1), 12),
        },
        headers=supabase_headers(),
        timeout=20,
    )
    if not response.ok:
        raise _supabase_error("vector search", response)
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise OwnerRAGError("Supabase owner-profile retrieval returned no resume chunks.")
    return {
        "source": RESUME_PATH.name,
        "embedding_model": get_config("PDF_EMBEDDING_MODEL", PDF_EMBEDDING_MODEL),
        "retrieval_mode": "supabase_vector",
        "matches": [
            {
                "id": str(row.get("chunk_key") or row.get("id") or "owner-profile-chunk"),
                "page": int(row.get("page_number") or 1),
                "section": str(row.get("section") or "General"),
                "title": str(row.get("title") or ""),
                "order": int(row.get("chunk_index") or 0),
                "text": str(row.get("chunk_text") or ""),
                "score": round(float(row.get("similarity") or 0), 4),
            }
            for row in rows
            if str(row.get("chunk_text") or "").strip()
        ],
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return -1.0
    return dot / (left_norm * right_norm)


def _lexical_score(question: str, chunk: dict[str, Any]) -> float:
    """Stable fallback for a small owner profile when vectors are unavailable."""
    query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    chunk_terms = re.findall(r"[a-z0-9]+", str(chunk.get("text") or "").lower())
    if not query_terms or not chunk_terms:
        return 0.0
    matched = sum(1 for term in query_terms if term in chunk_terms)
    heading = f"{chunk.get('section', '')} {chunk.get('title', '')}".lower()
    heading_matches = sum(1 for term in query_terms if term in heading)
    return (matched / len(query_terms)) + (heading_matches * 0.15)


def _identity_anchor(index: dict[str, Any]) -> dict[str, Any] | None:
    for chunk in index.get("chunks", []):
        if chunk.get("section") == "Profile":
            return {
                "id": chunk["id"],
                "page": chunk["page"],
                "section": chunk.get("section", "Profile"),
                "title": chunk.get("title", "Owner identity"),
                "text": chunk["text"],
                "score": 1.0,
                "reason": "profile_anchor",
            }
    return None


def retrieve_owner_context(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Retrieve owner facts from Supabase, with a local lexical safety net."""
    cleaned_question = " ".join(question.split())
    if not cleaned_question:
        raise OwnerRAGError("Question cannot be empty.")

    requested_sections = _query_sections(cleaned_question)
    if len(requested_sections) >= 3:
        top_k = max(top_k, 8)

    retrieval: dict[str, Any] | None = None
    if _supabase_enabled():
        try:
            retrieval = _retrieve_supabase_context(cleaned_question, top_k)
        except Exception as exc:
            # The owner profile remains usable during a temporary Supabase or
            # embedding outage. This never falls back to unrelated web data.
            logger.warning("owner_profile.supabase_unavailable error_type=%s", type(exc).__name__)

    if retrieval is None:
        index = load_owner_index()
        chunks = index["chunks"]
        vector_query: list[float] | None = None
        if all(chunk.get("embedding") for chunk in chunks):
            try:
                vector_query = embed_texts([cleaned_question])[0]
            except EmbeddingUnavailable:
                vector_query = None
        retrieval = {
            "source": index["source_name"],
            "embedding_model": index["embedding_model"],
            "retrieval_mode": "local_vector" if vector_query else "local_lexical_fallback",
            "matches": [],
        }
        scored = []
        for chunk in chunks:
            score = (
                _cosine_similarity(vector_query, chunk.get("embedding") or [])
                if vector_query
                else _lexical_score(cleaned_question, chunk)
            )
            scored.append({
                "id": chunk["id"],
                "page": chunk["page"],
                "section": chunk.get("section", "General"),
                "title": chunk.get("title", ""),
                "order": chunk.get("order", 0),
                "text": chunk["text"],
                "score": round(score, 4),
            })
        retrieval["matches"] = sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    scored = list(retrieval["matches"])
    matches = scored[:top_k]
    section_boost = 0.18
    for chunk in scored:
        section = chunk.get("section", "General")
        if section in requested_sections:
            chunk["score"] = round(float(chunk.get("score") or 0) + section_boost, 4)
    matches = sorted(matches, key=lambda item: item["score"], reverse=True)
    matches = _ensure_section_coverage(matches, scored, requested_sections, top_k)
    should_anchor_identity = (
        is_creator_identity_question(cleaned_question)
        or not requested_sections
        or "Profile" in requested_sections
    )
    if is_owner_question(cleaned_question) and should_anchor_identity:
        anchor = next((match for match in scored if match.get("section") == "Profile"), None)
        if anchor and all(match["id"] != anchor["id"] for match in matches):
            matches = [{**anchor, "score": max(float(anchor.get("score") or 0), 1.0), "reason": "profile_anchor"}, *matches[:max(0, top_k - 1)]]

    return {
        "ok": True,
        "question": cleaned_question,
        "source": retrieval["source"],
        "embedding_model": retrieval["embedding_model"],
        "retrieval_mode": retrieval["retrieval_mode"],
        "requested_sections": sorted(requested_sections),
        "matches": matches,
    }


def _ensure_section_coverage(
    matches: list[dict[str, Any]],
    scored: list[dict[str, Any]],
    requested_sections: set[str],
    top_k: int,
) -> list[dict[str, Any]]:
    if not requested_sections:
        return matches

    selected_by_id = {match["id"]: match for match in matches}
    for section in ("Profile", "Education", "Skills", "Experience", "Projects"):
        if section not in requested_sections:
            continue
        if any(match.get("section") == section for match in selected_by_id.values()):
            continue
        section_candidates = [
            candidate for candidate in scored
            if candidate.get("section") == section
        ]
        if section_candidates:
            best = max(section_candidates, key=lambda item: item["score"])
            selected_by_id[best["id"]] = best

    selected = list(selected_by_id.values())
    if len(selected) <= top_k:
        return sorted(selected, key=lambda item: item["score"], reverse=True)

    required_ids = set()
    for section in requested_sections:
        section_matches = [
            match for match in selected
            if match.get("section") == section
        ]
        if section_matches:
            required_ids.add(max(section_matches, key=lambda item: item["score"])["id"])

    required = [match for match in selected if match["id"] in required_ids]
    optional = sorted(
        [match for match in selected if match["id"] not in required_ids],
        key=lambda item: item["score"],
        reverse=True,
    )
    return sorted([*required, *optional[:max(0, top_k - len(required))]], key=lambda item: item["score"], reverse=True)


def answer_owner_question(question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Answer using only the relevant content retrieved from the resume PDF."""
    requested_sections = _query_sections(question)
    project_question = is_project_question(question)
    if project_question:
        top_k = max(top_k, 8)
    elif len(requested_sections) >= 3:
        top_k = max(top_k, 8)
    elif requested_sections:
        top_k = max(top_k, 5)
    retrieval = retrieve_owner_context(question, top_k=top_k)

    context = "\n\n".join(
        f"[{match['id']} | page {match['page']} | score {match['score']}]\n{match['text']}"
        for match in retrieval["matches"]
    )
    creator_relation_instruction = (
        f"Nexa's configured owner-profile subject is {_owner_profile_subject()}. "
        "For owner, creator, developer, or master identity questions, identify "
        "that subject directly. All biographical details after that identity "
        "statement must be grounded in the supplied resume excerpts."
        if is_creator_identity_question(question)
        else ""
    )
    system = (
        "Answer questions from the supplied resume excerpts. The excerpts are "
        "the only permitted source of personal facts. Do not use general "
        "knowledge, chat history, application configuration, or assumptions. "
        "For an owner, creator, developer, or master identity question, the "
        "configured owner-profile subject is an allowed application identity. "
        "Write resume facts "
        "about the subject in third person, never as the assistant's own "
        "experience. If the excerpts do not support the answer, say exactly: "
        "'I couldn't find that in Resume_Yashraj.pdf.' Be concise, avoid "
        "inventing details, and add [page N] after each factual sentence. If "
        "the answer is supported, do not add a not-found statement."
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Resume context:\n{context}\n\n"
        f"{creator_relation_instruction}\n\n"
        "Answer from the resume context only. Give the requested details "
        "directly, but include only claims supported by an excerpt."
    )
    answer = generate_text(prompt=prompt, system=system, temperature=0.1)
    pages = sorted({match["page"] for match in retrieval["matches"]})
    retrieval["answer"] = answer.strip()
    retrieval["source_pages"] = pages
    return retrieval


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build and query the owner resume RAG index.")
    parser.add_argument("command", choices=["build", "sync", "retrieve", "ask"])
    parser.add_argument("question", nargs="*", help="Question for retrieve or ask.")
    parser.add_argument("--force", action="store_true", help="Rebuild the index even if it is current.")
    args = parser.parse_args()

    try:
        if args.command == "build":
            index = build_owner_index(force=args.force)
            print(json.dumps({
                "ok": True,
                "index": str(INDEX_PATH),
                "chunks": len(index["chunks"]),
                "embedding_model": index["embedding_model"],
            }, indent=2))
        elif args.command == "sync":
            print(json.dumps(sync_owner_profile(force=args.force), indent=2))
        elif args.command == "retrieve":
            print(json.dumps(retrieve_owner_context(" ".join(args.question)), indent=2))
        else:
            print(json.dumps(answer_owner_question(" ".join(args.question)), indent=2))
    except (OwnerRAGError, EmbeddingUnavailable) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))


if __name__ == "__main__":
    _main()

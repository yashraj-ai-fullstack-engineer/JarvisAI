"""Private, reproducible PDF exports for completed deep-research runs."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import requests
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from Backend.LLMProvider import get_config
from Backend.Paths import DATA_DIR


class ResearchPdfError(RuntimeError):
    """A safe error while creating or serving a research export."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned[:100] or "research-report"


def report_version_hash(run: dict[str, Any]) -> str:
    """Hash only canonical export inputs, never transient UI state."""
    payload = {
        "id": str(run.get("id") or ""),
        "question": str(run.get("question") or ""),
        "report": str(run.get("report") or ""),
        "sources": list(run.get("evidence_manifest") or []),
        "source_ids": list(run.get("source_ids") or []),
        "source_errors": list(run.get("source_errors") or []),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _font_names() -> tuple[str, str]:
    """Prefer a Unicode font when deployment provides one; remain portable."""
    candidates = [
        (get_config("RESEARCH_PDF_FONT_PATH", "").strip(), get_config("RESEARCH_PDF_BOLD_FONT_PATH", "").strip()),
        (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for normal_path, bold_path in candidates:
        if normal_path and bold_path and Path(normal_path).is_file() and Path(bold_path).is_file():
            if "NexaResearchPdf" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("NexaResearchPdf", normal_path))
                pdfmetrics.registerFont(TTFont("NexaResearchPdfBold", bold_path))
            return "NexaResearchPdf", "NexaResearchPdfBold"
    return "Helvetica", "Helvetica-Bold"


def _pdf_text(value: Any, *, unicode_font: bool) -> str:
    text = str(value or "")
    text = re.sub(r"!?(?:\[([^\]]*)\])\((https?://[^)]+)\)", r"\1 <\2>", text)
    text = re.sub(r"[`*_>#]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not unicode_font:
        text = text.encode("latin-1", "replace").decode("latin-1")
    return html.escape(text)


def _footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d8dde8"))
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFillColor(colors.HexColor("#5b6474"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(18 * mm, 9 * mm, "Nexa Research Export")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {document.page}")
    canvas.restoreState()


def render_research_pdf(run: dict[str, Any], target: Path) -> None:
    """Render server-owned report data to a non-interactive PDF."""
    report = str(run.get("report") or "").strip()
    if not report:
        raise ResearchPdfError("This research run has no completed report to export.")
    normal_font, bold_font = _font_names()
    unicode_font = normal_font != "Helvetica"
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ResearchTitle", parent=styles["Title"], fontName=bold_font, fontSize=20, leading=25, textColor=colors.HexColor("#171d2d"), spaceAfter=7 * mm)
    heading = ParagraphStyle("ResearchHeading", parent=styles["Heading2"], fontName=bold_font, fontSize=13, leading=17, textColor=colors.HexColor("#29385f"), spaceBefore=5 * mm, spaceAfter=2 * mm)
    body = ParagraphStyle("ResearchBody", parent=styles["BodyText"], fontName=normal_font, fontSize=9.5, leading=14, alignment=TA_LEFT, textColor=colors.HexColor("#252b38"), spaceAfter=2.5 * mm, wordWrap="CJK")
    small = ParagraphStyle("ResearchSmall", parent=body, fontSize=8.5, leading=12, textColor=colors.HexColor("#5b6474"))

    question = _pdf_text(run.get("question") or "Research report", unicode_font=unicode_font)
    generated = _pdf_text(run.get("completed_at") or _utc_now(), unicode_font=unicode_font)
    topic = _pdf_text(run.get("topic") or "Research", unicode_font=unicode_font)
    story = [
        Paragraph("Nexa Research Report", title),
        Paragraph(f"<b>Question:</b> {question}", body),
        Paragraph(f"<b>Topic:</b> {topic}<br/><b>Generated:</b> {generated}", small),
        Spacer(1, 3 * mm),
    ]

    current_heading = "Report"
    story.append(Paragraph(current_heading, heading))
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched_heading = re.match(r"^#{1,3}\s+(.+)$", line)
        if matched_heading:
            current_heading = _pdf_text(matched_heading.group(1), unicode_font=unicode_font)
            story.append(Paragraph(current_heading, heading))
            continue
        if line.startswith(("- ", "* ")):
            story.append(Paragraph("• " + _pdf_text(line[2:], unicode_font=unicode_font), body))
            continue
        # Tables retain their data even when a full tabular layout would not
        # fit safely on a narrow page. A later chart/table export can build on
        # the structured report rather than scraping this PDF.
        if "|" in line:
            story.append(Paragraph(_pdf_text(" | ".join(part.strip() for part in line.strip("|").split("|")), unicode_font=unicode_font), small))
            continue
        story.append(Paragraph(_pdf_text(line, unicode_font=unicode_font), body))

    manifest = list(run.get("evidence_manifest") or [])
    if manifest:
        story.append(Paragraph("Verified source trace", heading))
        for source in manifest:
            label = _pdf_text(source.get("source_label") or source.get("source_id") or "Source", unicode_font=unicode_font)
            source_title = _pdf_text(source.get("title") or "Untitled source", unicode_font=unicode_font)
            url = _pdf_text(source.get("url") or "", unicode_font=unicode_font)
            reference = _pdf_text(source.get("reference") or "", unicode_font=unicode_font)
            date = _pdf_text(source.get("published_at") or "", unicode_font=unicode_font)
            story.append(Paragraph(f"<b>{reference} {source_title}</b><br/>{label}{(' - ' + date) if date else ''}<br/>{url}", small))

    errors = list(run.get("source_errors") or [])
    if errors:
        story.append(Paragraph("Retrieval limitations", heading))
        for error in errors:
            story.append(Paragraph("• " + _pdf_text(error, unicode_font=unicode_font), small))

    target.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(str(target), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=20 * mm, title="Nexa Research Report", author="Nexa")
    document.build(story, onFirstPage=_footer, onLaterPages=_footer)


def render_research_pdf_bytes(run: dict[str, Any]) -> bytes:
    """Generate a one-time PDF response without persisting an export."""
    descriptor, temporary_name = tempfile.mkstemp(prefix="nexa-research-", suffix=".pdf")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        render_research_pdf(run, temporary)
        return temporary.read_bytes()
    finally:
        temporary.unlink(missing_ok=True)


def research_export_filename(run: dict[str, Any]) -> str:
    topic = _safe_filename(str(run.get("topic") or "report"))
    run_id = _safe_filename(str(run.get("id") or "report"))[:8]
    return f"nexa-research-{topic}-{run_id}.pdf"


class _LocalStorage:
    name = "local"

    def __init__(self) -> None:
        self.root = DATA_DIR / "ResearchExports"

    def _path(self, key: str) -> Path:
        parts = PurePosixPath(key).parts
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ResearchPdfError("Invalid private PDF storage key.")
        destination = (self.root.joinpath(*parts)).resolve()
        if self.root.resolve() not in destination.parents:
            raise ResearchPdfError("Invalid private PDF storage path.")
        return destination

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put(self, key: str, content: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    def local_path(self, key: str) -> Path:
        path = self._path(key)
        if not path.is_file():
            raise ResearchPdfError("The cached research PDF is no longer available.")
        return path


class _SupabaseStorage:
    name = "supabase"

    def __init__(self) -> None:
        self.url = get_config("SUPABASE_URL", "").strip().rstrip("/")
        self.key = get_config("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        self.bucket = get_config("RESEARCH_EXPORT_BUCKET", "research-exports").strip()
        if not self.url or not self.key:
            raise ResearchPdfError("Supabase PDF storage requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    def put(self, key: str, content: bytes) -> None:
        response = requests.post(
            f"{self.url}/storage/v1/object/{self.bucket}/{key}",
            headers={"Authorization": f"Bearer {self.key}", "apikey": self.key, "Content-Type": "application/pdf", "x-upsert": "true"},
            data=content,
            timeout=30,
        )
        if not response.ok:
            raise ResearchPdfError("Private PDF storage could not save the export.")

    def signed_url(self, key: str) -> str:
        response = requests.post(
            f"{self.url}/storage/v1/object/sign/{self.bucket}/{key}",
            headers={"Authorization": f"Bearer {self.key}", "apikey": self.key, "Content-Type": "application/json"},
            json={"expiresIn": 300},
            timeout=15,
        )
        if not response.ok:
            raise ResearchPdfError("Private PDF storage could not open the export.")
        signed = str(response.json().get("signedURL") or "")
        if not signed:
            raise ResearchPdfError("Private PDF storage did not return a signed URL.")
        return signed if signed.startswith("http") else f"{self.url}/storage/v1{signed}"


def _storage() -> _LocalStorage | _SupabaseStorage:
    configured = get_config("RESEARCH_EXPORT_STORAGE", "local").strip().lower()
    if configured == "local":
        return _LocalStorage()
    if configured == "supabase":
        return _SupabaseStorage()
    raise ResearchPdfError("RESEARCH_EXPORT_STORAGE must be 'local' or 'supabase'.")


def create_or_reuse_export(run: dict[str, Any]) -> dict[str, Any]:
    """Return private-export metadata, rendering only when report content changed."""
    if str(run.get("status") or "") != "completed":
        raise ResearchPdfError("Only completed research reports can be exported as PDF.")
    user_id = str(run.get("user_id") or "")
    run_id = str(run.get("id") or "")
    if not user_id or not run_id:
        raise ResearchPdfError("This research report has incomplete ownership metadata.")
    version = report_version_hash(run)
    storage = _storage()
    cached = dict(run.get("pdf_export") or {})
    if cached.get("version_hash") == version and cached.get("storage") == storage.name:
        if storage.name != "local" or storage.exists(str(cached.get("storage_key") or "")):
            return cached

    user_partition = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:20]
    key = f"research-exports/{user_partition}/{run_id}/{version}.pdf"
    temp_root = DATA_DIR / "tmp" / "pdfs"
    temp_root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="research-", suffix=".pdf", dir=temp_root)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        render_research_pdf(run, temporary)
        content = temporary.read_bytes()
        storage.put(key, content)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "version_hash": version,
        "storage": storage.name,
        "storage_key": key,
        "byte_size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "created_at": _utc_now(),
        "filename": f"nexa-research-{_safe_filename(run.get('topic') or 'report')}-{run_id[:8]}.pdf",
    }


def local_export_path(export: dict[str, Any]) -> Path:
    if export.get("storage") != "local":
        raise ResearchPdfError("This PDF is stored remotely.")
    return _LocalStorage().local_path(str(export.get("storage_key") or ""))


def remote_export_url(export: dict[str, Any]) -> str:
    if export.get("storage") != "supabase":
        raise ResearchPdfError("This PDF is not stored remotely.")
    return _SupabaseStorage().signed_url(str(export.get("storage_key") or ""))

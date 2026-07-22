from dataclasses import dataclass
from pathlib import Path

import docx2txt
from pypdf import PdfReader


@dataclass
class LoadedPage:
    text: str
    page: int | None


def load_file(path: str | Path) -> list[LoadedPage]:
    """Extract text from a document, split by page where the format has pages.

    Supports PDF, DOCX, TXT and Markdown. Raises ValueError for anything else.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in (".txt", ".md"):
        return _load_text(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> list[LoadedPage]:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(LoadedPage(text=text, page=i))
    return pages


def _load_docx(path: Path) -> list[LoadedPage]:
    text = docx2txt.process(str(path))
    return [LoadedPage(text=text, page=None)] if text.strip() else []


def _load_text(path: Path) -> list[LoadedPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [LoadedPage(text=text, page=None)] if text.strip() else []

from __future__ import annotations

import re

from simbrain.models import DocumentPage, DocumentRecord


_CONTROL_CHARS = re.compile(r"[\x01-\x08\x0b-\x1f\x7f-\x9f]")
_PAGE_MARKERS = re.compile(r"--\s*\d+\s+of\s+\d+\s*--")
_MULTI_SPACES = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINES = re.compile(r"\n{3,}")


def _is_single_cjk(token: str) -> bool:
    return len(token) == 1 and bool(re.fullmatch(r"[一-鿿]", token))


def _normalize_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    tokens = [t for t in stripped.split() if t]
    if len(tokens) >= 3 and all(_is_single_cjk(t) for t in tokens):
        return "".join(tokens)
    return stripped


def _dedupe_lines(text: str) -> str:
    deduped, last = [], ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            deduped.append("")
            last = ""
            continue
        if s == last:
            continue
        deduped.append(s)
        last = s
    return "\n".join(deduped)


def clean_text(text: str) -> str:
    t = text.replace("\x00", "")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("　", " ").replace("\xa0", " ")
    t = _CONTROL_CHARS.sub("", t)
    t = _PAGE_MARKERS.sub("\n", t)
    t = "\n".join(_normalize_line(l) for l in t.splitlines())
    t = _dedupe_lines(t)
    t = _MULTI_SPACES.sub(" ", t)
    t = _MULTI_NEWLINES.sub("\n\n", t)
    return t.strip()


def clean_docs(documents: list[DocumentRecord]) -> list[DocumentRecord]:
    return [
        DocumentRecord(
            source_path=d.source_path,
            file_name=d.file_name,
            file_type=d.file_type,
            title=clean_text(d.title),
            pages=[DocumentPage(page_number=p.page_number, text=clean_text(p.text)) for p in d.pages],
            metadata=d.metadata,
        )
        for d in documents
    ]

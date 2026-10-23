"""知识库切分：按 Markdown 二级标题切语义单元，超长再按段落带重叠切。

为什么按 `##` 切而不是定长滑窗：
售后知识的边界天然是"一条政策"，定长切会把一条政策的条件和例外切到两个 chunk 里，
召回时只召回半条，答案就是错的 —— 这是 RAG 在业务知识库上最常见的失败模式。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    title: str
    domain: str
    section: str
    text: str
    order: int = 0
    updated_at: str | None = None
    valid_until: str | None = None
    version: str = "v1"


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONT.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    parts: list[str] = []
    buf = ""
    for para in paras:
        if len(buf) + len(para) + 1 <= max_chars:
            buf = f"{buf}\n{para}".strip()
        else:
            if buf:
                parts.append(buf)
            buf = (buf[-overlap:] + "\n" + para).strip() if overlap and buf else para
    if buf:
        parts.append(buf)
    return parts


def chunk_document(doc_id: str, raw: str, max_chars: int = 420, overlap: int = 70) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    title = meta.get("title", doc_id)
    domain = meta.get("domain", "general")

    sections: list[tuple[str, str]] = []
    current_title = "概述"
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if buf and any(x.strip() for x in buf):
                sections.append((current_title, "\n".join(buf).strip()))
            current_title = line[3:].strip()
            buf = []
        else:
            buf.append(line)
    if buf and any(x.strip() for x in buf):
        sections.append((current_title, "\n".join(buf).strip()))

    if not sections:
        sections = [("概述", body.strip())]

    chunks: list[Chunk] = []
    for idx, (section, text) in enumerate(sections):
        for sub_idx, piece in enumerate(_split_long(text, max_chars, overlap)):
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}#{idx}-{sub_idx}",
                    title=f"{title} · {section}" if section != "概述" else title,
                    domain=domain,
                    section=section,
                    text=piece,
                    order=len(chunks),
                    updated_at=meta.get("updated_at"),
                    valid_until=meta.get("valid_until"),
                    version=meta.get("version", "v1"),
                )
            )
    return chunks

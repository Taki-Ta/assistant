import hashlib
import re
from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from zhiyin.core.protocols import IndexableDocument
from zhiyin.db.models import Chunk

from .models import MAX_CHUNK_LENGTH

MAX_MARKDOWN_TITLE_LEVEL = 6
TITLE_REGEX = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_REGEX = re.compile(r"^ {0,3}(`{3,}|~{3,})")


@dataclass
class _Block:
    content: str
    start_line: int
    end_line: int


@dataclass
class _Section:
    level: int
    headings: tuple[str, ...]
    lines: list[tuple[int, str]]


def split_document(
    document: IndexableDocument, max_chunk_length: int = MAX_CHUNK_LENGTH
) -> list[Chunk]:
    """按 Markdown 标题和段落切块，索引字段表示从 1 开始的源文件行号。"""
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length 必须大于 0")
    if not document.content:
        return []

    chunks: list[Chunk] = []
    for section in _split_sections(document.content):
        blocks = _split_paragraphs(section.lines, max_chunk_length)
        for group in _pack_blocks(blocks, max_chunk_length):
            content = "\n\n".join(block.content for block in group)
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            sort_index = len(chunks)
            identity = f"{document.id}-{sort_index}-{content_hash}"
            chunks.append(
                Chunk(
                    id=uuid5(NAMESPACE_URL, identity),
                    document_id=document.id,
                    sort_index=sort_index,
                    level=section.level,
                    headings=section.headings,
                    content=content,
                    start_line=group[0].start_line,
                    end_line=group[-1].end_line,
                    hash=content_hash,
                )
            )
    return chunks


def _split_sections(content: str) -> list[_Section]:
    sections: list[_Section] = []
    headings: list[str] = []
    current = _Section(level=0, headings=(), lines=[])
    fence_marker: str | None = None

    for line_number, line in enumerate(content.splitlines(), start=1):
        fence_match = FENCE_REGEX.match(line)
        heading_match = None

        if fence_marker is None:
            if fence_match:
                fence_marker = fence_match.group(1)[0]
            else:
                heading_match = TITLE_REGEX.match(line)
        elif fence_match and fence_match.group(1)[0] == fence_marker:
            fence_marker = None

        if heading_match:
            if current.lines:
                sections.append(current)
            level = len(heading_match.group(1))
            headings = headings[: level - 1]
            headings.append(heading_match.group(2).strip())
            current = _Section(level=level, headings=tuple(headings), lines=[])

        current.lines.append((line_number, line))

    if current.lines:
        sections.append(current)
    return sections


def _split_paragraphs(
    lines: list[tuple[int, str]], max_chunk_length: int
) -> list[_Block]:
    blocks: list[_Block] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if not current:
            return
        block = _Block(
            content="\n".join(line for _, line in current),
            start_line=current[0][0],
            end_line=current[-1][0],
        )
        blocks.extend(_split_long_block(block, max_chunk_length))
        current.clear()

    for line_number, line in lines:
        if line.strip():
            current.append((line_number, line))
        else:
            flush()
    flush()
    return blocks


def _split_long_block(block: _Block, max_chunk_length: int) -> list[_Block]:
    if len(block.content) <= max_chunk_length:
        return [block]
    return [
        _Block(
            content=block.content[start : start + max_chunk_length],
            start_line=block.start_line,
            end_line=block.end_line,
        )
        for start in range(0, len(block.content), max_chunk_length)
    ]


def _pack_blocks(blocks: list[_Block], max_chunk_length: int) -> list[list[_Block]]:
    groups: list[list[_Block]] = []
    current: list[_Block] = []
    current_length = 0

    for block in blocks:
        separator_length = 2 if current else 0
        next_length = current_length + separator_length + len(block.content)
        if current and next_length > max_chunk_length:
            groups.append(current)
            current = []
            current_length = 0
            separator_length = 0
        current.append(block)
        current_length += separator_length + len(block.content)

    if current:
        groups.append(current)
    return groups

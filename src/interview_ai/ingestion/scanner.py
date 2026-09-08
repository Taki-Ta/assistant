import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import MAX_CHUNK_LENGTH, TIME_ZONE_LOCAL, Chunk, MarkdownFile

IGNORE_DIRS = [".git", ".venv", "assistant"]

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


def scan_path(path: Path) -> list[MarkdownFile]:
    """接收一个路径,返回路径下所有md格式的文件"""
    if not Path.exists(path):
        raise FileNotFoundError("文件不存在")
    if Path.is_file(path):
        if path.suffix == ".md":
            return [get_file_info_from_path(path)]
        else:
            return []
    return _scan_file(path, [])


def get_file_info_from_path(path: Path) -> MarkdownFile:
    """根据文件路径获取文件信息"""
    if not Path.exists(path) or not Path.is_file(path):
        raise FileNotFoundError("文件不存在")
    with open(path, "r", encoding="utf-8") as f:
        info = path.stat()
        content = f.read()
        content_bytes = content.encode("utf-8")
        return MarkdownFile(
            path,
            path.name,
            datetime.fromtimestamp(info.st_birthtime, TIME_ZONE_LOCAL),
            datetime.fromtimestamp(info.st_mtime, TIME_ZONE_LOCAL),
            info.st_size,
            content,
            hashlib.sha256(content_bytes).hexdigest(),
        )


def split_file(
    file: MarkdownFile, max_chunk_length: int = MAX_CHUNK_LENGTH
) -> list[Chunk]:
    """按 Markdown 标题和段落切块，索引字段表示从 1 开始的源文件行号。"""
    if max_chunk_length <= 0:
        raise ValueError("max_chunk_length 必须大于 0")
    if not file.content:
        return []

    chunks: list[Chunk] = []
    for section in _split_sections(file.content):
        blocks = _split_paragraphs(section.lines, max_chunk_length)
        for group in _pack_blocks(blocks, max_chunk_length):
            content = "\n\n".join(block.content for block in group)
            chunks.append(
                Chunk(
                    document_id=file.id,
                    sort_index=len(chunks),
                    level=section.level,
                    headings=section.headings,
                    content=content,
                    start_line=group[0].start_line,
                    end_line=group[-1].end_line,
                    hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
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


def _scan_file(
    path: Path, files: list[MarkdownFile] | None = None
) -> list[MarkdownFile]:
    """接收一个文件夹路径,返回路径下所有md格式的文件"""
    if files is None:
        files = []
    for child in path.rglob("*.md", case_sensitive=False):
        try:
            relative_parts = child.relative_to(path).parts
            parent_parts = relative_parts[:-1]
            if any(part in IGNORE_DIRS for part in parent_parts):
                continue
            file = get_file_info_from_path(child)
            files.append(file)
        except Exception as e:  # noqa: BLE001
            logging.warning(f"文件读取失败:{e}")  # noqa: LOG015
    return files

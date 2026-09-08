import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from interview_ai.ingestion.models import MarkdownFile
from interview_ai.ingestion.scanner import split_file


def make_file(content: str) -> MarkdownFile:
    now = datetime.now(UTC)
    return MarkdownFile(
        path=Path("knowledge.md"),
        name="knowledge.md",
        create_time=now,
        modify_time=now,
        size=len(content.encode("utf-8")),
        content=content,
        hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def test_split_file_uses_headings_and_keeps_source_lines():
    file = make_file(
        "# Python\n\n基础。\n\n## 模块\n\n模块内容。\n\n### 导入\n\n导入内容。"
    )

    chunks = split_file(file, max_chunk_length=100)

    assert [chunk.headings for chunk in chunks] == [
        ("Python",),
        ("Python", "模块"),
        ("Python", "模块", "导入"),
    ]
    assert [chunk.level for chunk in chunks] == [1, 2, 3]
    assert (chunks[1].start_line, chunks[1].end_line) == (5, 7)
    assert all(chunk.document_id == file.id for chunk in chunks)
    assert [chunk.sort_index for chunk in chunks] == [0, 1, 2]


def test_chunk_from_split_file_should_not_change():
    file = make_file(
        "# Python\n\n基础。\n\n## 模块\n\n模块内容。\n\n### 导入\n\n导入内容。"
    )

    chunks = split_file(file, max_chunk_length=100)
    chunks1 = split_file(file, max_chunk_length=100)

    for i in range(len(chunks)):
        assert chunks[i].id == chunks1[i].id


def test_split_file_respects_maximum_length():
    chunks = split_file(make_file("# 标题\n\n" + "a" * 25), max_chunk_length=10)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 10 for chunk in chunks)
    assert all(len(chunk.hash) == 64 for chunk in chunks)


def test_heading_inside_code_fence_does_not_start_section():
    file = make_file("# Python\n\n```python\n# 不是标题\nprint('ok')\n```")

    chunks = split_file(file, max_chunk_length=100)

    assert len(chunks) == 1
    assert chunks[0].headings == ("Python",)
    assert "# 不是标题" in chunks[0].content


def test_split_file_handles_empty_content_and_invalid_length():
    assert split_file(make_file("")) == []

    with pytest.raises(ValueError, match="max_chunk_length"):
        split_file(make_file("content"), max_chunk_length=0)

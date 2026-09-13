import json
import re
from hashlib import sha256
from pathlib import Path

from interview_ai.db.models import Document
from interview_ai.ingestion.scanner import split_document

DATASET_ROOT = Path(__file__).parents[2] / "evals" / "retrieval"
QUESTIONS_PATH = DATASET_ROOT / "questions.jsonl"
CORPUS_ROOT = DATASET_ROOT / "corpus"
HEADING_PATTERN = re.compile(
    r"^(?P<marks>#{1,6})\s+(?P<title>.+?)(?:\s+\{\s*#[^}]+\})?\s*$"
)


def _load_questions() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _section_content(document: str, expected_heading: str) -> str:
    lines = document.splitlines()
    headings: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        match = HEADING_PATTERN.match(line)
        if match:
            headings.append((len(match.group("marks")), match.group("title"), index))

    matches = [item for item in headings if item[1] == expected_heading]
    assert len(matches) == 1, f"标题不存在或不唯一：{expected_heading}"
    level, _, start = matches[0]
    end = len(lines)
    for next_level, _, next_start in headings:
        if next_start > start and next_level <= level:
            end = next_start
            break
    return "\n".join(lines[start:end])


def _visible_heading(heading: str) -> str:
    match = HEADING_PATTERN.match(f"# {heading}")
    assert match is not None
    return match.group("title")


def test_retrieval_dataset_has_at_least_thirty_unique_questions() -> None:
    questions = _load_questions()
    ids = [question["id"] for question in questions]

    assert len(questions) >= 30
    assert len(ids) == len(set(ids))
    assert all(str(question["query"]).strip() for question in questions)


def test_every_relevance_label_points_to_existing_corpus_content() -> None:
    for question in _load_questions():
        sources = question["relevant_sources"]
        assert isinstance(sources, list) and sources, question["id"]

        for source in sources:
            assert isinstance(source, dict)
            corpus_path = CORPUS_ROOT / str(source["document"])
            assert corpus_path.is_file(), corpus_path

            document = corpus_path.read_text(encoding="utf-8")
            section = _section_content(document, str(source["heading"]))
            assert str(source["contains"]) in section, (
                f"{question['id']} 的锚点不在标注章节中：{source['contains']}"
            )


def test_every_relevance_label_matches_a_chunk_from_current_chunker() -> None:
    chunks_by_document = {}
    for corpus_path in CORPUS_ROOT.glob("*.md"):
        content = corpus_path.read_text(encoding="utf-8")
        content_bytes = content.encode("utf-8")
        document = Document(
            owner_id="evaluation-user",
            name=corpus_path.name,
            content=content,
            content_hash=sha256(content_bytes).hexdigest(),
            size_bytes=len(content_bytes),
        )
        chunks_by_document[corpus_path.name] = split_document(document)

    for question in _load_questions():
        for source in question["relevant_sources"]:
            matching_chunks = [
                chunk
                for chunk in chunks_by_document[str(source["document"])]
                if str(source["heading"])
                in {_visible_heading(heading) for heading in chunk.headings}
                and str(source["contains"]) in chunk.content
            ]
            assert matching_chunks, (
                f"{question['id']} 的标注无法匹配当前切块结果：{source}"
            )

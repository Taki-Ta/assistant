import hashlib
import json
from collections import Counter
from pathlib import Path

from interview_ai.db.models import Document
from interview_ai.evaluation.dataset import load_retrieval_cases
from interview_ai.ingestion.scanner import split_document

DATASET_ROOT = Path(__file__).parents[2] / "evals" / "python_tutorial"
CORPUS_ROOT = DATASET_ROOT / "corpus"


def _visible_heading(heading: str) -> str:
    return heading.strip().rstrip("#").rstrip()


def test_python_tutorial_corpus_matches_source_manifest() -> None:
    manifest = json.loads(
        (DATASET_ROOT / "source_manifest.json").read_text(encoding="utf-8")
    )
    documents = manifest["documents"]

    assert manifest["upstream_revision"] == ("abfd0ab613b2f3a4b5003a69acda53ea67b69b03")
    assert len(documents) == 29
    assert {path.name for path in CORPUS_ROOT.glob("*.md")} == {
        document["document"] for document in documents
    }
    for document in documents:
        content = (CORPUS_ROOT / document["document"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == document["sha256"]


def test_python_tutorial_questions_cover_every_chapter() -> None:
    cases = load_retrieval_cases(DATASET_ROOT / "questions.jsonl")
    chapter_counts = Counter(case.tags[0] for case in cases)

    assert len(cases) == 58
    assert chapter_counts == Counter(
        {f"chapter-{chapter:02d}": 2 for chapter in range(29)}
    )


def test_python_tutorial_labels_match_current_chunks() -> None:
    chunks_by_document = {}
    for corpus_path in CORPUS_ROOT.glob("*.md"):
        content = corpus_path.read_text(encoding="utf-8")
        content_bytes = content.encode("utf-8")
        document = Document(
            owner_id="evaluation-user",
            name=corpus_path.name,
            content=content,
            content_hash=hashlib.sha256(content_bytes).hexdigest(),
            size_bytes=len(content_bytes),
        )
        chunks_by_document[corpus_path.name] = split_document(document)

    for case in load_retrieval_cases(DATASET_ROOT / "questions.jsonl"):
        for source in case.relevant_sources:
            matches = [
                chunk
                for chunk in chunks_by_document[source.document]
                if source.heading
                in {_visible_heading(heading) for heading in chunk.headings}
                and source.contains in chunk.content
            ]
            assert matches, f"{case.id} 的标注无法匹配当前切块结果：{source}"

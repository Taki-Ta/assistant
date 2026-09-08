import json
from dataclasses import asdict

import pytest
from interview_ai.ingestion.manifest import (
    generate_manifest,
    generate_manifest_dict,
    load_manifest,
    save_manifest,
)
from interview_ai.ingestion.models import Manifest
from interview_ai.ingestion.scanner import split_file

from .util import make_file


def test_save_manifest_should_work(tmp_path):
    path = tmp_path / "python" / "knowledge.md"
    document = make_file("# Python\n\n基础内容。", path)
    chunks = split_file(document, max_chunk_length=100)

    manifest = generate_manifest(
        tmp_path,
        [document],
        chunks,
        max_chunk_length=100,
    )
    file_path = save_manifest(manifest, tmp_path / "manifests")

    assert file_path.exists()
    assert file_path.suffix == ".json"
    assert ":" not in file_path.name


def test_manifest_json_round_trip(tmp_path):
    document = make_file("# Python\n\n基础内容。", tmp_path / "knowledge.md")
    original = generate_manifest(
        tmp_path,
        [document],
        split_file(document, max_chunk_length=100),
        max_chunk_length=100,
    )

    restored = Manifest.from_json(original.to_json())

    assert restored.schema_version == original.schema_version
    assert restored.generated_at == original.generated_at
    assert restored.source_root == original.source_root
    assert restored.pipeline == original.pipeline
    assert restored.documents == original.documents


def test_save_then_load_manifest(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    original = generate_manifest(tmp_path, [document], split_file(document))
    file_path = save_manifest(original, tmp_path / "manifests")

    restored = load_manifest(file_path)

    assert restored.to_dic() == original.to_dic()


def test_manifest_rejects_unknown_schema_version(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    data = generate_manifest(tmp_path, [document], split_file(document)).to_dic()
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="schema_version"):
        Manifest.from_dict(data)


def test_generate_manifest_contains_document_and_chunk_metadata(tmp_path):
    path = tmp_path / "python" / "knowledge.md"
    document = make_file("# Python\n\n基础内容。", path)
    chunks = split_file(document, max_chunk_length=100)

    manifest = generate_manifest(
        tmp_path,
        [document],
        chunks,
        max_chunk_length=100,
    )

    assert manifest.schema_version == 1
    assert manifest.source_root == tmp_path.resolve()
    assert manifest.generated_at.tzinfo is not None
    assert manifest.pipeline.chunker_version == 1
    assert manifest.pipeline.max_chunk_length == 100

    entry = manifest.documents["python/knowledge.md"]
    assert entry["document_id"] == str(document.id)
    assert entry["content_hash"] == document.hash
    assert entry["size_bytes"] == document.size
    assert entry["chunks"] == [
        {
            "chunk_id": str(chunks[0].id),
            "index": 0,
            "content_hash": chunks[0].hash,
            "headings": ["Python"],
            "start_line": 1,
            "end_line": 3,
        }
    ]


def test_manifest_documents_are_json_serializable(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    manifest = generate_manifest(tmp_path, [document], split_file(document))

    serialized = json.dumps(manifest.documents, ensure_ascii=False)

    assert "knowledge.md" in serialized
    assert str(document.id) in serialized


def test_generate_manifest_dict_returns_empty_mapping_for_no_documents():
    assert generate_manifest_dict([], []) == {}


def test_manifest_keeps_same_named_files_from_different_directories(tmp_path):
    first = make_file("# Python", tmp_path / "python" / "knowledge.md")
    second = make_file("# Java", tmp_path / "java" / "knowledge.md")
    chunks = [*split_file(first), *split_file(second)]

    documents = generate_manifest_dict([first, second], chunks, root=tmp_path)

    assert set(documents) == {"python/knowledge.md", "java/knowledge.md"}
    assert (
        documents["python/knowledge.md"]["document_id"]
        != documents["java/knowledge.md"]["document_id"]
    )


def test_manifest_only_attaches_chunks_to_their_own_document(tmp_path):
    first = make_file("# Python", tmp_path / "python.md")
    second = make_file("# Java", tmp_path / "java.md")
    first_chunks = split_file(first)
    second_chunks = split_file(second)

    manifest = generate_manifest(
        tmp_path,
        [first, second],
        [*second_chunks, *first_chunks],
    )

    assert [
        chunk["chunk_id"] for chunk in manifest.documents["python.md"]["chunks"]
    ] == [str(first_chunks[0].id)]
    assert [chunk["chunk_id"] for chunk in manifest.documents["java.md"]["chunks"]] == [
        str(second_chunks[0].id)
    ]


def test_manifest_rejects_document_outside_source_root(tmp_path):
    outside = make_file("# Outside", tmp_path.parent / "outside.md")

    with pytest.raises(ValueError, match="不在知识库根目录"):
        generate_manifest(tmp_path, [outside], split_file(outside))


def test_manifest_dataclass_keeps_runtime_types(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    manifest = generate_manifest(tmp_path, [document], split_file(document))

    data = asdict(manifest)
    assert data["source_root"] == tmp_path.resolve()
    assert data["pipeline"]["max_chunk_length"] == 500

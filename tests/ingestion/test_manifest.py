import json
from dataclasses import asdict

import pytest

from interview_ai.ingestion.manifest import (
    compare_manifests,
    generate_manifest,
    generate_manifest_dict,
    load_manifest,
    save_manifest,
)
from interview_ai.ingestion.models import Manifest
from interview_ai.ingestion.scanner import split_document

from .util import make_file


def test_save_manifest_should_work(tmp_path):
    path = tmp_path / "python" / "knowledge.md"
    document = make_file("# Python\n\n基础内容。", path)
    chunks = split_document(document, max_chunk_length=100)

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
        split_document(document, max_chunk_length=100),
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
    original = generate_manifest(tmp_path, [document], split_document(document))
    file_path = save_manifest(original, tmp_path / "manifests")

    restored = load_manifest(file_path)

    assert restored.to_dic() == original.to_dic()


def test_manifest_rejects_unknown_schema_version(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    data = generate_manifest(tmp_path, [document], split_document(document)).to_dic()
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="schema_version"):
        Manifest.from_dict(data)


def test_compare_manifests_detects_all_document_states(tmp_path):
    original_a = make_file("# A", tmp_path / "a.md")
    original_b = make_file("# B", tmp_path / "b.md")
    deleted = make_file("# Deleted", tmp_path / "deleted.md")
    old_documents = [original_a, original_b, deleted]
    old_chunks = [
        chunk for document in old_documents for chunk in split_document(document)
    ]
    old_manifest = generate_manifest(tmp_path, old_documents, old_chunks)

    unchanged = make_file("# A", tmp_path / "a.md")
    modified = make_file("# B changed", tmp_path / "b.md")
    added = make_file("# Added", tmp_path / "added.md")
    new_documents = [unchanged, modified, added]
    new_chunks = [
        chunk for document in new_documents for chunk in split_document(document)
    ]
    new_manifest = generate_manifest(tmp_path, new_documents, new_chunks)

    changes = compare_manifests(old_manifest, new_manifest)

    assert changes.added == ("added.md",)
    assert changes.modified == ("b.md",)
    assert changes.unchanged == ("a.md",)
    assert changes.deleted == ("deleted.md",)
    assert changes.pipeline_changed is False


def test_compare_manifests_without_old_manifest_marks_all_documents_added(tmp_path):
    second = make_file("# B", tmp_path / "b.md")
    first = make_file("# A", tmp_path / "a.md")
    documents = [second, first]
    chunks = [chunk for document in documents for chunk in split_document(document)]
    manifest = generate_manifest(tmp_path, documents, chunks)

    changes = compare_manifests(None, manifest)

    assert changes.added == ("a.md", "b.md")
    assert changes.modified == ()
    assert changes.unchanged == ()
    assert changes.deleted == ()
    assert changes.pipeline_changed is False


def test_compare_manifests_rejects_different_source_roots(tmp_path):
    old_manifest = generate_manifest(tmp_path / "old", [], [])
    new_manifest = generate_manifest(tmp_path / "new", [], [])

    with pytest.raises(ValueError, match="不同知识库根目录"):
        compare_manifests(old_manifest, new_manifest)


def test_compare_manifests_ignores_document_metadata_changes(tmp_path):
    path = tmp_path / "knowledge.md"
    old_document = make_file("# Python", path)
    new_document = make_file("# Python", path)
    new_document.size += 10
    old_manifest = generate_manifest(
        tmp_path, [old_document], split_document(old_document)
    )
    new_manifest = generate_manifest(
        tmp_path, [new_document], split_document(new_document)
    )

    changes = compare_manifests(old_manifest, new_manifest)

    assert changes.modified == ()
    assert changes.unchanged == ("knowledge.md",)


def test_compare_manifests_reindexes_common_documents_when_pipeline_changes(tmp_path):
    document = make_file("# Python", tmp_path / "python.md")
    chunks = split_document(document)
    old_manifest = generate_manifest(tmp_path, [document], chunks, max_chunk_length=500)
    new_manifest = generate_manifest(tmp_path, [document], chunks, max_chunk_length=800)

    changes = compare_manifests(old_manifest, new_manifest)

    assert changes.modified == ("python.md",)
    assert changes.unchanged == ()
    assert changes.pipeline_changed is True


def test_generate_manifest_contains_document_and_chunk_metadata(tmp_path):
    path = tmp_path / "python" / "knowledge.md"
    document = make_file("# Python\n\n基础内容。", path)
    chunks = split_document(document, max_chunk_length=100)

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
    manifest = generate_manifest(tmp_path, [document], split_document(document))

    serialized = json.dumps(manifest.documents, ensure_ascii=False)

    assert "knowledge.md" in serialized
    assert str(document.id) in serialized


def test_generate_manifest_dict_returns_empty_mapping_for_no_documents():
    assert generate_manifest_dict([], []) == {}


def test_generate_manifest_rejects_duplicate_document_paths(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")

    with pytest.raises(ValueError, match="重复文档路径"):
        generate_manifest(tmp_path, [document, document], split_document(document))


def test_manifest_keeps_same_named_files_from_different_directories(tmp_path):
    first = make_file("# Python", tmp_path / "python" / "knowledge.md")
    second = make_file("# Java", tmp_path / "java" / "knowledge.md")
    chunks = [*split_document(first), *split_document(second)]

    documents = generate_manifest_dict([first, second], chunks, root=tmp_path)

    assert set(documents) == {"python/knowledge.md", "java/knowledge.md"}
    assert (
        documents["python/knowledge.md"]["document_id"]
        != documents["java/knowledge.md"]["document_id"]
    )


def test_manifest_only_attaches_chunks_to_their_own_document(tmp_path):
    first = make_file("# Python", tmp_path / "python.md")
    second = make_file("# Java", tmp_path / "java.md")
    first_chunks = split_document(first)
    second_chunks = split_document(second)

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


def test_manifest_sorts_chunks_by_index(tmp_path):
    document = make_file("# Python\n\nabcdefghij\n\nklmnopqrst", tmp_path / "python.md")
    chunks = split_document(document, max_chunk_length=10)

    manifest = generate_manifest(tmp_path, [document], list(reversed(chunks)))

    stored_chunks = manifest.documents["python.md"]["chunks"]
    assert [chunk["index"] for chunk in stored_chunks] == list(range(len(chunks)))


def test_manifest_rejects_document_outside_source_root(tmp_path):
    outside = make_file("# Outside", tmp_path.parent / "outside.md")

    with pytest.raises(ValueError, match="不在知识库根目录"):
        generate_manifest(tmp_path, [outside], split_document(outside))


def test_manifest_dataclass_keeps_runtime_types(tmp_path):
    document = make_file("# Python", tmp_path / "knowledge.md")
    manifest = generate_manifest(tmp_path, [document], split_document(document))

    data = asdict(manifest)
    assert data["source_root"] == tmp_path.resolve()
    assert data["pipeline"]["max_chunk_length"] == 500


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    [
        ("pipeline", [], "Manifest.pipeline 必须是对象"),
        ("documents", [], "Manifest.documents 必须是对象"),
    ],
)
def test_manifest_rejects_invalid_object_fields(
    tmp_path, field, invalid_value, message
):
    data = generate_manifest(tmp_path, [], []).to_dic()
    data[field] = invalid_value

    with pytest.raises(TypeError, match=message):
        Manifest.from_dict(data)

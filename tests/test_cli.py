import json

from typer.testing import CliRunner

from interview_ai.cli import app
from interview_ai.ingestion.manifest import generate_manifest, save_manifest
from interview_ai.ingestion.scanner import scan_path, split_file

runner = CliRunner()


def test_plan_json_marks_all_documents_added_without_manifest(tmp_path):
    (tmp_path / "python.md").write_text("# Python\n\n基础。", encoding="utf-8")

    result = runner.invoke(app, ["plan", str(tmp_path), "--json"])

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["document_count"] == 1
    assert output["chunk_count"] == 1
    assert output["changes"]["added"] == ["python.md"]
    assert output["changes"]["modified"] == []
    assert output["changes"]["deleted"] == []
    assert output["changes"]["unchanged"] == []


def test_plan_compares_with_existing_manifest(tmp_path):
    root = tmp_path / "knowledge"
    root.mkdir()
    (root / "python.md").write_text("# Python", encoding="utf-8")
    documents = scan_path(root)
    chunks = [chunk for document in documents for chunk in split_file(document)]
    manifest = generate_manifest(root, documents, chunks)
    manifest_path = save_manifest(manifest, tmp_path / "manifests")

    result = runner.invoke(
        app,
        ["plan", str(root), "--manifest", str(manifest_path), "--json"],
    )

    assert result.exit_code == 0
    output = json.loads(result.stdout)
    assert output["changes"]["added"] == []
    assert output["changes"]["modified"] == []
    assert output["changes"]["deleted"] == []
    assert output["changes"]["unchanged"] == ["python.md"]


def test_plan_reports_missing_explicit_manifest(tmp_path):
    result = runner.invoke(
        app,
        ["plan", str(tmp_path), "--manifest", str(tmp_path / "missing.json")],
    )

    assert result.exit_code == 1
    assert "Manifest 文件不存在" in result.stderr

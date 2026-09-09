import logging
from datetime import datetime
from pathlib import Path

from .models import (
    MAX_CHUNK_LENGTH,
    TIME_ZONE_LOCAL,
    ChangeSet,
    Chunk,
    Manifest,
    ManifestPipeline,
    MarkdownFile,
)

MANIFEST_DIR = "manifest"
logger = logging.getLogger(__name__)


def compare_manifests(
    old_manifest: Manifest | None,
    new_manifest: Manifest,
) -> ChangeSet:
    """比较已索引快照与当前候选快照，生成只读变更计划。"""
    if old_manifest is None:
        return ChangeSet(
            added=tuple(sorted(new_manifest.documents)),
            modified=(),
            unchanged=(),
            deleted=(),
        )

    if old_manifest.source_root.resolve() != new_manifest.source_root.resolve():
        raise ValueError("不能比较不同知识库根目录的 Manifest")

    old_paths = set(old_manifest.documents)
    new_paths = set(new_manifest.documents)
    added = new_paths - old_paths
    deleted = old_paths - new_paths
    common = old_paths & new_paths
    pipeline_changed = old_manifest.pipeline != new_manifest.pipeline
    modified: set[str] = set()
    unchanged: set[str] = set()

    for path in common:
        old_document = old_manifest.documents[path]
        new_document = new_manifest.documents[path]
        content_changed = old_document["content_hash"] != new_document["content_hash"]
        if content_changed or pipeline_changed:
            modified.add(path)
        else:
            unchanged.add(path)

    return ChangeSet(
        added=tuple(sorted(added)),
        modified=tuple(sorted(modified)),
        unchanged=tuple(sorted(unchanged)),
        deleted=tuple(sorted(deleted)),
        pipeline_changed=pipeline_changed,
    )


def generate_manifest(
    root: Path,
    documents: list[MarkdownFile],
    chunks: list[Chunk],
    *,
    max_chunk_length: int = MAX_CHUNK_LENGTH,
) -> Manifest:
    """生成上一次成功索引状态的 Manifest。"""
    return Manifest(
        source_root=root.resolve(),
        documents=generate_manifest_dict(documents, chunks, root=root),
        pipeline=ManifestPipeline(max_chunk_length=max_chunk_length),
    )


def generate_manifest_dict(
    documents: list[MarkdownFile],
    chunks: list[Chunk],
    *,
    root: Path | None = None,
) -> dict[str, dict[str, object]]:
    """生成可直接进行 JSON 序列化的文档索引字典。"""
    manifest_documents: dict[str, dict[str, object]] = {}
    for document in documents:
        key = _document_key(document.path, root)
        if key in manifest_documents:
            raise ValueError(f"Manifest 中存在重复文档路径：{key}")

        modified_at = document.modify_time
        if modified_at.tzinfo is None:
            modified_at = modified_at.replace(tzinfo=TIME_ZONE_LOCAL)

        document_chunks = sorted(
            (chunk for chunk in chunks if chunk.document_id == document.id),
            key=lambda chunk: chunk.sort_index,
        )
        manifest_documents[key] = {
            "document_id": str(document.id),
            "content_hash": document.hash,
            "size_bytes": document.size,
            "modified_at": modified_at.astimezone(TIME_ZONE_LOCAL).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "chunks": [
                {
                    "chunk_id": str(chunk.id),
                    "index": chunk.sort_index,
                    "content_hash": chunk.hash,
                    "headings": list(chunk.headings),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                }
                for chunk in document_chunks
            ],
        }

    return manifest_documents


def save_manifest(manifest: Manifest, directory: Path | None = None) -> Path:
    """保存 Manifest 并返回生成的文件路径。"""
    output_directory = directory or Path.cwd() / MANIFEST_DIR
    output_directory.mkdir(parents=True, exist_ok=True)
    file_path = output_directory / _generate_manifest_file_name()
    file_path.write_text(manifest.to_json(), encoding="utf-8")
    logger.info("Manifest 已保存：%s", file_path)
    return file_path


def load_manifest(file_path: Path) -> Manifest:
    """从 UTF-8 JSON 文件恢复 Manifest。"""
    return Manifest.from_json(file_path.read_text(encoding="utf-8"))


def _generate_manifest_file_name() -> str:
    now = datetime.now(TIME_ZONE_LOCAL)
    return f"manifest-{now.strftime('%Y-%m-%d__%H-%M-%S-%f')}.json"


def _document_key(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"文档不在知识库根目录中：{path}") from exc

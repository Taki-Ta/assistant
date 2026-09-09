import json
from pathlib import Path
from typing import Annotated

import typer

from .ingestion.manifest import (
    compare_manifests,
    generate_manifest,
    load_manifest,
)
from .ingestion.models import MAX_CHUNK_LENGTH, ChangeSet
from .ingestion.scanner import scan_path, split_file

app = typer.Typer()


# 如果不写以下代码段,typer会简化为单命令模式,调用时不再需要function name
@app.callback()
def main():
    """面经 AI 助手。"""


@app.command()
def scan(path: str):
    """扫描目录下的所有md文件"""
    try:
        result = scan_path(Path(path))
        typer.echo("name\tsize\tcreate_time\tmodify_time\tcontent")
        for item in result:
            typer.echo(
                f"{item.name}\t{item.size:,}\t{item.create_time.strftime('%Y-%m-%d %H:%M:%S')}\t{item.modify_time.strftime('%Y-%m-%d %H:%M:%S')}\t{item.content[:20]!r}"
            )

        typer.echo(f"total={len(result)}")
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except Exception as e:
        typer.echo(e)
        raise typer.Exit(code=1) from e


@app.command()
def plan(
    path: Path,
    manifest_path: Annotated[
        Path | None,
        typer.Option("--manifest", help="上一次成功索引的 Manifest 文件"),
    ] = None,
    max_chunk_length: Annotated[
        int,
        typer.Option("--max-chunk-length", min=1, help="单个 Chunk 最大字符数"),
    ] = MAX_CHUNK_LENGTH,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="以 JSON 格式输出"),
    ] = False,
    show_unchanged: Annotated[
        bool,
        typer.Option("--show-unchanged", help="在文本结果中列出未变化文件"),
    ] = False,
):
    """预览文档索引变化，不保存 Manifest，也不修改向量索引。"""
    try:
        if manifest_path is not None and not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest 文件不存在：{manifest_path}")

        documents = scan_path(path)
        chunks = [
            chunk
            for document in documents
            for chunk in split_file(document, max_chunk_length=max_chunk_length)
        ]
        current_manifest = generate_manifest(
            path,
            documents,
            chunks,
            max_chunk_length=max_chunk_length,
        )
        old_manifest = load_manifest(manifest_path) if manifest_path else None
        changes = compare_manifests(old_manifest, current_manifest)

        if json_output:
            _print_plan_json(path, documents, chunks, changes)
        else:
            _print_plan_text(path, documents, chunks, changes, show_unchanged)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        typer.echo(f"生成变更计划失败：{exc}", err=True)
        raise typer.Exit(code=1) from exc


def _print_plan_json(path, documents, chunks, changes: ChangeSet) -> None:
    result = {
        "source_root": str(path.resolve()),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "changes": {
            "added": list(changes.added),
            "modified": list(changes.modified),
            "deleted": list(changes.deleted),
            "unchanged": list(changes.unchanged),
            "pipeline_changed": changes.pipeline_changed,
        },
    }
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


def _print_plan_text(
    path,
    documents,
    chunks,
    changes: ChangeSet,
    show_unchanged: bool,
) -> None:
    typer.echo(f"知识库：{path.resolve()}")
    typer.echo(f"扫描文档：{len(documents)}")
    typer.echo(f"生成片段：{len(chunks)}")
    typer.echo("\n变更计划：")
    typer.echo(f"  新增：{len(changes.added)}")
    typer.echo(f"  修改：{len(changes.modified)}")
    typer.echo(f"  删除：{len(changes.deleted)}")
    typer.echo(f"  未变化：{len(changes.unchanged)}")
    typer.echo(f"  切块配置变化：{'是' if changes.pipeline_changed else '否'}")
    _print_paths("新增", "+", changes.added)
    _print_paths("修改", "~", changes.modified)
    _print_paths("删除", "-", changes.deleted)
    if show_unchanged:
        _print_paths("未变化", "=", changes.unchanged)


def _print_paths(title: str, marker: str, paths: tuple[str, ...]) -> None:
    if not paths:
        return
    typer.echo(f"\n{title}：")
    for path in paths:
        typer.echo(f"  {marker} {path}")


if __name__ == "__main__":
    app()

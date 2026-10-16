import hashlib
import logging
from datetime import datetime
from pathlib import Path

from .models import TIME_ZONE_LOCAL, LocalDocument

IGNORE_DIRS = [
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "assistant",
]


def scan_path(path: Path) -> list[LocalDocument]:
    """接收一个路径,返回路径下所有md格式的文件"""
    if not Path.exists(path):
        raise FileNotFoundError("文件不存在")
    if Path.is_file(path):
        if path.suffix == ".md":
            return [get_file_info_from_path(path)]
        else:
            return []
    return _scan_file(path, [])


def get_file_info_from_path(path: Path) -> LocalDocument:
    """根据文件路径获取文件信息"""
    if not Path.exists(path) or not Path.is_file(path):
        raise FileNotFoundError("文件不存在")
    with open(path, "r", encoding="utf-8") as f:
        info = path.stat()
        content = f.read()
        content_bytes = content.encode("utf-8")
        return LocalDocument(
            path=path,
            name=path.name,
            create_time=datetime.fromtimestamp(info.st_birthtime, TIME_ZONE_LOCAL),
            modify_time=datetime.fromtimestamp(info.st_mtime, TIME_ZONE_LOCAL),
            size=info.st_size,
            content=content,
            hash=hashlib.sha256(content_bytes).hexdigest(),
        )


def _scan_file(
    path: Path, files: list[LocalDocument] | None = None
) -> list[LocalDocument]:
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

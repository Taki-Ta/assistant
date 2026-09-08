import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import MarkdownFile

TIME_ZONE_LOCAL = timezone(timedelta(hours=8))
IGNORE_DIRS = [".git", ".venv", "assistant"]


def scan_path(path: Path) -> list[MarkdownFile]:
    """接收一个路径,返回路径下所有md格式的文件"""
    if not Path.exists(path):
        raise FileNotFoundError("文件不存在")
    if Path.is_file(path):
        if path.suffix == ".md":
            return [get_file_info_from_path(path)]
        else:
            return []
    return scan_file(path, [])


def scan_file(
    path: Path, files: list[MarkdownFile] | None = None
) -> list[MarkdownFile]:
    """接收一个文件夹路径,返回路径下所有md格式的文件"""
    if files is None:
        files = []
    for child in path.rglob("*.md"):
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


def get_file_info_from_path(path: Path) -> MarkdownFile:
    """根据文件路径获取文件信息"""
    if not Path.exists(path) or not Path.is_file(path):
        raise FileNotFoundError("文件不存在")
    with open(path, "r", encoding="utf-8") as f:
        info = path.stat()
        return MarkdownFile(
            path,
            path.name,
            datetime.fromtimestamp(info.st_birthtime, TIME_ZONE_LOCAL),
            datetime.fromtimestamp(info.st_mtime, TIME_ZONE_LOCAL),
            info.st_size,
            f.read(),
        )

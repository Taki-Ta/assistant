import string

from interview_ai.ingestion.scanner import (
    IGNORE_DIRS,
    get_file_info_from_path,
    scan_path,
)


def test_get_file_info_from_path_should_work(tmp_path):
    file = tmp_path / "test.md"
    content = string.ascii_letters
    file.write_text(content, encoding="utf-8")

    info = get_file_info_from_path(file)
    print(f"{info.size=}")
    assert info.size == len(content)
    assert info.content == content


def test_scan_path_should_work(tmp_path):
    file = tmp_path / "test01.txt"
    content = string.ascii_letters
    file.write_text(content, encoding="utf-8")

    files = scan_path(tmp_path)
    assert len(files) == 0

    file = tmp_path / "test01.md"
    content = string.ascii_letters
    file.write_text(content, encoding="utf-8")

    files = scan_path(tmp_path)
    assert len(files) == 1

    file_path = tmp_path / "test"
    file_path.mkdir()
    file = file_path / "test02.md"
    content = string.ascii_letters
    file.write_text(content, encoding="utf-8")

    files = scan_path(tmp_path)
    assert len(files) == 2

    ignored_dir = tmp_path / IGNORE_DIRS[0]
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "test03.md"
    content = string.ascii_letters
    ignored_file.write_text(content, encoding="utf-8")

    files = scan_path(tmp_path)
    assert len(files) == 2

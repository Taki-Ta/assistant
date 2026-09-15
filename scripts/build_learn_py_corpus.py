from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

UPSTREAM_REPOSITORY = "https://github.com/walter201230/Python"
UPSTREAM_REVISION = "abfd0ab613b2f3a4b5003a69acda53ea67b69b03"
SOURCE_ROOT = Path("Article/PythonBasis")


@dataclass(frozen=True)
class SourcePart:
    path: str
    title: str


@dataclass(frozen=True)
class Chapter:
    number: int
    slug: str
    title: str
    parts: tuple[SourcePart, ...]

    @property
    def filename(self) -> str:
        return f"{self.number:02d}-{self.slug}.md"


def part(path: str, title: str) -> SourcePart:
    return SourcePart(path=path, title=title)


CHAPTERS = (
    Chapter(
        0,
        "为什么学-python",
        "为什么学 Python",
        (part("python0/WhyStudyPython.md", "为什么学 Python"),),
    ),
    Chapter(
        1,
        "安装与第一个程序",
        "Python 安装与第一个程序",
        (
            part("python1/Introduction.md", "Python 简介"),
            part("python1/Installation.md", "Python 的安装"),
            part("python1/The_first_procedure.md", "第一个 Python 程序"),
            part("python1/IDE.md", "集成开发环境（IDE）：PyCharm"),
        ),
    ),
    Chapter(
        2,
        "基本数据类型与变量",
        "基本数据类型与变量",
        (
            part("python2/Grammar.md", "Python 语法的简要说明"),
            part("python2/print.md", "print() 函数"),
            part("python2/Type_of_data.md", "Python 的基本数据类型"),
            part("python2/StringCoding.md", "字符串的编码问题"),
            part("python2/Type_conversion.md", "基本数据类型转换"),
            part("python2/Variable.md", "Python 中的变量"),
        ),
    ),
    Chapter(
        3,
        "list-与-tuple",
        "List 与 Tuple",
        (
            part("python3/List.md", "List（列表）"),
            part("python3/tuple.md", "tuple（元组）"),
        ),
    ),
    Chapter(
        4,
        "dict-与-set",
        "Dict 与 Set",
        (
            part("python4/Dict.md", "字典（Dictionary）"),
            part("python4/Set.md", "Set"),
        ),
    ),
    Chapter(
        5,
        "条件与循环",
        "条件语句与循环语句",
        (
            part("python5/If.md", "条件语句"),
            part("python5/Cycle.md", "循环语句"),
            part("python5/Example.md", "条件语句与循环语句综合实例"),
        ),
    ),
    Chapter(
        6,
        "函数",
        "函数",
        tuple(
            part(f"python6/{name}.md", title)
            for name, title in (
                ("1", "自定义函数的基本步骤"),
                ("2", "函数返回值"),
                ("3", "函数的参数"),
                ("4", "函数传值问题"),
                ("5", "匿名函数"),
            )
        ),
    ),
    Chapter(
        7,
        "迭代器与生成器",
        "迭代器与生成器",
        tuple(
            part(f"python7/{name}.md", title)
            for name, title in (
                ("1", "迭代"),
                ("2", "Python 迭代器"),
                ("3", "列表生成式"),
                ("4", "生成器"),
                ("5", "迭代器和生成器综合例子"),
            )
        ),
    ),
    Chapter(
        8,
        "面向对象",
        "面向对象",
        tuple(
            part(f"python8/{number}.md", title)
            for number, title in enumerate(
                (
                    "面向对象的概念",
                    "类的定义和调用",
                    "类方法",
                    "修改和增加类属性",
                    "类和对象",
                    "初始化函数",
                    "类的继承",
                    "类的多态",
                    "类的访问控制",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        9,
        "模块与包",
        "模块与包",
        tuple(
            part(f"python9/{number}.md", title)
            for number, title in enumerate(
                (
                    "Python 模块简介",
                    "模块的使用",
                    "主模块和非主模块",
                    "包",
                    "作用域",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        10,
        "魔法方法",
        "Python 的 Magic Method",
        tuple(
            part(f"python10/{number}.md", title)
            for number, title in enumerate(
                (
                    "Python 的 Magic Method",
                    "构造（__new__）和初始化（__init__）",
                    "属性的访问控制",
                    "对象的描述器",
                    "自定义容器（Container）",
                    "运算符相关的魔术方法",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        11,
        "枚举类",
        "枚举类",
        tuple(
            part(f"python11/{number}.md", title)
            for number, title in enumerate(
                (
                    "枚举类的使用",
                    "Enum 的源码",
                    "自定义类型的枚举",
                    "枚举的比较",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        12,
        "元类",
        "元类",
        tuple(
            part(f"python12/{number}.md", title)
            for number, title in enumerate(
                (
                    "Python 中类也是对象",
                    "使用 type() 动态创建类",
                    "什么是元类",
                    "自定义元类",
                    "使用元类",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        13,
        "线程与进程",
        "线程与进程",
        tuple(
            part(f"python13/{number}.md", title)
            for number, title in enumerate(
                (
                    "线程与进程",
                    "多线程编程",
                    "进程",
                ),
                start=1,
            )
        ),
    ),
    Chapter(
        14,
        "正则表达式",
        "正则表达式",
        tuple(
            part(f"python14/{number}.md", title)
            for number, title in enumerate(
                (
                    "初识 Python 正则表达式",
                    "字符集",
                    "数量词",
                    "边界匹配符和组",
                    "re.sub",
                    "re.match 和 re.search",
                ),
                start=1,
            )
        ),
    ),
    Chapter(15, "闭包", "闭包", (part("python15/1.md", "闭包"),)),
    Chapter(16, "装饰器", "装饰器", (part("python16/1.md", "装饰器"),)),
    Chapter(17, "类型注解", "类型注解", (part("python17/1.md", "类型注解"),)),
    Chapter(
        18,
        "pathlib-路径处理",
        "pathlib 路径处理",
        (part("python18/1.md", "pathlib 路径处理"),),
    ),
    Chapter(
        19,
        "异常处理与异常组",
        "异常处理与异常组",
        (part("python19/1.md", "异常处理与异常组"),),
    ),
    Chapter(
        20,
        "dataclass-与-pydantic",
        "dataclass 与 Pydantic",
        (part("python20/1.md", "dataclass 与 Pydantic"),),
    ),
    Chapter(
        21, "上下文管理器", "上下文管理器", (part("python21/1.md", "上下文管理器"),)
    ),
    Chapter(
        22,
        "async-await-与并发",
        "async/await 与并发",
        (part("python22/1.md", "async/await 与并发"),),
    ),
    Chapter(
        23,
        "pyproject-与-uv",
        "工程基线 pyproject 与 uv",
        (part("python23/1.md", "工程基线 pyproject 与 uv"),),
    ),
    Chapter(24, "ruff", "代码风格 ruff", (part("python24/1.md", "代码风格 ruff"),)),
    Chapter(
        25, "pytest", "单元测试 pytest", (part("python25/1.md", "单元测试 pytest"),)
    ),
    Chapter(
        26, "logging", "标准日志 logging", (part("python26/1.md", "标准日志 logging"),)
    ),
    Chapter(
        27,
        "打包发布与-typer",
        "打包发布与 typer",
        (part("python27/1.md", "打包发布与 typer"),),
    ),
    Chapter(
        28, "学完之后", "学完之后做什么", (part("python28/1.md", "学完之后做什么"),)
    ),
)


HEADING_RE = re.compile(r"^(#{1,6})(\s+.*)$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
IMAGE_ONLY_RE = re.compile(
    r"^\s*(?:!\[[^]]*]\([^)]*\)|<img\b[^>]*>)\s*$", re.IGNORECASE
)


def git_revision(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def markdown_heading_levels(text: str) -> list[int]:
    levels: list[int] = []
    fence: str | None = None
    for line in text.splitlines():
        marker = FENCE_RE.match(line)
        if marker:
            current = marker.group(1)[0]
            fence = None if fence == current else current if fence is None else fence
            continue
        if fence is None and (heading := HEADING_RE.match(line)):
            levels.append(len(heading.group(1)))
    return levels


def transform_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    levels = markdown_heading_levels(text)
    offset = 3 - min(levels) if levels else 0
    output: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        line = line.expandtabs(4)
        marker = FENCE_RE.match(line)
        if marker:
            current = marker.group(1)[0]
            fence = None if fence == current else current if fence is None else fence
            output.append(line.rstrip())
            continue
        if fence is None and IMAGE_ONLY_RE.match(line):
            continue
        if fence is None and (heading := HEADING_RE.match(line)):
            level = min(6, len(heading.group(1)) + offset)
            line = f"{'#' * level}{heading.group(2)}"
        output.append(line.rstrip())
    return "\n".join(output).strip()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build(source: Path, output: Path) -> None:
    revision = git_revision(source)
    if revision != UPSTREAM_REVISION:
        raise SystemExit(f"上游版本不匹配：期望 {UPSTREAM_REVISION}，实际 {revision}。")

    corpus = output / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, object]] = []

    for chapter in CHAPTERS:
        sections = [f"# {chapter.title}"]
        source_paths: list[str] = []
        for source_part in chapter.parts:
            relative_path = SOURCE_ROOT / source_part.path
            source_path = source / relative_path
            if not source_path.is_file():
                raise SystemExit(f"缺少上游文件：{relative_path.as_posix()}")
            transformed = transform_markdown(source_path.read_text(encoding="utf-8"))
            sections.extend((f"## {source_part.title}", transformed))
            source_paths.append(relative_path.as_posix())

        content = "\n\n".join(section for section in sections if section).strip() + "\n"
        destination = corpus / chapter.filename
        destination.write_text(content, encoding="utf-8", newline="\n")
        documents.append(
            {
                "chapter": chapter.number,
                "title": chapter.title,
                "document": chapter.filename,
                "source_paths": source_paths,
                "sha256": sha256(content),
            }
        )

    manifest = {
        "schema_version": 1,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_revision": UPSTREAM_REVISION,
        "license": "CC BY 4.0",
        "transformations": [
            "按上游 README 的章节和子节顺序合并 Markdown",
            "为每章和每个上游子节增加统一标题",
            "仅在代码围栏外调整 Markdown 标题层级",
            "移除未随评测集分发的独立图片引用",
            "将制表符统一展开为四个空格",
            "统一为 UTF-8 和 LF 换行",
        ],
        "documents": documents,
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    license_path = source / "LICENSE"
    if not license_path.is_file():
        raise SystemExit("上游仓库缺少 LICENSE 文件")
    (output / "LICENSE.learn-py").write_text(
        license_path.read_text(encoding="utf-8"),
        encoding="utf-8",
        newline="\n",
    )
    print(f"已生成 {len(documents)} 个章节文档：{corpus}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="构建 learn-py.org Python 教程评测语料"
    )
    parser.add_argument("--source", type=Path, required=True, help="上游仓库本地路径")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/python_tutorial"),
        help="数据集输出目录",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    build(arguments.source.resolve(), arguments.output.resolve())

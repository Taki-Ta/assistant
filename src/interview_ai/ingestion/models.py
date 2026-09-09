import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

TIME_ZONE_LOCAL = timezone(timedelta(hours=8))
MAX_CHUNK_LENGTH = 500


@dataclass(frozen=True)
class ChangeSet:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    unchanged: tuple[str, ...]
    deleted: tuple[str, ...]
    pipeline_changed: bool = False


@dataclass
class Document:
    id: UUID = field(init=False)
    path: Path
    name: str
    create_time: datetime
    modify_time: datetime
    size: float
    content: str
    hash: str

    def __post_init__(self):
        self.id = uuid5(NAMESPACE_URL, self.path.resolve().as_uri())


@dataclass
class Chunk:
    id: UUID = field(init=False)
    document_id: UUID
    sort_index: int
    level: int
    headings: tuple[str, ...]
    content: str
    start_line: int
    end_line: int
    hash: str

    def __post_init__(self):
        identity = f"{self.document_id}-{self.sort_index}-{self.hash}"
        self.id = uuid5(NAMESPACE_URL, identity)


class State(Enum):
    added = 1
    modified = 2
    unchanged = 3
    deleted = 4


@dataclass
class ManifestPipeline:
    chunker_version: int = 1
    max_chunk_length: int = MAX_CHUNK_LENGTH

    def to_dic(self):
        return {
            "chunker_version": self.chunker_version,
            "max_chunk_length": self.max_chunk_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ManifestPipeline":
        return cls(
            chunker_version=int(data["chunker_version"]),
            max_chunk_length=int(data["max_chunk_length"]),
        )


@dataclass
class Manifest:
    schema_version: int = field(init=False, default=1)
    generated_at: datetime = field(
        init=False, default_factory=lambda: datetime.now(TIME_ZONE_LOCAL)
    )
    source_root: Path
    documents: dict[str, dict[str, object]]
    pipeline: ManifestPipeline = field(default_factory=ManifestPipeline)

    def to_dic(self):
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at.isoformat(),
            "source_root": str(self.source_root),
            "pipeline": self.pipeline.to_dic(),
            "documents": self.documents,
        }

    def to_json(self):
        return json.dumps(
            self.to_dic(),
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Manifest":
        """从 JSON 解码后的字典恢复 Manifest 及其运行时类型。"""
        schema_version = int(data["schema_version"])
        if schema_version != 1:
            raise ValueError(f"不支持的 Manifest schema_version：{schema_version}")

        generated_at = datetime.fromisoformat(str(data["generated_at"]))
        if generated_at.tzinfo is None:
            generated_at = generated_at.replace(tzinfo=TIME_ZONE_LOCAL)

        pipeline_data = data["pipeline"]
        documents = data["documents"]
        if not isinstance(pipeline_data, dict):
            raise TypeError("Manifest.pipeline 必须是对象")
        if not isinstance(documents, dict):
            raise TypeError("Manifest.documents 必须是对象")

        manifest = cls(
            source_root=Path(str(data["source_root"])),
            documents=documents,
            pipeline=ManifestPipeline.from_dict(pipeline_data),
        )
        manifest.schema_version = schema_version
        manifest.generated_at = generated_at
        return manifest

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        return cls.from_dict(json.loads(text))

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL
        REFERENCES documents (id) ON DELETE CASCADE,
    sort_index INTEGER NOT NULL,
    level INTEGER NOT NULL,
    headings TEXT[] NOT NULL,
    content TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_chunks_document_sort_index
        UNIQUE (document_id, sort_index),
    CONSTRAINT ck_chunks_sort_index CHECK (sort_index >= 0),
    CONSTRAINT ck_chunks_level CHECK (level BETWEEN 0 AND 6),
    CONSTRAINT ck_chunks_start_line CHECK (start_line >= 1),
    CONSTRAINT ck_chunks_line_range CHECK (end_line >= start_line)
);

CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id UUID PRIMARY KEY
        REFERENCES chunks (id) ON DELETE CASCADE,
    vector VECTOR(1536) NOT NULL,
    embedding_model VARCHAR(128) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_vectors_vector_hnsw
    ON vectors
    USING hnsw (vector vector_cosine_ops);

COMMIT;

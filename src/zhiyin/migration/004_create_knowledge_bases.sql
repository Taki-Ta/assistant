BEGIN;

CREATE TABLE knowledge_bases (
    id UUID PRIMARY KEY,
    owner_id VARCHAR(128) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_deleted BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_knowledge_bases_owner ON knowledge_bases(owner_id);
CREATE UNIQUE INDEX uq_knowledge_bases_default_owner
    ON knowledge_bases(owner_id) WHERE is_default = true;

CREATE TABLE knowledge_base_members (
    knowledge_base_id UUID NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    user_id VARCHAR(128) NOT NULL,
    role VARCHAR(16) NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (knowledge_base_id, user_id),
    CONSTRAINT ck_knowledge_base_members_role CHECK (role IN ('viewer', 'editor', 'admin'))
);
CREATE INDEX ix_knowledge_base_members_user ON knowledge_base_members(user_id);

ALTER TABLE documents ADD COLUMN knowledge_base_id UUID;
INSERT INTO knowledge_bases(id, owner_id, name, is_default)
SELECT md5('default-knowledge-base:' || owner_id)::uuid, owner_id, '默认知识库', true
FROM documents GROUP BY owner_id;
UPDATE documents SET knowledge_base_id = md5('default-knowledge-base:' || owner_id)::uuid;
ALTER TABLE documents ALTER COLUMN knowledge_base_id SET NOT NULL;
ALTER TABLE documents ADD CONSTRAINT fk_documents_knowledge_base
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases(id) ON DELETE RESTRICT;
CREATE INDEX ix_documents_knowledge_base ON documents(knowledge_base_id);

COMMENT ON COLUMN documents.owner_id IS '上传者身份；访问权限由所属知识库决定';
COMMIT;

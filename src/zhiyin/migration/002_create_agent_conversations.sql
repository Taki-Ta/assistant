BEGIN;

CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    owner_id VARCHAR(128) NOT NULL,
    title VARCHAR(255),
    is_deleted BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ix_sessions_owner_updated
    ON sessions (owner_id, updated_at);

CREATE TABLE turns (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL
        REFERENCES sessions (id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'in_progress',
    provider VARCHAR(64),
    model VARCHAR(128),
    final_provider_response_id VARCHAR(255),
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    CONSTRAINT uq_turns_session_sequence UNIQUE (session_id, sequence),
    CONSTRAINT ck_turns_sequence CHECK (sequence >= 0),
    CONSTRAINT ck_turns_status
        CHECK (status IN ('in_progress', 'completed', 'failed', 'cancelled')),
    CONSTRAINT ck_turns_input_tokens
        CHECK (input_tokens IS NULL OR input_tokens >= 0),
    CONSTRAINT ck_turns_output_tokens
        CHECK (output_tokens IS NULL OR output_tokens >= 0),
    CONSTRAINT ck_turns_total_tokens
        CHECK (total_tokens IS NULL OR total_tokens >= 0)
);

CREATE INDEX ix_turns_session_id ON turns (session_id);
CREATE INDEX ix_turns_status_updated ON turns (status, updated_at);

CREATE TABLE conversation_items (
    id UUID PRIMARY KEY,
    turn_id UUID NOT NULL
        REFERENCES turns (id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    item_type VARCHAR(64) NOT NULL,
    role VARCHAR(32),
    provider_item_id VARCHAR(255),
    call_id VARCHAR(255),
    tool_name VARCHAR(255),
    text_content TEXT,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_conversation_items_turn_sequence UNIQUE (turn_id, sequence),
    CONSTRAINT ck_conversation_items_sequence CHECK (sequence >= 0),
    CONSTRAINT ck_conversation_items_role
        CHECK (role IS NULL OR role IN ('system', 'developer', 'user', 'assistant'))
);

CREATE INDEX ix_conversation_items_turn_id
    ON conversation_items (turn_id);

CREATE INDEX ix_conversation_items_call_id
    ON conversation_items (call_id);

COMMIT;

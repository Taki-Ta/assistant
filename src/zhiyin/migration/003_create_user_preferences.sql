BEGIN;
CREATE TABLE user_preferences (
    owner_id VARCHAR(128) NOT NULL,
    key VARCHAR(32) NOT NULL,
    value VARCHAR(32) NOT NULL,
    source_turn_id UUID REFERENCES turns(id) ON DELETE SET NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (owner_id, key),
    CONSTRAINT ck_user_preferences_key
        CHECK (key IN ('response_language', 'answer_detail', 'code_language'))
);
COMMIT;

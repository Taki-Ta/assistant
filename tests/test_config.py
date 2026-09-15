from interview_ai.config import Settings


def test_Settings_reads_debug_flag_case_insensitively(monkeypatch):
    monkeypatch.setenv("IS_DEBUG", "TrUe")

    assert Settings().is_debug is True


def test_settings_reads_max_tool_calls(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_CALLS", "3")

    assert Settings().max_tool_calls == 3


def test_settings_reads_embedding_batch_size(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "12")

    assert Settings().embedding_batch_size == 12


def test_settings_reads_search_defaults(monkeypatch):
    monkeypatch.setenv("SEARCH_TOP_K", "7")
    monkeypatch.setenv("SEARCH_SCORE_THRESHOLD", "0.6")

    settings = Settings()

    assert settings.search_top_k == 7
    assert settings.search_score_threshold == 0.6

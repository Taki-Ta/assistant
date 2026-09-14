from interview_ai.config import Settings


def test_Settings_reads_debug_flag_case_insensitively(monkeypatch):
    monkeypatch.setenv("IS_DEBUG", "TrUe")

    assert Settings().is_debug is True


def test_settings_reads_max_tool_calls(monkeypatch):
    monkeypatch.setenv("MAX_TOOL_CALLS", "3")

    assert Settings().max_tool_calls == 3

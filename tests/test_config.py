from interview_ai.config import Settings


def test_Settings_reads_debug_flag_case_insensitively(monkeypatch):
    monkeypatch.setenv("IS_DEBUG", "TrUe")

    assert Settings().is_debug is True

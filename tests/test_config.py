from interview_ai.config import Config


def test_config_uses_defaults_for_empty_optional_values(monkeypatch):
    monkeypatch.setenv("DIMENSIONS", "")
    monkeypatch.setenv("JWT_EXP_SECONDS", "")
    monkeypatch.delenv("IS_DEBUG", raising=False)

    value = Config()

    assert value.dimensions == 256
    assert value.jwt_exp == 300
    assert value.is_debug is False


def test_config_reads_debug_flag_case_insensitively(monkeypatch):
    monkeypatch.setenv("IS_DEBUG", "TrUe")

    assert Config().is_debug is True

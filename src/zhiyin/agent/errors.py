"""Errors shared by agent providers, services and API handlers."""


class ToolCallLimitExceeded(RuntimeError):
    """The agent exceeded the configured tool-call limit."""


class ModelResponseError(RuntimeError):
    """模型未正常完成生成，不能作为完整回答保存。"""


class SessionNotFoundError(LookupError):
    """The requested session does not exist or is inaccessible."""

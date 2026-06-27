"""Tests for labs/00b_engineering_foundations.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

LAB_PATH = Path("/Users/srmallip/projects/AgenticCourse/labs/00b_engineering_foundations.py")


def _load_lab() -> object:
    """Load the lab module fresh, bypassing any cached import."""
    mod_name = "engineering_foundations_00b"
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, LAB_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chat_endpoint_returns_text() -> None:
    """POST /chat with a valid message returns 200 and the Claude reply."""
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="Hello from Claude")]
    )
    with patch("anthropic.Anthropic", return_value=mock_client):
        lab = _load_lab()
        from fastapi.testclient import TestClient

        client = TestClient(lab.create_app())
    resp = client.post("/chat", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.json()["reply"] == "Hello from Claude"


def test_chat_endpoint_rejects_empty_message() -> None:
    """POST /chat with an empty string must return HTTP 422."""
    with patch("anthropic.Anthropic"):
        lab = _load_lab()
        from fastapi.testclient import TestClient

        client = TestClient(lab.create_app())
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422

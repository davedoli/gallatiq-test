from invoice_agent import llm


def test_build_client_configures_gemini_when_its_key_is_set(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")

    def fake_chat_client(**kwargs):
        return kwargs

    monkeypatch.setattr(llm, "ChatClient", fake_chat_client)

    client = llm.build_client()

    assert client == {
        "api_key": "gemini-test-key",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }


def test_build_client_keeps_xai_precedence(monkeypatch):
    monkeypatch.setattr(llm, "load_env", lambda: None)
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key")
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    def fake_chat_client(**kwargs):
        return kwargs

    monkeypatch.setattr(llm, "ChatClient", fake_chat_client)

    client = llm.build_client()

    assert client["api_key"] == "xai-test-key"
    assert client["base_url"] == "https://api.x.ai/v1"

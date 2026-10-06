import pytest


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """No test touches ~/.agent-collab or the network unless it opts in."""
    monkeypatch.setenv("AGENT_COLLAB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_COLLAB_SEARCH", "off")
    monkeypatch.delenv("AGENT_COLLAB_MOCK", raising=False)

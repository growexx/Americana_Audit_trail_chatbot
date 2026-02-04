import builtins
import io
import os
import pytest
from types import SimpleNamespace

# import the function under test
from app.services.title import create_new_chat


# ---------------------------
# Helper / mock classes
# ---------------------------
class DummyLLMInferenceClient:
    def __init__(self, response):
        self.response = response

    def inference_single_input(self, user_query, prompt):
        return self.response


class DummyResponseExtractor:
    def __init__(self, raise_error=False, title=None):
        self.raise_error = raise_error
        self.title = title
        self.data = None

    def set_data(self, data):
        if self.raise_error:
            raise ValueError("Extractor error")
        self.data = data

    def get(self, key, default=None):
        if self.raise_error:
            raise KeyError("Missing key")
        return self.title if self.title is not None else default


class DummySQLLoader:
    def __init__(self, raise_error=False):
        self.raise_error = raise_error

    def insert_user_chat(self, user_id, chat_id, title):
        if self.raise_error:
            raise RuntimeError("DB error")
        return {"insert_user_chat": f"INSERT INTO chat VALUES('{user_id}','{chat_id}','{title}')"}


class DummyADBClient:
    def __init__(self, raise_error=False):
        self.raise_error = raise_error
        self.executed = False

    def execute_single_non_query(self, query):
        if self.raise_error:
            raise RuntimeError("Execution error")
        self.executed = True


# ---------------------------
# Pytest fixtures
# ---------------------------
@pytest.fixture
def prompt_file(tmp_path, monkeypatch):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    file_path = prompts_dir / "create_chat_title.txt"
    file_path.write_text("Title for: {user_message}")

    monkeypatch.chdir(tmp_path)
    return file_path


# ---------------------------
# Test cases
# ---------------------------

def test_happy_path_with_user_id(prompt_file):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient("raw llm output"),
        llm_response_extractor=DummyResponseExtractor(title="My Chat Title"),
        sql_loader=DummySQLLoader(),
        adb_client=DummyADBClient(),
    )

    user_id, chat_id, title = create_new_chat(
        service=service,
        user_id="user123",
        chat_id="chat123456789",
        user_query="hello"
    )

    assert user_id == "user123"
    assert chat_id == "chat123456789"
    assert title == "My Chat Title"
    assert service.adb_client.executed is True


def test_llm_returns_tuple(prompt_file):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient(("tuple output",)),
        llm_response_extractor=DummyResponseExtractor(title="Tuple Title"),
        sql_loader=DummySQLLoader(),
        adb_client=DummyADBClient(),
    )

    _, _, title = create_new_chat(
        service,
        user_id="user1",
        chat_id="abcdef123456",
        user_query="test"
    )

    assert title == "Tuple Title"


def test_extractor_failure_fallback_title(prompt_file):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient("raw"),
        llm_response_extractor=DummyResponseExtractor(raise_error=True),
        sql_loader=DummySQLLoader(),
        adb_client=DummyADBClient(),
    )

    _, chat_id, title = create_new_chat(
        service,
        user_id="user1",
        chat_id="fallback123456",
        user_query="test"
    )

    assert title == "Chat_fallback"


def test_without_user_id_no_db_call(prompt_file):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient("raw"),
        llm_response_extractor=DummyResponseExtractor(title="No User"),
        sql_loader=DummySQLLoader(),
        adb_client=DummyADBClient(),
    )

    user_id, _, title = create_new_chat(
        service,
        user_id="",
        chat_id="nouser123456",
        user_query="hello"
    )

    assert user_id == ""
    assert title == "No User"
    assert service.adb_client.executed is False


def test_db_insert_failure_is_swallowed(prompt_file):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient("raw"),
        llm_response_extractor=DummyResponseExtractor(title="DB Fail"),
        sql_loader=DummySQLLoader(raise_error=True),
        adb_client=DummyADBClient(),
    )

    # Should not raise even though DB insert fails
    _, _, title = create_new_chat(
        service,
        user_id="user1",
        chat_id="dbfail123456",
        user_query="hello"
    )

    assert title == "DB Fail"


def test_missing_prompt_file_raises_file_not_found(monkeypatch):
    service = SimpleNamespace(
        llm_inference_client=DummyLLMInferenceClient("raw"),
        llm_response_extractor=DummyResponseExtractor(title="X"),
        sql_loader=DummySQLLoader(),
        adb_client=DummyADBClient(),
    )

    def mock_open(*args, **kwargs):
        raise FileNotFoundError("Prompt file missing")

    monkeypatch.setattr("builtins.open", mock_open)

    with pytest.raises(FileNotFoundError):
        create_new_chat(
            service,
            user_id="user1",
            chat_id="noprompt123",
            user_query="hello"
        )

import pytest
from code_modules.sql_queries_loader import SqlQueryLoader


def test_load_chat_history_by_id():
    chat_id = "chat123"

    result = SqlQueryLoader.load_chat_history_by_id(chat_id)

    assert "load_chat_history" in result
    query = result["load_chat_history"]

    assert "SELECT CHAT_ID" in query
    assert f"chat_id = '{chat_id}'" in query


def test_load_user_chats_previews():
    user_id = "user456"

    result = SqlQueryLoader.load_user_chats_previews(user_id)

    assert "load_chats_preview" in result
    query = result["load_chats_preview"]

    assert "FROM USER_CHATS" in query
    assert f"user_id = '{user_id}'" in query


def test_delete_chat_queries():
    chat_id = "delete789"

    result = SqlQueryLoader.delete_chat_queries(chat_id)

    assert "delete_chat_history" in result
    assert "delete_chat_preview" in result

    assert f"chat_id = '{chat_id}'" in result["delete_chat_history"]
    assert f"chat_id = '{chat_id}'" in result["delete_chat_preview"]


def test_insert_user_chat_basic():
    result = SqlQueryLoader.insert_user_chat(
        user_id="user1",
        chat_id="chat1",
        title="My Chat"
    )

    assert "insert_user_chat" in result
    query = result["insert_user_chat"]

    assert "INSERT INTO USER_CHATS" in query
    assert "'user1'" in query
    assert "'chat1'" in query
    assert "'My Chat'" in query


def test_insert_user_chat_escapes_single_quotes():
    result = SqlQueryLoader.insert_user_chat(
        user_id="user1",
        chat_id="chat1",
        title="Bob's Chat"
    )

    query = result["insert_user_chat"]

    # Single quote must be escaped for SQL
    assert "Bob''s Chat" in query
    assert "Bob's Chat" not in query


def test_insert_chat_message_basic():
    result = SqlQueryLoader.insert_chat_message(
        chat_id="chat9",
        message_no=1,
        role="user",
        message="Hello world"
    )

    assert "insert_chat_message" in result
    query = result["insert_chat_message"]

    assert "INSERT INTO CHAT_MESSAGES" in query
    assert "'chat9'" in query
    assert "1" in query
    assert "'user'" in query
    assert "'Hello world'" in query


def test_insert_chat_message_escapes_single_quotes():
    result = SqlQueryLoader.insert_chat_message(
        chat_id="chat9",
        message_no=2,
        role="assistant",
        message="It's working"
    )

    query = result["insert_chat_message"]

    # Single quote must be escaped
    assert "It''s working" in query
    assert "It's working" not in query

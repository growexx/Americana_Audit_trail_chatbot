import types
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
import pytest
from unittest.mock import ANY
import json
import unittest.mock

# Import router module (adjust path if needed)
from app.api.routes.chat import router, service


# -------------------------------------------------------------------
# Test App Fixture
# -------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    """
    Creates a FastAPI test app with mocked ChatService
    """
    app = FastAPI()

    # Mock app.state (runtime memory)
    app.state.chat_history = {}
    app.state.last_sql_query = {}
    app.state.last_sql_queries = {}
    app.state.last_chat_id = None
    app.state.user_chats = {}

    # Attach router
    app.include_router(router)

    # Mock all ChatService methods
    service.handle_inquiry = MagicMock(return_value={"status": 0, "data": "ok"})
    service.load_chat_history = MagicMock(return_value={"status": 0, "history": []})
    service.load_user_chats_previews = MagicMock(return_value={"status": 0, "chats": []})
    service.chat_runtime_cleanup = MagicMock(return_value={"status": 0})
    service.delete_chat_history = MagicMock(return_value={"status": 0})

    return TestClient(app)


# -------------------------------------------------------------------
# /chat-inquiry
# -------------------------------------------------------------------

def test_chat_inquiry(client):
    payload = {
        "user_id": "u1",
        "chat_id": "c1",
        "user_message": "hello"
    }

    response = client.post("/chat-inquiry", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == 0

    service.handle_inquiry.assert_called_once_with(
        "u1", "c1", "hello", unittest.mock.ANY
    )


# -------------------------------------------------------------------
# /load-historical-chat
# -------------------------------------------------------------------

def test_load_chat_history(client):
    payload = {
        "user_id": "u1",
        "chat_id": "c1"
    }

    response = client.post("/load-historical-chat", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == 0

    service.load_chat_history.assert_called_once_with(
        "u1", "c1", unittest.mock.ANY
    )


# -------------------------------------------------------------------
# /load-chats-preview
# -------------------------------------------------------------------

def test_load_chat_previews(client):
    payload = {
        "user_id": "u1"
    }

    response = client.post("/load-chats-preview", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == 0

    service.load_user_chats_previews.assert_called_once_with(
        "u1", unittest.mock.ANY
    )


# -------------------------------------------------------------------
# /user-signout
# -------------------------------------------------------------------

def test_user_signout(client):
    payload = {
        "user_id": "u1"
    }

    response = client.post("/user-signout", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == 0

    service.chat_runtime_cleanup.assert_called_once_with(
        "u1", unittest.mock.ANY
    )


# -------------------------------------------------------------------
# /delete-chats
# -------------------------------------------------------------------

def test_delete_chats(client):
    payload = {
        "user_id": "u1",
        "chat_ids": ["c1", "c2"]
    }

    response = client.delete("/delete-chats", data=json.dumps(payload), headers={"Content-Type":"application/json"})

    assert response.status_code == 200
    assert response.json()["status"] == 0

    service.delete_chat_history.assert_called_once_with(
        "u1", ["c1", "c2"], unittest.mock.ANY
    )


# -------------------------------------------------------------------
# /view-state
# -------------------------------------------------------------------

def test_view_state(client):
    response = client.get("/view-state")

    assert response.status_code == 200
    body = response.json()

    assert "chat_history" in body["_state"]
    assert "last_sql_query" in body
    assert "user_chats" in body

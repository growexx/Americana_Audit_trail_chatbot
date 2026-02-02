from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd
from app.api.routes.chat import router
from app.services.chat_service import check_if_df_all_null_or_zero

from app.services.chat_service import ChatService

@patch("app.api.routes.chat.service")
def test_chat_inquiry_success(mock_service):
    """
    Sonar-safe test:
    - Mocks ChatService
    - Verifies request → service wiring
    """

    # Arrange
    app = FastAPI()
    app.state.chat_history = {}
    app.state.last_chat_id = ""
    app.state.last_sql_query = {}

    app.include_router(router, prefix="/api/v1/chat")
    client = TestClient(app)

    mock_service.handle_inquiry.return_value = {
        "status": 1,
        "response": "hello"
    }

    payload = {
        "chat_id": "chat123",
        "user_message": "Hello bot"
    }

    # Act
    response = client.post("/api/v1/chat/inquiry", json=payload)

    # Assert
    assert response.status_code == 200
    assert response.json()["status"] == 1

    mock_service.handle_inquiry.assert_called_once_with(
        "chat123",
        "Hello bot",
        app.state
    )



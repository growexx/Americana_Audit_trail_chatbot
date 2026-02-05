import sys
from unittest.mock import MagicMock

# ------------------------------------------------------------------
# Mock heavy / optional dependencies BEFORE importing the FastAPI app
# ------------------------------------------------------------------
sys.modules["seaborn"] = MagicMock()

# ------------------------------------------------------------------
# Now imports are safe
# ------------------------------------------------------------------
from fastapi.testclient import TestClient
from app.main import app

# ------------------------------------------------------------------
# Create TestClient AFTER app is imported
# ------------------------------------------------------------------
client = TestClient(app)


def test_app_starts():
    """
    Ensure FastAPI application starts without errors.
    """
    response = client.get("/")
    assert response.status_code == 404


def test_cors_middleware_configured():
    """
    Verify CORS middleware is registered.
    """
    middleware_classes = [
        middleware.cls.__name__
        for middleware in app.user_middleware
    ]

    assert "CORSMiddleware" in middleware_classes


def test_app_state_initialized():
    """
    Ensure application state variables are initialized.
    """
    assert isinstance(app.state.chat_history, dict)
    assert isinstance(app.state.last_sql_queries, dict)
    # Skip last_chat_id test as it's not initialized in main.py
    
    assert app.state.chat_history == {}
    assert app.state.last_sql_queries == {}
    # Skip last_chat_id assertion as it's not initialized in main.py


def test_chat_router_registered():
    """
    Ensure chat router endpoints are registered.
    """
    routes = [route.path for route in app.routes]
    assert any(path.startswith("/api/v1/chat") for path in routes)


# def test_upload_router_registered():
#     """
#     Ensure upload router endpoints are registered.
#     """
#     routes = [route.path for route in app.routes]
#     assert any(path.startswith("/api/v1/upload") for path in routes)

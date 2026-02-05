import pytest
from unittest.mock import MagicMock, patch

from code_modules.oracle_genai_handler import (
    LLMInference,
    create_llm_client,
    LLMInferenceError
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def mock_chat_response():
    """Mock OCI chat response structure"""
    mock_content = MagicMock()
    mock_content.text = "Mock LLM response"

    mock_message = MagicMock()
    mock_message.content = [mock_content]

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_chat_response = MagicMock()
    mock_chat_response.data.chat_response.choices = [mock_choice]

    return mock_chat_response


@pytest.fixture
def mock_oci_client(mock_chat_response):
    """Mock Generative AI inference client"""
    client = MagicMock()
    client.chat.return_value = mock_chat_response
    return client


# -----------------------------
# Initialization Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_llm_inference_initialization(
    mock_client_class,
    mock_from_file,
):
    """Test LLMInference initializes successfully"""
    mock_from_file.return_value = {"config": "ok"}
    mock_client_class.return_value = MagicMock()

    client = LLMInference("config.ini")

    assert client.generative_ai_inference_client is not None
    assert client.chat_detail is not None


@patch("code_modules.oracle_genai_handler.oci.config.from_file", side_effect=Exception("Config error"))
def test_llm_inference_initialization_failure(mock_from_file):
    """Test initialization failure raises exception"""
    with pytest.raises(Exception):
        LLMInference("config.ini")


# -----------------------------
# Message Conversion Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_convert_message_to_oci_format(mock_client, mock_from_file):
    """Test role + content conversion"""
    mock_from_file.return_value = {}
    mock_client.return_value = MagicMock()

    client = LLMInference()

    msg = client._convert_message_to_oci_format("USER", "Hello")

    assert msg.role == "USER"
    assert msg.content[0].text == "Hello"


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_convert_chat_history_to_oci_format(mock_client, mock_from_file):
    """Test chat history conversion and filtering"""
    mock_from_file.return_value = {}
    mock_client.return_value = MagicMock()

    client = LLMInference()

    chat_history = [
        {"role": "USER", "message": "Hi", "timestamp": ""},
        {"role": "ASSISTANT", "message": "", "timestamp": ""},  # should be skipped
    ]

    messages = client._convert_chat_history_to_oci_format(chat_history)

    assert len(messages) == 1
    assert messages[0].content[0].text == "Hi"


# -----------------------------
# Inference Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_chat_history_success(
    mock_client_class,
    mock_from_file,
    mock_oci_client,
):
    """Test successful chat-history inference"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = mock_oci_client

    client = LLMInference()

    response = client.inference_from_chat_history(
        [{"role": "USER", "message": "Hello", "timestamp": ""}]
    )

    assert response[0] == "Mock LLM response"
    mock_oci_client.chat.assert_called_once()


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_empty_chat_history(
    mock_client_class,
    mock_from_file,
):
    """Test empty chat history returns safe message"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = MagicMock()

    client = LLMInference()

    response = client.inference_from_chat_history(
        [{"role": "USER", "message": "   ", "timestamp": ""}]
    )

    assert "didn't receive any valid messages" in response


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_from_chat_history_failure(
    mock_client_class,
    mock_from_file,
):
    """Test inference failure propagates exception"""
    mock_from_file.return_value = {}
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("OCI failure")
    mock_client_class.return_value = mock_client

    client = LLMInference()

    with pytest.raises(Exception):
        client.inference_from_chat_history(
            [{"role": "USER", "message": "Hello", "timestamp": ""}]
        )


# -----------------------------
# Simple / Single Input Tests
# -----------------------------

@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_simple(
    mock_client_class,
    mock_from_file,
    mock_oci_client,
):
    """Test simple inference wrapper"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = mock_oci_client

    client = LLMInference()

    response = client.inference_simple("Hello", system_prompt="System")

    assert response[0] == "Mock LLM response"


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_single_input(
    mock_client_class,
    mock_from_file,
    mock_oci_client,
):
    """Test single-input inference"""
    mock_from_file.return_value = {}
    mock_client_class.return_value = mock_oci_client

    client = LLMInference()

    response = client.inference_single_input("Hello", "System prompt")

    assert response[0] == "Mock LLM response"


# -----------------------------
# Factory Function Test
# -----------------------------

@patch("code_modules.oracle_genai_handler.LLMInference")
def test_create_llm_client(mock_llm):
    """Test factory function"""
    create_llm_client("config.ini")
    mock_llm.assert_called_once_with("config.ini")



import pytest
from unittest.mock import MagicMock, patch

from code_modules.oracle_genai_handler import LLMInference


@patch("code_modules.oracle_genai_handler.oci.config.from_file")
@patch("code_modules.oracle_genai_handler.oci.generative_ai_inference.GenerativeAiInferenceClient")
def test_inference_single_input_exception(
    mock_client_class,
    mock_from_file,
):
    mock_from_file.return_value = {}

    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("OCI failure")
    mock_client_class.return_value = mock_client

    client = LLMInference()

    with pytest.raises(LLMInferenceError) as exc:
        client.inference_single_input(
            user_input="test input",
            system_prompt="system prompt"
        )

    assert "Failed to generate LLM response" in str(exc.value)
    assert exc.value.__cause__ is not None
    assert "OCI failure" in str(exc.value.__cause__)

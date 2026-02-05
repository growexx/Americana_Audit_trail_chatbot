import pytest
from code_modules.llm_response_extractor import extract_json, LLMResponseExtractor


def test_extract_json_plain_json():
    text = '{"a": 1, "b": "test"}'
    result = extract_json(text)
    assert result == {"a": 1, "b": "test"}


def test_extract_json_embedded_json():
    text = "Here is some text {\"x\": 10, \"y\": 20} end"
    result = extract_json(text)
    assert result == {"x": 10, "y": 20}


def test_extract_json_fenced_json():
    text = """
    ```json
    {
        "name": "ashish",
        "age": 25
    }
    ```
    """
    result = extract_json(text)
    assert result["name"] == "ashish"
    assert result["age"] == 25


def test_extract_json_invalid_json():
    text = "this is not json"
    with pytest.raises(Exception):
        extract_json(text)


def test_llm_response_extractor_set_and_get():
    text = '{"key1": "value1", "key2": 2}'
    extractor = LLMResponseExtractor()
    extractor.set_data(text)

    assert extractor.get("key1") == "value1"
    assert extractor.get("missing") is None
    assert extractor.get("missing", "default") == "default"


def test_llm_response_extractor_get_many():
    text = '{"a": 1, "b": 2}'
    extractor = LLMResponseExtractor()
    extractor.set_data(text)

    result = extractor.get_many(["a", "b", "c"], defaults={"c": 3})
    assert result == (1, 2, 3)

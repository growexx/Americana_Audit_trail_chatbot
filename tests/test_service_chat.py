import pytest
import pandas as pd
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from app.services.chat_service import ChatService

from app.services.chat_service import (
    ChatService,
    prepare_metadata_string,
    generate_categorical_plots,
    check_if_df_all_null_or_zero,
    JumpToFinally
)

# ============================================
# Helper fixtures
# ============================================

@pytest.fixture
def app_state():
    """Simulated FastAPI app.state"""
    return SimpleNamespace(
        chat_history={},
        last_sql_query={},
        last_sql_queries={},
        last_chat_id=None,
        user_chats={}
    )

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Category": ["A", "B", "C"],
        "Value": [10, 20, 30]
    })

@pytest.fixture
def chat_service():
    """ChatService instance with mocked clients"""
    service = ChatService()
    # Mock clients
    service.llm_inference_client = MagicMock()
    service.llm_response_extractor = MagicMock()
    service.prompt_generator_client = MagicMock()
    service.adb_client = MagicMock()
    service.sql_loader = MagicMock()
    return service

# ============================================
# Utility function tests
# ============================================

def test_prepare_metadata_string(tmp_path, monkeypatch):
    # create fake json file
    table_file = tmp_path / "table_metadata"
    table_file.mkdir()
    file_path = table_file / "table1.json"
    file_path.write_text(json.dumps({"col": "val"}))

    monkeypatch.chdir(tmp_path)
    os.makedirs("table_metadata", exist_ok=True)
    file_path.write_text(json.dumps({"col": "val"}))

    result = prepare_metadata_string(["table1"])
    assert "TABLE1" in result
    assert '"col": "val"' in result

def test_generate_categorical_plots_single(sample_df, tmp_path):
    paths = generate_categorical_plots(sample_df, str(tmp_path), "test")
    assert len(paths) == 1
    assert os.path.exists(paths[0])


def test_handle_inquiry_with_mocked_metadata():
    chat_service = ChatService()
    
    # Mock the prepare_metadata_string to avoid file access
    with patch("app.services.chat_service.prepare_metadata_string") as mock_meta:
        mock_meta.return_value = "FAKE_METADATA"
        
        # Also mock other dependencies to avoid real DB/LLM calls
        chat_service.guard_rail = MagicMock(return_value=("yes", ["T1"]))
        chat_service.text_2_sql = MagicMock(return_value=("SELECT 1", "analysis", 0))
        chat_service.adb_client.execute_query_df = MagicMock(return_value=MagicMock(empty=False, to_dict=lambda **kwargs: [{"a":1}]))
        chat_service.prompt_generator_client.generate_main_prompt = MagicMock(return_value="MP")
        chat_service.prompt_generator_client.generate_assistant_prompt = MagicMock(return_value="AP")
        chat_service.llm_inference_client.inference_from_chat_history = MagicMock(return_value="LLM")
        chat_service.llm_response_extractor.get_many = MagicMock(return_value=("analysis","message"))
        
        # Fake app_state
        class DummyState:
            chat_history = {}
            last_sql_query = {}
            last_sql_queries = {}
            last_chat_id = None
            user_chats = {}
        app_state = DummyState()
        
        response = chat_service.handle_inquiry("c1", "query", app_state)
        
        assert response["status"] == 1
        assert response["llm_response"] == "message"

def test_generate_categorical_plots_two_categoricals(tmp_path):
    df = pd.DataFrame({
        "cat1": ["A", "B", "C", "A"],
        "cat2": ["X", "Y", "Z", "X"],
        "val": [1,2,3,4]
    })
    paths = generate_categorical_plots(df, str(tmp_path), "test")
    assert len(paths) == 1
    assert os.path.exists(paths[0])

def test_check_if_df_all_null_or_zero():
    df1 = pd.DataFrame({"a":[0,None],"b":[0,None]})
    df2 = pd.DataFrame({"a":[1,0],"b":[0,0]})
    assert check_if_df_all_null_or_zero(df1) is True
    assert check_if_df_all_null_or_zero(df2) is False

# ============================================
# Guardrail / Text2SQL
# ============================================

def test_guard_rail(chat_service):
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = {"r":"yes","t":["T1"]}
    chat_service.llm_response_extractor.get_many.return_value = ("yes", ["T1"])
    relevant, tables = chat_service.guard_rail("hello")
    assert relevant == "yes"
    assert tables == ["T1"]

def test_text_2_sql(chat_service, monkeypatch):
    chat_service.prompt_generator_client.generate_sql_prompt.return_value = "PROMPT"
    chat_service.llm_inference_client.inference_single_input.return_value = '{"sql_query":"SELECT 1","scenario":"analysis","error_status":0}'
    chat_service.llm_response_extractor.get_many.return_value = ("SELECT 1","analysis",0)
    sql, scenario, error = chat_service.text_2_sql("msg", ["T1"], None, "meta")
    assert sql == "SELECT 1"
    assert scenario == "analysis"
    assert error == 0

# ============================================
# handle_inquiry scenarios
# ============================================

def test_handle_inquiry_guardrail_reject(chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("no", []))
    response = chat_service.handle_inquiry("c1", "bad query", app_state)
    assert response["status"] == 2
    assert "irrelevance" in response["llm_response"]

def test_handle_inquiry_sql_error(chat_service, app_state, monkeypatch):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["T1"]))
    chat_service.text_2_sql = MagicMock(return_value=("SQL", "analysis", 1))
    response = chat_service.handle_inquiry("c1", "query", app_state)
    assert response["status"] == 2
    assert "extends beyond the scope" in response["llm_response"]

def test_handle_inquiry_empty_df(chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["T1"]))
    chat_service.text_2_sql = MagicMock(return_value=("SELECT 1","analysis",0))
    chat_service.adb_client.execute_query_df.return_value = pd.DataFrame()
    response = chat_service.handle_inquiry("c1","query", app_state)
    assert response["status"] == 3
    assert "No data found" in response["llm_response"]

def test_handle_inquiry_raw_data(chat_service, app_state, monkeypatch, tmp_path):
    df = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
    chat_service.guard_rail = MagicMock(return_value=("yes", ["T1"]))
    chat_service.text_2_sql = MagicMock(return_value=("SELECT 1","raw_data",0))
    chat_service.adb_client.execute_query_df.return_value = df
    chat_service.prompt_generator_client.generate_main_prompt.return_value = "MP"
    chat_service.prompt_generator_client.generate_assistant_prompt.return_value = "AP"
    chat_service.llm_inference_client.inference_from_chat_history.return_value = "LLM"
    chat_service.llm_response_extractor.get_many.return_value = ("raw_data","message")
    # Mock object storage
    class DummyOCI:
        def upload_file_to_bucket(self, f, bucket_folder_name): return {"ok":True}
    monkeypatch.setattr(chat_service,"prepare_raw_data_response",chat_service.prepare_raw_data_response)
    response = chat_service.handle_inquiry("c1","query", app_state)
    assert response["status"] == 1

def test_handle_inquiry_analysis(chat_service, app_state, monkeypatch):
    df = pd.DataFrame({"a":[1,2,3],"b":[4,5,6]})
    chat_service.guard_rail = MagicMock(return_value=("yes", ["T1"]))
    chat_service.text_2_sql = MagicMock(return_value=("SELECT 1","analysis",0))
    chat_service.adb_client.execute_query_df.return_value = df
    chat_service.prompt_generator_client.generate_main_prompt.return_value = "MP"
    chat_service.prompt_generator_client.generate_assistant_prompt.return_value = "AP"
    chat_service.llm_inference_client.inference_from_chat_history.return_value = "LLM"
    chat_service.llm_response_extractor.get_many.return_value = ("analysis","message")
    response = chat_service.handle_inquiry("c1","query", app_state)
    assert response["status"] == 1
    assert "llm_response" in response

# ============================================
# load_chat_history
# ============================================

def test_load_chat_history_empty(chat_service, app_state):
    chat_service.sql_loader.load_chat_history_by_id.return_value = {"load_chat_history":"SELECT 1"}
    chat_service.adb_client.execute_query_df.return_value = pd.DataFrame()
    res = chat_service.load_chat_history("u1","c1",app_state)
    assert res["status"] == 0

def test_load_chat_history_with_sql(chat_service, app_state):
    df = pd.DataFrame({
        "ROLE":["User","SQL"],
        "MESSAGE":["hi","SELECT 1"],
        "MESSAGE_NO":[1,2]
    })
    chat_service.sql_loader.load_chat_history_by_id.return_value = {"load_chat_history":"SELECT 1"}
    chat_service.adb_client.execute_query_df.side_effect = [df, pd.DataFrame([{"val":1}])]
    chat_service.prompt_generator_client.generate_main_prompt.return_value = "MP"
    res = chat_service.load_chat_history("u1","c1",app_state)
    assert res["status"] == 1

# ============================================
# load_user_chats_previews
# ============================================

def test_load_user_chats_previews_success(chat_service):
    df = pd.DataFrame([{"chat":"c1"}])
    chat_service.sql_loader.load_user_chats_previews.return_value = {"load_chats_preview":"SELECT 1"}
    chat_service.adb_client.execute_query_df.return_value = df
    res = chat_service.load_user_chats_previews("u1",SimpleNamespace())
    assert res["status"] == 1

def test_load_user_chats_previews_error(chat_service):
    chat_service.sql_loader.load_user_chats_previews.side_effect = Exception("fail")
    res = chat_service.load_user_chats_previews("u1",SimpleNamespace())
    assert res["status"] == 0

# ============================================
# chat_runtime_cleanup
# ============================================

def test_chat_runtime_cleanup_success(chat_service):
    state = SimpleNamespace(
        chat_history={"c1":[]},
        last_sql_query={"c1":"SQL"},
        user_chats={"u1":["c1"]}
    )
    res = chat_service.chat_runtime_cleanup("u1", state)
    assert res["status"] == 1
    assert "Deleted" in res["message"]

def test_chat_runtime_cleanup_nodata(chat_service):
    state = SimpleNamespace(chat_history={}, last_sql_query={}, user_chats={})
    res = chat_service.chat_runtime_cleanup("u1", state)
    assert res["status"] == 0

# ============================================
# delete_chat_history
# ============================================

def test_delete_chat_history_success(chat_service):
    state = SimpleNamespace(
        chat_history={"c1":[]},
        last_sql_queries={"c1":"SQL"}
    )
    chat_service.sql_loader.delete_chat_queries.return_value = {"delete_chat_history":"Q1","delete_chat_preview":"Q2"}
    chat_service.adb_client.execute_single_non_query.return_value = None
    res = chat_service.delete_chat_history("u1",["c1"],state)
    assert res["status"] == 1

def test_delete_chat_history_error(chat_service):
    state = SimpleNamespace(chat_history={"c1":[]}, last_sql_queries={"c1":"SQL"})
    chat_service.sql_loader.delete_chat_queries.side_effect = Exception("fail")
    res = chat_service.delete_chat_history("u1",["c1"],state)
    assert res["status"] == 0

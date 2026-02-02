import pandas as pd
import pytest
from unittest.mock import MagicMock, patch



from app.services.chat_service import (
    ChatService,
    generate_categorical_plots,
    prepare_metadata_string,
)


# ------------------------------------------------------------------
# Dummy App State
# ------------------------------------------------------------------

class DummyAppState:
    def __init__(self):
        self.chat_history = {}
        self.last_chat_id = ""
        self.last_sql_query = {}


@pytest.fixture
def app_state():
    return DummyAppState()


@pytest.fixture
def chat_service():
    with patch("app.services.chat_service.create_llm_client"), \
         patch("app.services.chat_service.LLMResponseExtractor"), \
         patch("app.services.chat_service.PromptGenerator"):
        return ChatService()


# ------------------------------------------------------------------
# prepare_metadata_string tests
# ------------------------------------------------------------------

@patch("builtins.open")
@patch("app.services.chat_service.json.load")
def test_prepare_metadata_string(mock_json_load, mock_open):
    mock_json_load.return_value = {"col": "value"}

    result = prepare_metadata_string(["property_sales"])

    assert "PROPERTY_SALES" in result
    assert "col" in result


# ------------------------------------------------------------------
# prepare_raw_data_response tests
# ------------------------------------------------------------------

@patch.object(pd.DataFrame, "to_csv")
@patch("app.services.chat_service.os.remove")
def test_prepare_raw_data_response(mock_remove, mock_to_csv, chat_service):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    mock_storage = MagicMock()
    mock_storage.upload_file_to_bucket.return_value = "uploaded"

    response = chat_service.prepare_raw_data_response(
        selected_df=df,
        scenario="raw_data",
        sql_query="SELECT *",
        object_storage_client=mock_storage,
        message="Here is raw data"
    )

    assert response["status"] == 1
    assert response["scenario"] == "raw_data"
    assert response["file_path"].endswith(".csv")
    assert "llm_response" in response
    mock_to_csv.assert_called_once()
    mock_remove.assert_called_once()


# ------------------------------------------------------------------
# prepare_analysis_response tests
# ------------------------------------------------------------------

@patch("app.services.chat_service.generate_categorical_plots", return_value=["plot.png"])
@patch("app.services.chat_service.os.remove")
def test_prepare_analysis_response_with_plot(
    mock_remove,
    _,
    chat_service
):
    df = pd.DataFrame({
        "City": ["A"],
        "Sales": [100]
    })

    mock_storage = MagicMock()
    mock_storage.upload_file_to_bucket.return_value = "uploaded"

    response = chat_service.prepare_analysis_response(
        selected_df=df,
        scenario="summary",
        sql_query="SELECT *",
        object_storage_client=mock_storage,
        message="Analysis result"
    )

    assert response["status"] == 1
    assert response["diagram"] is not None
    assert response["scenario"] == "summary"
    assert "results_df" in response
    mock_remove.assert_called_once()


@patch("app.services.chat_service.generate_categorical_plots", return_value=[])
def test_prepare_analysis_response_no_plot(_, chat_service):
    df = pd.DataFrame({
        "City": ["A"],
        "Sales": [100]
    })

    response = chat_service.prepare_analysis_response(
        selected_df=df,
        scenario="summary",
        sql_query="SELECT *",
        object_storage_client=MagicMock(),
        message="Analysis result"
    )

    assert response["status"] == 1
    assert response["diagram"] is None


# ------------------------------------------------------------------
# guard_rail tests
# ------------------------------------------------------------------

def test_guard_rail_relevant(chat_service):
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = {
        "relevant_question": "yes",
        "tables_related": ["property_sales_summary"]
    }

    chat_service.llm_response_extractor.get_many.return_value = (
        "yes", ["property_sales_summary"]
    )

    relevant, tables = chat_service.guard_rail("show property_sales_summary")

    assert relevant == "yes"
    assert tables == ["property_sales_summary"]


def test_guard_rail_not_relevant(chat_service):
    chat_service.prompt_generator_client.guardrail_check_inference_call.return_value = {}
    chat_service.llm_response_extractor.get_many.return_value = ("no", [])

    relevant, tables = chat_service.guard_rail("random joke")

    assert relevant == "no"
    assert tables == []


# ------------------------------------------------------------------
# text_2_sql tests
# ------------------------------------------------------------------

def test_text_2_sql_success(chat_service):
    chat_service.prompt_generator_client.generate_sql_prompt.return_value = "SQL_PROMPT"
    chat_service.llm_inference_client.inference_single_input.return_value = {}
    chat_service.llm_response_extractor.get_many.return_value = (
        "SELECT * FROM property_sales_summary", "summary", 0
    )

    sql, scenario, error = chat_service.text_2_sql(
        "show property_sales_summary",
        ["property_sales_summary"],
        None,
        "META"
    )

    assert sql.startswith("SELECT")
    assert scenario == "summary"
    assert error == 0


def test_text_2_sql_error(chat_service):
    chat_service.prompt_generator_client.generate_sql_prompt.return_value = "SQL_PROMPT"
    chat_service.llm_inference_client.inference_single_input.return_value = {}
    chat_service.llm_response_extractor.get_many.return_value = (
        "", "", 1
    )

    sql, scenario, error = chat_service.text_2_sql(
        "bad query",
        ["property_sales_summary"],
        None,
        "META"
    )

    assert error == 1


# ------------------------------------------------------------------
# handle_inquiry tests
# ------------------------------------------------------------------

def test_handle_inquiry_guardrail_rejected(chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("no", []))

    response = chat_service.handle_inquiry(
        chat_id="c1",
        user_message="irrelevant",
        app_state=app_state
    )

    assert response["status"] == 2
    assert "rejected" in response["llm_response"].lower()


@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
@patch("app.services.chat_service.OracleADBClient")
def test_handle_inquiry_empty_df(mock_adb, _, chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["table"]))
    chat_service.text_2_sql = MagicMock(return_value=("SELECT *", "summary", 0))

    mock_adb.return_value.execute_query_df.return_value = pd.DataFrame()

    response = chat_service.handle_inquiry("c1", "query", app_state)

    assert response["status"] == 3



@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
@patch("app.services.chat_service.OracleADBClient")
def test_handle_inquiry_all_zero_df(mock_adb, _, chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["table"]))
    chat_service.text_2_sql = MagicMock(return_value=("SELECT *", "summary", 0))

    # This DF is all zeros → triggers check_if_df_all_null_or_zero
    df = pd.DataFrame({"amount": [0, 0, 0]})
    mock_adb.return_value.execute_query_df.return_value = df

    response = chat_service.handle_inquiry("c1", "query", app_state)

    assert response["status"] == 3



@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
def test_handle_inquiry_text2sql_error(_, chat_service, app_state):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["table"]))
    chat_service.text_2_sql = MagicMock(return_value=("", "", 1))

    response = chat_service.handle_inquiry("c1", "query", app_state)

    assert response["status"] == 2



@patch("app.services.chat_service.generate_categorical_plots", return_value=[])
@patch("app.services.chat_service.OCIObjectStorageClient")
@patch("app.services.chat_service.OracleADBClient")
@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
def test_handle_inquiry_success_summary(
    _,
    mock_adb,
    __,
    ___,
    chat_service,
    app_state
):
    chat_service.guard_rail = MagicMock(return_value=("yes", ["property_sales_summary"]))
    chat_service.text_2_sql = MagicMock(
        return_value=("SELECT * FROM property_sales_summary", "summary", 0)
    )

    df = pd.DataFrame({"city": ["A"], "amount": [100]})
    mock_adb.return_value.execute_query_df.return_value = df

    chat_service.llm_inference_client.inference_from_chat_history.return_value = {}
    chat_service.llm_response_extractor.get_many.return_value = (
        "summary", "Here is result"
    )

    response = chat_service.handle_inquiry(
        "c1", "show property_sales_summary", app_state
    )

    assert response["status"] == 1
    assert response["scenario"] == "summary"
    assert "results_df" in response


def test_handle_inquiry_exception(chat_service, app_state):
    """
    Covers:
    - Generic exception handling path
    - Ensures fallback error response is returned
    """

    chat_service.guard_rail = MagicMock(side_effect=Exception("boom"))

    response = chat_service.handle_inquiry(
        "c1",
        "fail",
        app_state
    )

    assert response["status"] == 0
    assert response["chat_id"] == "c1"
    assert "failed due to error" in response["llm_response"].lower()



# ------------------------------------------------------------------
# generate_categorical_plots tests
# ------------------------------------------------------------------

def test_generate_plots_no_categorical(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []


def test_generate_plots_no_numeric(tmp_path):
    df = pd.DataFrame({"category": ["A", "B", "C"]})

    result = generate_categorical_plots(df, tmp_path, "test")

    assert result == []


@patch("app.services.chat_service.plt.savefig")
@patch("app.services.chat_service.sns.barplot")
def test_generate_plots_single_categorical(
    mock_barplot,
    mock_savefig,
    tmp_path
):
    df = pd.DataFrame({
        "City": ["A", "B", "C"],
        "Sales": [10, 20, 30]
    })

    result = generate_categorical_plots(df, tmp_path, "sales")

    assert len(result) == 1
    assert result[0].endswith("_City_bar.png")
    mock_barplot.assert_called_once()
    mock_savefig.assert_called_once()


@patch("app.services.chat_service.plt.savefig")
@patch("app.services.chat_service.sns.barplot")
def test_generate_plots_two_categorical_grouped_bar(
    mock_barplot,
    mock_savefig,
    tmp_path
):
    df = pd.DataFrame({
        "City": ["A", "A", "B", "B"],
        "Type": ["X", "Y", "X", "Y"],
        "Sales": [10, 20, 30, 40]
    })

    result = generate_categorical_plots(df, tmp_path, "sales")

    assert len(result) == 1
    assert "_grouped_bar.png" in result[0]
    mock_barplot.assert_called_once()
    mock_savefig.assert_called_once()


@patch("app.services.chat_service.plt.savefig")
@patch("app.services.chat_service.sns.heatmap")
def test_generate_plots_two_categorical_heatmap(
    mock_heatmap,
    mock_savefig,
    tmp_path
):
    df = pd.DataFrame({
        "City": ["A"] * 6 + ["B"] * 6,
        "Type": ["T1", "T2", "T3", "T4", "T5", "T6"] * 2,
        "Sales": range(12)
    })

    result = generate_categorical_plots(df, tmp_path, "sales")

    assert len(result) == 1
    assert "_heatmap.png" in result[0]
    mock_heatmap.assert_called_once()
    mock_savefig.assert_called_once()



import pandas as pd
from unittest.mock import MagicMock, patch

@patch("app.services.chat_service.generate_categorical_plots", return_value=[])
@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
@patch("app.services.chat_service.OCIObjectStorageClient")
@patch("app.services.chat_service.OracleADBClient")
def test_handle_inquiry_removes_previous_chat_history(
    mock_adb,
    mock_oci,
    _,
    __,
    chat_service,
    app_state
):
    """
    Covers:
    if last_chat_id and last_chat_id in chat_history:
        del chat_history[last_chat_id]
        del last_sql_query_of_chat[last_chat_id]
    """

    # ---------------------------
    # Arrange
    # ---------------------------
    old_chat_id = "old_chat"
    new_chat_id = "new_chat"

    # Existing previous chat state
    app_state.chat_history[old_chat_id] = [
        {"role": "Assistant", "message": "old history"}
    ]
    app_state.last_sql_query[old_chat_id] = "OLD SQL"
    app_state.last_chat_id = old_chat_id

    chat_service.guard_rail = MagicMock(
        return_value=("yes", ["property_sales_summary"])
    )

    chat_service.text_2_sql = MagicMock(
        return_value=("SELECT * FROM property_sales_summary", "summary", 0)
    )

    df = pd.DataFrame({
        "city": ["A"],
        "amount": [100]
    })
    mock_adb.return_value.execute_query_df.return_value = df

    chat_service.llm_inference_client.inference_from_chat_history.return_value = {
        "scenario": "summary",
        "message": "Here is the result"
    }

    chat_service.llm_response_extractor.get_many.return_value = (
        "summary",
        "Here is the result"
    )

    # ---------------------------
    # Act
    # ---------------------------
    response = chat_service.handle_inquiry(
        new_chat_id,
        "show property sales",
        app_state
    )

    # ---------------------------
    # Assert
    # ---------------------------
    # ✅ Old chat removed
    assert old_chat_id not in app_state.chat_history
    assert old_chat_id not in app_state.last_sql_query

    # ✅ New chat created
    assert new_chat_id in app_state.chat_history
    assert app_state.last_chat_id == new_chat_id

    # ✅ Successful response
    assert response["status"] == 1



import pandas as pd
from unittest.mock import MagicMock, patch

@patch("app.services.chat_service.os.remove")
@patch("app.services.chat_service.OCIObjectStorageClient")
@patch("app.services.chat_service.prepare_metadata_string", return_value="META")
@patch("app.services.chat_service.OracleADBClient")
def test_handle_inquiry_raw_data_scenario(
    mock_adb,
    _,
    mock_oci,
    mock_remove,
    chat_service,
    app_state
):
    """
    Covers:
    if scenario == "raw_data":
        return self.prepare_raw_data_response(...)
    """

    # ---------------------------
    # Arrange
    # ---------------------------
    chat_service.guard_rail = MagicMock(
        return_value=("yes", ["property_sales"])
    )

    chat_service.text_2_sql = MagicMock(
        return_value=("SELECT * FROM property_sales", "raw_data", 0)
    )

    df = pd.DataFrame({
        "city": ["A", "B"],
        "amount": [100, 200]
    })
    mock_adb.return_value.execute_query_df.return_value = df

    # Mock OCI upload
    mock_oci.return_value.upload_file_to_bucket.return_value = "uploaded"

    chat_service.llm_inference_client.inference_from_chat_history.return_value = {
        "scenario": "raw_data",
        "message": "Here is raw data"
    }

    chat_service.llm_response_extractor.get_many.return_value = (
        "raw_data",
        "Here is raw data"
    )

    # ---------------------------
    # Act
    # ---------------------------
    response = chat_service.handle_inquiry(
        "chat_raw",
        "download raw data",
        app_state
    )

    # ---------------------------
    # Assert
    # ---------------------------
    assert response["status"] == 1
    assert response["scenario"] == "raw_data"
    assert "file_path" in response
    assert response["llm_response"] == "Here is raw data"

    # CSV cleanup happened
    mock_remove.assert_called_once()



import pandas as pd
from unittest.mock import patch
from app.services.chat_service import generate_categorical_plots


@patch("app.services.chat_service.plt.savefig")
@patch("app.services.chat_service.os.makedirs")
def test_generate_plots_single_category_reduce_categories(
    mock_makedirs,
    mock_savefig
):
    """
    Covers:
    if df[cat].nunique() > max_categories
    """

    # 20 unique categories → exceeds max_categories=15
    df = pd.DataFrame({
        "city": [f"City_{i}" for i in range(20)],
        "amount": list(range(20))
    })

    paths = generate_categorical_plots(
        df=df,
        output_dir="plots",
        file_prefix="test",
        max_categories=15
    )

    # Assertions
    assert len(paths) == 1
    assert paths[0].endswith("_city_bar.png")

    mock_makedirs.assert_called_once()
    mock_savefig.assert_called_once()



import pandas as pd
from unittest.mock import patch
from app.services.chat_service import generate_categorical_plots


@patch("app.services.chat_service.plt.savefig")
@patch("app.services.chat_service.os.makedirs")
def test_generate_plots_two_categories_reduce_cat1(
    mock_makedirs,
    mock_savefig
):
    """
    Covers:
    if df[cat1].nunique() > max_categories
    """

    # cat1 has 20 unique values (>15)
    df = pd.DataFrame({
        "region": [f"R{i}" for i in range(20)],
        "type": ["A"] * 20,   # cat2 nunique <= 5 → grouped bar
        "amount": list(range(20))
    })

    paths = generate_categorical_plots(
        df=df,
        output_dir="plots",
        file_prefix="test",
        max_categories=15
    )

    # Assertions
    assert len(paths) == 1
    assert paths[0].endswith("_region_type_grouped_bar.png")

    mock_makedirs.assert_called_once()
    mock_savefig.assert_called_once()

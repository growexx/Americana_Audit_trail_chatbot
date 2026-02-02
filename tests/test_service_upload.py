import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.services.upload_service import UploadService

@patch("app.services.upload_service.pd.read_csv")
def test_file_2_df_csv(mock_read_csv):
    service = UploadService()

    df = pd.DataFrame({"A": [1]})
    mock_read_csv.return_value = df

    result = service.file_2_df("data.csv")

    mock_read_csv.assert_called_once_with("data.csv")
    assert result.equals(df)

@patch("app.services.upload_service.pd.read_excel")
def test_file_2_df_xlsx(mock_read_excel):
    service = UploadService()

    df = pd.DataFrame({"A": [1]})
    mock_read_excel.return_value = df

    result = service.file_2_df("data.xlsx")

    mock_read_excel.assert_called_once_with("data.xlsx")
    assert result.equals(df)

@patch("app.services.upload_service.os.remove")
@patch("app.services.upload_service.json.load")
@patch("app.services.upload_service.open")
@patch("app.services.upload_service.OracleADBClient")
@patch("app.services.upload_service.OCIObjectStorageClient")
@patch("app.services.upload_service.load_adw_config")
@patch.object(UploadService, "file_2_df")
def test_upload_file_success(
    mock_file_2_df,
    mock_load_config,
    mock_oci,
    mock_oracle,
    mock_open,
    mock_json_load,
    mock_remove,
):
    service = UploadService()

    df = pd.DataFrame({
        "Date": ["01-01-2024"],
        "Year": [2024],
        "Quarter": [1],
        "Month": [1],
        "Developer": ["ABC"],
        "Project": ["P1"],
        "City": ["Mumbai"],
        "PropertyType": ["Flat"],
        "UnitType": ["2BHK"],
        "ConstructionStatus": ["Ready"],
        "UnitsSold": [10],
        "AvgSqFt": [1200],
        "PricePerSqFt": [10000],
        "SalesValueCr": [12.0],
        "GrossProfitCr": [3.0],
    })

    mock_file_2_df.return_value = df
    mock_oci.return_value.get_pdf_files_from_bucket.return_value = "/tmp/data.csv"

    oracle_instance = MagicMock()
    oracle_instance.execute_query_df.return_value = df.head(1)
    mock_oracle.return_value = oracle_instance

    mock_json_load.return_value = {"meta": "data"}

    response = service.upload_file("chat1", "data.csv")

    assert response["status"] == 1
    assert "successfully" in response["message"].lower()
    oracle_instance.execute_multiple_non_query.assert_called_once()


@patch("app.services.upload_service.os.remove")
@patch.object(UploadService, "file_2_df")
@patch("app.services.upload_service.OCIObjectStorageClient")
@patch("app.services.upload_service.OracleADBClient")
@patch("app.services.upload_service.load_adw_config")
def test_upload_file_missing_columns(
    mock_load_config,
    mock_oracle,
    mock_oci,
    mock_file_2_df,
    mock_remove,
):
    service = UploadService()

    df = pd.DataFrame({
        "Date": ["01-01-2024"],
        "Year": [2024],
    })

    mock_file_2_df.return_value = df
    mock_oci.return_value.get_pdf_files_from_bucket.return_value = "/tmp/data.csv"

    response = service.upload_file("chat1", "data.csv")

    assert response["status"] == 0
    assert "Mismatch in structure" in response["error"]
    assert "Missing cols" in response["error"]

    mock_remove.assert_called_once_with("/tmp/data.csv")


@patch("app.services.upload_service.os.remove")
@patch.object(UploadService, "file_2_df")
@patch("app.services.upload_service.OCIObjectStorageClient")
@patch("app.services.upload_service.OracleADBClient")
@patch("app.services.upload_service.load_adw_config")
def test_upload_file_extra_columns(
    mock_load_config,
    mock_oracle,
    mock_oci,
    mock_file_2_df,
    mock_remove,
):
    service = UploadService()

    df = pd.DataFrame({
        "Date": ["01-01-2024"],
        "Year": [2024],
        "Quarter": [1],
        "Month": [1],
        "Developer": ["ABC"],
        "Project": ["P1"],
        "City": ["Mumbai"],
        "PropertyType": ["Flat"],
        "UnitType": ["2BHK"],
        "ConstructionStatus": ["Ready"],
        "UnitsSold": [10],
        "AvgSqFt": [1200],
        "PricePerSqFt": [10000],
        "SalesValueCr": [12.0],
        "GrossProfitCr": [3.0],
        "Unexpected": ["BAD"],
    })

    mock_file_2_df.return_value = df
    mock_oci.return_value.get_pdf_files_from_bucket.return_value = "/tmp/data.csv"

    response = service.upload_file("chat1", "data.csv")

    assert response["status"] == 0
    assert "Mismatch in structure" in response["error"]
    assert "extrac_cols" in response["error"]

    mock_remove.assert_called_once_with("/tmp/data.csv")

@patch.object(UploadService, "file_2_df", side_effect=Exception("Boom"))
@patch("app.services.upload_service.OCIObjectStorageClient")
@patch("app.services.upload_service.OracleADBClient")
@patch("app.services.upload_service.load_adw_config")
def test_upload_file_exception(
    mock_load_config,
    mock_oracle,
    mock_oci,
    mock_file_2_df,
):
    service = UploadService()

    mock_oci.return_value.get_pdf_files_from_bucket.return_value = "/tmp/data.csv"

    response = service.upload_file("chat1", "data.csv")

    assert response["status"] == 0
    assert "failed" in response["message"].lower()

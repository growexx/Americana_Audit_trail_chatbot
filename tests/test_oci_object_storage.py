import builtins
import pytest
from unittest.mock import MagicMock, patch

from code_modules.oci_object_storage import OCIObjectStorageClient


@patch("code_modules.oci_object_storage.oci.object_storage.ObjectStorageClient")
@patch("code_modules.oci_object_storage.configparser.ConfigParser")
def test_client_initialization(mock_config_parser, mock_oci_client):
    """Test OCI client initialization"""
    mock_config = MagicMock()
    mock_config.__getitem__.return_value = {
        "user": "user",
        "key_file": "key",
        "fingerprint": "fp",
        "tenancy": "tenancy"
    }

    mock_config_parser.return_value = mock_config

    client = OCIObjectStorageClient()

    assert client.client is not None
    mock_oci_client.assert_called_once()


@patch("code_modules.oci_object_storage.open", new_callable=MagicMock)
@patch("code_modules.oci_object_storage.oci.object_storage.ObjectStorageClient")
@patch("code_modules.oci_object_storage.configparser.ConfigParser")
def test_get_pdf_files_from_bucket(
    mock_config_parser,
    mock_oci_client,
    mock_open
):
    """Test downloading a file from OCI bucket"""
    mock_config = MagicMock()
    mock_config.__getitem__.return_value = {
        "user": "user",
        "key_file": "key",
        "fingerprint": "fp",
        "tenancy": "tenancy"
    }
    mock_config_parser.return_value = mock_config

    # Mock OCI get_object response
    mock_stream = MagicMock()
    mock_stream.stream.return_value = [b"test-data"]
    mock_response = MagicMock()
    mock_response.data.raw = mock_stream

    mock_client_instance = MagicMock()
    mock_client_instance.get_object.return_value = mock_response
    mock_oci_client.return_value = mock_client_instance

    client = OCIObjectStorageClient()
    result = client.get_pdf_files_from_bucket("test.pdf")

    assert result == "test.pdf"
    mock_client_instance.get_object.assert_called_once()


@patch("code_modules.oci_object_storage.open", new_callable=MagicMock)
@patch("code_modules.oci_object_storage.oci.object_storage.ObjectStorageClient")
@patch("code_modules.oci_object_storage.configparser.ConfigParser")
def test_upload_file_to_bucket(
    mock_config_parser,
    mock_oci_client,
    mock_open
):
    """Test uploading a file to OCI bucket"""
    mock_config = MagicMock()
    mock_config.__getitem__.return_value = {
        "user": "user",
        "key_file": "key",
        "fingerprint": "fp",
        "tenancy": "tenancy"
    }
    mock_config_parser.return_value = mock_config

    mock_client_instance = MagicMock()
    mock_client_instance.put_object.return_value = "UPLOAD_OK"
    mock_oci_client.return_value = mock_client_instance

    client = OCIObjectStorageClient()
    result = client.upload_file_to_bucket("file.csv", "csv/")

    assert result == "csv/file.csv"
    mock_client_instance.put_object.assert_called_once()

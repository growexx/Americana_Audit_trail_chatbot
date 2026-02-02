import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from code_modules.oracle_adb_handler import OracleADBClient


@pytest.fixture
def mock_config():
    """Mock ADWConfig object"""
    config = MagicMock()
    config.username = "user"
    config.password = "password"
    config.dsn = "dsn"
    config.config_dir = "/config"
    config.wallet_loc = "/wallet"
    config.wallet_pw = "wallet_pw"
    return config


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_query_df_success(mock_connect, mock_config):
    """Test successful SELECT query returning DataFrame"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    expected_df = pd.DataFrame({"id": [1, 2]})

    with patch("code_modules.oracle_adb_handler.pd.read_sql", return_value=expected_df):
        client = OracleADBClient(mock_config)
        result = client.execute_query_df("SELECT * FROM table")

    assert result.equals(expected_df)
    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_query_df_failure(mock_connect, mock_config):
    """Test SELECT query failure raises exception and closes connection"""
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    with patch(
        "code_modules.oracle_adb_handler.pd.read_sql",
        side_effect=Exception("DB error"),
    ):
        client = OracleADBClient(mock_config)
        with pytest.raises(Exception):
            client.execute_query_df("SELECT * FROM table")

    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_multiple_non_query_with_params(mock_connect, mock_config):
    """Test executemany with params commits successfully"""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    client = OracleADBClient(mock_config)
    client.execute_multiple_non_query(
        "INSERT INTO t VALUES (:1)",
        params=[(1,), (2,)],
    )

    mock_cursor.executemany.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_multiple_non_query_without_params(mock_connect, mock_config):
    """Test executemany without params commits successfully"""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    client = OracleADBClient(mock_config)
    client.execute_multiple_non_query("DELETE FROM table")

    mock_cursor.executemany.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_multiple_non_query_failure(mock_connect, mock_config):
    """Test rollback is called on failure"""
    mock_cursor = MagicMock()
    mock_cursor.executemany.side_effect = Exception("Insert failed")

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    client = OracleADBClient(mock_config)
    with pytest.raises(Exception):
        client.execute_multiple_non_query("INSERT INTO t VALUES (:1)", [(1,)])

    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_single_non_query_success(mock_connect, mock_config):
    """Test execute single non-query commits successfully"""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    client = OracleADBClient(mock_config)
    client.execute_single_non_query(
        "UPDATE t SET c = :1",
        params=(1,),
    )

    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


@patch("code_modules.oracle_adb_handler.oracledb.connect")
def test_execute_single_non_query_failure(mock_connect, mock_config):
    """Test rollback on execute failure"""
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = Exception("Update failed")

    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_connect.return_value = mock_conn

    client = OracleADBClient(mock_config)
    with pytest.raises(Exception):
        client.execute_single_non_query("UPDATE t SET c = 1")

    mock_conn.rollback.assert_called_once()
    mock_conn.close.assert_called_once()

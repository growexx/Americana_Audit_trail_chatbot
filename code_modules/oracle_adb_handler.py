"""
This is Oracle adb handler module, All code interacting with
Oracle Autonoumous Database Warehourse.
"""
from typing import Optional, Sequence

import pandas as pd
import oracledb

from config_loader import ADWConfig , load_adw_config

class OracleADBClient:
    """
    Client for interacting with Oracle Autonomous Database (ADB).

    Responsibilities:
    - Execute SELECT queries and return results as pandas DataFrames
    - Execute DML/DDL statements with no returned output
    """

    def __init__(self, config: ADWConfig):
        self._config = config

    def _get_connection(self):
        return oracledb.connect(
            user=self._config.username,
            password=self._config.password,
            dsn=self._config.dsn,
            config_dir=self._config.config_dir,
            wallet_location=self._config.wallet_loc,
            wallet_password=self._config.wallet_pw,
        )

    def execute_query_df(
        self,
        query: str,
        params: Optional[Sequence] = None,
    ):
        """
        Execute a SELECT query and return results as a pandas DataFrame.

        Args:
            query (str): SQL SELECT query
            params (Optional[Sequence]): Query bind parameters

        Returns:
            pd.DataFrame: Query result
        """
        print("Executing SELECT query")

        conn = self._get_connection()
        try:
            df = pd.read_sql(query, conn, params=params)
            print("Query executed successfully, rows fetched: %d", len(df))
            return df
        except Exception:
            print("Failed to execute SELECT query")
            raise
        finally:
            conn.close()

    def execute_multiple_non_query(
        self,
        query: str,
        params: Optional[Sequence] = None,
    ) -> None:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE, DDL).

        Args:
            query (str): SQL statement
            params (Optional[Sequence]): Query bind parameters
        """
        print("Executing non-SELECT query")

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.executemany(query, params)
                else:
                    cursor.executemany(query)

            conn.commit()
            print("Query committed successfully")
        except Exception:
            conn.rollback()
            print("Failed to execute non-SELECT query")
            raise
        finally:
            conn.close()

    def execute_single_non_query(
        self,
        query: str,
        params: Optional[Sequence] = None,
    ) -> None:
        """
        Execute a non-SELECT query (INSERT, UPDATE, DELETE, DDL).

        Args:
            query (str): SQL statement
            params (Optional[Sequence]): Query bind parameters
        """
        print("Executing non-SELECT query")

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

            conn.commit()
            print("Query committed successfully")
        except Exception:
            conn.rollback()
            print("Failed to execute non-SELECT query")
            raise
        finally:
            conn.close()

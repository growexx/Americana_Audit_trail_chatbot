"""
Sql Quuery Loader For American Audit Trails.

This module is responsible for loading all sql queries used by the system,

Queries are loaded as variables
using runtime inputs such as user queries, SQL statements, and data samples.
"""

import re
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

class SqlQueryLoader:
    """
    Loads SQL Queries
    """
    @staticmethod
    def load_chat_history_by_id(chat_id:str):
        """
        Load query
        """
        return {
            "load_chat_history": f"""SELECT CHAT_ID, MESSAGE_NO , DBMS_LOB.SUBSTR(MESSAGE, 4000, 1) AS message, ROLE 
                FROM CHAT_MESSAGES WHERE chat_id = '{chat_id}'"""
        }

    @staticmethod
    def load_user_chats_previews(user_id:str):
        """
        Load query
        """
        return {
            "load_chats_preview": f""" SELECT * FROM USER_CHATS  WHERE user_id = '{user_id}'"""
        }

    @staticmethod
    def delete_chat_queries(chat_id:str):
        """
        Load query
        """
        return {
            "delete_chat_history": f"""DELETE FROM CHAT_MESSAGES WHERE chat_id = '{chat_id}'""",
            "delete_chat_preview": f"""DELETE FROM USER_CHATS WHERE chat_id = '{chat_id}'"""
        }

    @staticmethod
    def insert_user_chat(user_id: str, chat_id: str, title: str):
        """
        Return SQL to insert a new user chat preview row.
        """
        # Escape single quotes in title
        title_safe = title.replace("'", "''")
        return {
            "insert_user_chat": (
                f"INSERT INTO USER_CHATS (USER_ID, CHAT_ID, CHAT_TITLE) "
                f"VALUES ('{user_id}', '{chat_id}', '{title_safe}')"
            )
        }

    @staticmethod
    def insert_chat_message(chat_id: str, message_no: int, role: str, message: str):
        """
        Return SQL to insert a single chat message row into CHAT_MESSAGES.
        """
        # Escape single quotes in message body
        msg_safe = message.replace("'", "''")
        return {
            "insert_chat_message": (
                f"INSERT INTO CHAT_MESSAGES (CHAT_ID, MESSAGE_NO, MESSAGE, ROLE) "
                f"VALUES ('{chat_id}', {message_no}, '{msg_safe}', '{role}')"
            )
        }



"""
Chat service module for the Real Estate Conversational BI application.

This module orchestrates the end-to-end workflow for handling user chat inquiries:
- Applies guardrails to validate relevance
- Converts natural language to SQL using LLMs
- Executes SQL queries against Oracle Autonomous Database
- Processes results for analysis or raw data download
- Generates plots and uploads artifacts to OCI Object Storage

It integrates database access, LLM inference, prompt generation,
response parsing, and visualization utilities.
"""

from code_modules.oracle_adb_handler import OracleADBClient
from config_loader import load_adw_config
from code_modules.prompt_generator import PromptGenerator
from code_modules.llm_response_extractor import LLMResponseExtractor
from code_modules.sql_queries_loader import SqlQueryLoader
from code_modules.oracle_genai_handler import create_llm_client
import json
import pandas as pd
from code_modules.oracle_adb_handler import OracleADBClient
import io
from datetime import datetime
import traceback
import matplotlib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

from app.services.title import create_new_chat_title

matplotlib.use("Agg")

def prepare_metadata_string(tables):
    """
    Prepare metadata string for prompt construction."""
    metadata_string = ""

    for table in tables:
        file_name = f"table_metadata/{table.lower()}.json"
        with open(file_name, "r") as f:
            metadata = json.load(f)
        metadata_string += f"\n\n### TABLE: {table.upper()}\n"
        metadata_string += json.dumps(metadata, indent=2)
    return metadata_string

def check_if_df_all_null_or_zero(df: pd.DataFrame) -> bool:
    """
    Check whether all values in the dataframe are null or zero.

    Args:
        df (pd.DataFrame): Dataframe to evaluate.

    Returns:
        bool: True if all values are null or zero, otherwise False.
    """
    return bool(((df.isna()) | (df == 0)).all().all())


class JumpToFinally(Exception):
    """Custom exception just to jump to finally"""
    pass

class ChatService:
    """
    Service layer responsible for handling chat-based BI queries.

    This class manages:
    - Guardrail validation
    - Natural language to SQL conversion
    - Database query execution
    - Context-aware LLM inference
    - Response formatting for analysis or raw data scenarios
    """
    def __init__(self):
        """
        Initialize ChatService with LLM clients, prompt generator,
        and response extractor.
        """
        self.llm_inference_client = create_llm_client()
        self.llm_response_extractor = LLMResponseExtractor()
        self.prompt_generator_client = PromptGenerator()
        self.adb_client = OracleADBClient(load_adw_config())
        self.sql_loader = SqlQueryLoader()

    def handle_inquiry(self,user_id:str ,chat_id: str , user_message: str,app_state):
        """
        Handle a user chat inquiry end-to-end.

        This method:
        - Validates the query using guardrails
        - Converts the query to SQL
        - Executes the SQL against Oracle ADB
        - Maintains chat context and history
        - Generates either analytical insights or raw data output

        Args:
            chat_id (str): Unique identifier for the chat session.
            user_message (str): User's natural language query.
            app_state: FastAPI application state for maintaining context.

        Returns:
            dict: Structured response containing query results,
                status, and optional artifacts.
        """

        final_response ={"chat_id":chat_id,"llm_response":"Failed due to error",
                        "user_query":"","status":0 }
        try:

            print(50*'1',"Guard Rail Call",50*'1')

            relevant, tables = self.guard_rail(user_message)
            print(f"relevance={relevant}, tables={tables}") 

            if relevant != "yes":
                print("Query rejected by guardrail")
                final_response = {
                    "chat_id":chat_id,
                    "llm_response":"Query rejected by guardrail due to irrelevance.",
                    "user_query":user_message,
                    "status":2
                }
                raise JumpToFinally()

            print(50*'2',"Text-2-SQL Call",50*'2')


            last_sql_query_of_chat = app_state.last_sql_queries
            last_sql_query = last_sql_query_of_chat.get(chat_id,None)

            if last_sql_query is None:
                get_last_sql_query = self.sql_loader.last_sql_query_for_chat(chat_id)['last_sql_query_of_chat']
                last_sql_df = self.adb_client.execute_query_df(get_last_sql_query)
                print(last_sql_df)
                if not last_sql_df.empty:
                    last_sql_query =  last_sql_df.iloc[0]["MESSAGE"]

            print(f"Last sql query for chat_id {chat_id} is {last_sql_query}.")
            
            metadata_string = prepare_metadata_string(tables)
            metadata_string = re.sub(r'\s+', ' ', metadata_string).strip()
            print(f"meta data is {metadata_string[:300]} ..")
            sql_query, error_status = self.text_2_sql(user_message, tables, last_sql_query, metadata_string)

            print(f"SQL Generation error status is {error_status}")
            if error_status == 1 or error_status=="1":
                final_response = {
                    "chat_id":chat_id,
                    "llm_response":"Failed to generate query as it extends beyond the scope of Data , Please try with another query.",
                    "user_query":user_message,
                    "status":2
                }
                raise JumpToFinally()

            print(f"Extracted Sql is {sql_query}")

            print(50*'3',"SQL Against Database",50*'3')

            selected_df = self.adb_client.execute_query_df(sql_query)
            df_is_empty = check_if_df_all_null_or_zero(selected_df)

            if selected_df.empty or df_is_empty:
                print(f"Df is empty :{df_is_empty}")
                final_response = {
                    "chat_id":chat_id,
                    "llm_response":"No data found for following search",
                    "sql_query":sql_query,
                    "status":3
                }
                raise JumpToFinally()

            print(selected_df)
            print(selected_df.shape)
            print(50*'4',"Context management",50*'4')
            
            app_state.last_sql_queries[chat_id] = sql_query

            chat_histories = app_state.chat_history
            users_chats = app_state.user_chats
            
            current_user_chat_id = users_chats.get(user_id,None)

            user_df_for_chat_ids_query = self.sql_loader.load_user_chats_previews(user_id)['load_chats_preview']
            user_df_for_chat_ids = self.adb_client.execute_query_df(user_df_for_chat_ids_query)["CHAT_ID"].tolist()
            print(user_df_for_chat_ids)
            if chat_id not in user_df_for_chat_ids:
                print(f"New chat history for chat_id {chat_id} Developed")
                chat_histories[chat_id] = [{"role":"Assistant","message": self.prompt_generator_client.generate_main_prompt()}]
                create_new_chat_title(self,user_id,chat_id,user_message)
                users_chats[user_id] = chat_id
            else:
                print(f"Chat history for chat_id {chat_id} already exists")
                if chat_id != current_user_chat_id:
                    print(chat_id)
                    print(current_user_chat_id)
                    print("Chat history not in ram , need to load it")
                    self.load_chat_history(user_id,chat_id,app_state)
            
            chat_history =  chat_histories[chat_id]
            print(50*'5',"Main Call Inference",50*'5')
            
            assistant_prompt = self.prompt_generator_client.generate_assistant_prompt(sql_query, selected_df)

            chat_history.append({"role":"User","message":user_message})
            chat_history.append({"role":"Assistant","message":assistant_prompt})

            print(f"chat_history is {chat_history}")
            result = self.llm_inference_client.inference_from_chat_history(chat_history)
            chat_history.pop()
            chat_history.append({"role":"System","message":result})

            print(f"Result is {result}-----------------------------------------------------")
            self.llm_response_extractor.set_data(result)
            message = self.llm_response_extractor.get("message","")
            print(f"Main Call message is {message} ,")

            last_message_no_query = self.sql_loader.get_last_message_no(chat_id)['last_message_no']

            message_no_df = self.adb_client.execute_query_df(last_message_no_query)
            if message_no_df.empty:
                message_no = 1
            else:
                message_no = int(message_no_df.iloc[0]["MESSAGE_NO"])
                message_no= message_no+1
            insert_user_query =  self.sql_loader.insert_chat_history(chat_id,message_no,user_message,"user")
            print(insert_user_query)
            message_no = message_no+1
            self.adb_client.execute_single_non_query(insert_user_query['query'],insert_user_query['params'])
            insert_system_query =  self.sql_loader.insert_chat_history(chat_id,message_no,message,"system")
            print(insert_system_query)
            message_no = message_no+1
            self.adb_client.execute_single_non_query(insert_system_query['query'],insert_system_query['params'])
            insert_sql_query =  self.sql_loader.insert_chat_history(chat_id,message_no,sql_query,"SQL")
            print(insert_sql_query)
            self.adb_client.execute_single_non_query(insert_sql_query['query'],insert_sql_query['params'])

            print(50*'6',"Result Modification",50*'6')

            final_response = self.prepare_data_response(selected_df,sql_query,message)
        
        except Exception as e:
            traceback.print_exc()
            if not final_response:
                final_response =  {
                        "chat_id":chat_id,
                        "llm_response":f"Failed due to error {e}",
                        "user_query":e,
                        "status":0
                    }
        finally:
            print("Inference code is executed.")
        return final_response

    @staticmethod
    def prepare_data_response(selected_df,sql_query,message):
        """
        Prepare a raw data response for download.

        Saves query results as a CSV file, uploads it to object storage,
        and returns the downloadable file reference.

        Args:
            selected_df (pd.DataFrame): Query result dataframe.
            scenario (str): Identified scenario type.
            sql_query (str): Executed SQL query.
            message (str): LLM-generated explanation.

        Returns:
            dict: Raw data response payload.
        """
        safe_df = selected_df.head(10).replace([np.nan, np.inf, -np.inf], None)
        results_df = safe_df.to_dict(orient="records")
        return {
            "results_df":results_df,
            "llm_response":message,
            "sql_query":sql_query,
            "status":1
        }

    def guard_rail(self, user_message,):
        """
        Apply guardrail validation to a user query.

        Uses an LLM-based guardrail to determine whether the query
        is relevant and which database tables are required.

        Args:
            user_message (str): User's natural language query.

        Returns:
            tuple[str, list[str]]: Relevance flag and list of related tables.
        """

        user_guard_rail_message= f"User message is '{user_message}'"
        guard_rail_result = self.prompt_generator_client.guardrail_check_inference_call(self.llm_inference_client, user_guard_rail_message)

        self.llm_response_extractor.set_data(guard_rail_result)
        print(f"Guard rail result is {guard_rail_result}")

        relevant, tables = self.llm_response_extractor.get_many(["relevant_question", "tables_related"],{"relevant_question":"no","tables_related":[]})

        return relevant , tables

    def text_2_sql(self, user_message, tables, last_sql_query, metadata):
        """
        Convert a natural language query into SQL.

        Uses LLM inference with schema metadata and conversation context
        to generate a SQL query and determine execution scenario.

        Args:
            user_message (str): User's natural language query.
            tables (list[str]): Relevant database tables.
            last_sql_query (str | None): Previously executed SQL for context.
            metadata (str): Table metadata for prompt construction.

        Returns:
            tuple[str, str, int]: SQL query, scenario, and error status.
        """

        metadata_string = prepare_metadata_string(tables)

        sql_prompt = self.prompt_generator_client.generate_sql_prompt(user_message, metadata_string, last_sql_query)
        llm_sql_raw = self.llm_inference_client.inference_single_input(user_message, sql_prompt)
        self.llm_response_extractor.set_data(llm_sql_raw)
        sql_query, error_status = self.llm_response_extractor.get_many(
            ["sql_query", "error_status"],
            {"sql_query":"","error_status":0}
        )
        return sql_query, error_status

    def load_chat_history(self,user_id,chat_id,app_state):
        """
        This message is used to load chat history for a specific user and chat id
        Assumed that Message not called frequently, So we fetch it from db and not store in state.
        """
        try:

            get_chat_history_query = self.sql_loader.load_chat_history_by_id(chat_id)['load_chat_history']
            chat_history_df = self.adb_client.execute_query_df(get_chat_history_query)

            # No Chat history Found exit 
            if chat_history_df.empty:
                return {
                    "chat_id":chat_id,
                    "systen_message":"No chat history found for chat_id {chat_id}",
                    "status":0
                }

            print("Now we manage and Chat history and store it in state")
            
            # Load it in load in chat_history
            main_prompt = self.prompt_generator_client.generate_main_prompt()
            chat_history_state = [{"role":"Assistant","message":main_prompt }]
            chat_history_front_end = []


            print(chat_history_df)
            chat_history_df = chat_history_df.sort_values(by=["MESSAGE_NO"])

            for _, row in chat_history_df.sort_values("MESSAGE_NO").iterrows():

                if row["ROLE"].upper() == "SQL":
                    # Replace message with executed SQL result
                    message_content_df = self.adb_client.execute_query_df(row["MESSAGE"])
                    message_content = message_content_df.to_dict(orient="records")
                    chat_history_front_end.append({
                        "role": row["ROLE"],
                        "message": message_content
                    })
                else:
                    chat_history_front_end.append({
                        "role": row["ROLE"],
                        "message": row["MESSAGE"]
                    })
                    chat_history_state.append({
                        "role": row["ROLE"],
                        "message": row["MESSAGE"]
                    })

            print(f"Chat history front end is {chat_history_front_end}")
            print(f"Chat history state is {chat_history_state}")
            
            last_sql_message = chat_history_df[chat_history_df['ROLE'].str.upper() == 'SQL'].tail(1)

            app_state.chat_history[chat_id] = chat_history_state
            if not last_sql_message.empty:
                app_state.last_sql_queries[chat_id] = last_sql_message['MESSAGE'].values[0]
            else:
                app_state.last_sql_queries[chat_id] = None   # or skip

            app_state.user_chats[user_id] = chat_id

            return {
                    "chat_id":chat_id,
                    "system_response":chat_history_front_end,
                    "status":1
                }
        except Exception as e:
            traceback.print_exc()
            return {
                "chat_id":chat_id,
                "error_message":e,
                "status":0
            }

    def load_user_chats_previews(self,user_id,app_state):
        """
        This message is used to load chat history for a specific user
        """
        try:
            get_chat_history_query = self.sql_loader.load_user_chats_previews(user_id)['load_chats_preview']
            print(get_chat_history_query)

            chat_history_df =self.adb_client.execute_query_df(get_chat_history_query)
            print(chat_history_df)

            results = chat_history_df.to_dict(orient="records")
            return {
                    "previous_chat_previous":results,
                    "status":1
                }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }

    def chat_runtime_cleanup(self,user_id,app_state):
        try:
            print("Chat runtime cleanup")
            chat_histories = app_state.chat_history
            last_sql_queries = app_state.last_sql_query
            user_chats = app_state.user_chats

            user_chat_ids = user_chats.get(user_id)
            if user_chat_ids is None:
                return {
                    "status":0,
                    "message":f"No chat history found for user {user_id} in state"
                }

            n = len(user_chat_ids)
            print(user_chat_ids)

            for chat_id in user_chat_ids:
                del chat_histories[chat_id]
                del last_sql_queries[chat_id]
            
            return {
                "status":1,
                "message":f"Deleted {n} chat history for user {user_id}"
            }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }

    def delete_chat_history(self,user_id,chat_ids,app_state):
        try:
            for chat_id in chat_ids:
                delete_queries = self.sql_loader.delete_chat_queries(chat_id)
                self.adb_client.execute_single_non_query(delete_queries['delete_chat_history'])
                self.adb_client.execute_single_non_query(delete_queries['delete_chat_preview'])

                app_state.chat_history.pop(chat_id, None)
                app_state.last_sql_queries.pop(chat_id, None)

            return {
                "status":1,
                "message":f"Deleted {len(chat_ids)} chat history for user {user_id}"
            }
        except Exception as e:
            print(e)
            return {
                "error_message":e,
                "status":0
            }
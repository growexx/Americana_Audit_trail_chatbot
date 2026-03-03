"""
The following module is used to extarct json from LLM response
"""
import re
import json
from typing import Dict, Any
import traceback

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from:
    - pure JSON string
    - text containing JSON
    - ```json fenced blocks
    """
    try:
        text = text.strip()

        # Case 1: fenced ```json
        fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fenced:
            return json.loads(fenced.group(1))

        # Case 2: embedded JSON
        embedded = re.search(r"\{.*\}", text, re.DOTALL)
        if embedded:
            return json.loads(embedded.group(0))

        # Case 3: plain JSON
        result = json.loads(text)
    except Exception as e:
        traceback.print_exc()
        result = {"response":text,
                "error":str(e)}
    return result

class LLMResponseExtractor:
    """
    We Declare json data with set_data and get single or multiple keys
    """
    def set_data(self, text:str):
        self.data = extract_json(text)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def get_many(self, fields, defaults=None):
        defaults = defaults or {}
        return tuple(self.get(f, defaults.get(f)) for f in fields)


ap_column_prirority ={
    "BUSINESS_UNIT":1,
    "VENDOR_NAME":2,
    "INVOICE_NUM":3,
    "INVOICE_DATE":4,
    "INVOICE_AMOUNT":5,
    "INVOICE_CURRENCY_CODE":6,
    "WFAPPROVAL_STATUS":7,
    "PAYMENT_STATUS_FLAG":8,
    "INV_CREATOR_NAME":9,
    "APPROVER_FULL_NAME":10
}

gl_column_priority={
"LEDGER_NAME":1,
"PERIOD_NAME":2,
"JE_SOURCE":3,
"JE_CATEGORY":4,
"BATCH_NAME":5,
"JOURNAL_NAME":6,
"CURRENCY_CODE":7,
"POSTED_DATE":8,
"JOURNAL_DR":9,
"JOURNAL_CR":10,
"APPROVAL_STATUS_CODE":11,
"JOURNAL_STATUS":12
}

user_column_priority = {
"APPLICATION_ID":1,
"RESPONSIBILITY_NAME":2,
"USER_NAME":3,
"CREATION_DATE":4,
"LAST_UPDATE_DATE":5
}

def smart_reorder(query, columns):
    """
    Takes column names and query and rearranges them to fit the order
    """
    print(f"smart reorder start with input columns {columns}")
    query_lower = query.lower()

    if "ap_audit_data" in query_lower:
        table_column_priority = ap_column_prirority
        print('ap audit data reorder started')
    elif "gl_audit_data" in query_lower:
        print('gl audit data reorder started')
        table_column_priority = gl_column_priority
    elif "user_audit_data" in query_lower:
        print('user audit data reorder started')
        table_column_priority = user_column_priority
    else:
        table_column_priority = {}

    # Sort columns based on priority (unknown columns go last)
    ordered_columns = sorted(
        columns,
        key=lambda col: table_column_priority.get(col, 999)
    )
    print(f"Soted columns are {ordered_columns}")

    return ordered_columns

"""
The following module is used to extarct json from LLM response
"""
import re
import json
from typing import Dict, Any

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from:
    - pure JSON string
    - text containing JSON
    - ```json fenced blocks
    """
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
    return json.loads(text)

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


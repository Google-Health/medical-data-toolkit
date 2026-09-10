# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for LLM responses."""


import re


def extract_json_from_llm_response(raw_text: str) -> str:
  """Extracts ONLY the JSON object from the raw LLM response.

  Args:
    raw_text: The raw text response from the LLM.

  Returns:
    A string containing the extracted JSON object.

  Raises:
    ValueError: If valid JSON brackets {} were not found in the response.
  """
  # Priority 1: Check for explicit markdown code block
  match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
  if match:
    return match.group(1)

  # Fallback to broad bracket extraction
  start_idx = raw_text.find("{")
  end_idx = raw_text.rfind("}")

  if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
    return raw_text[start_idx : end_idx + 1]
  raise ValueError("Valid JSON brackets {} were not found in the response.")


def extract_json_list_from_llm_response(raw_text: str) -> str:
  """Extracts ONLY the JSON list from the raw LLM response.

  Args:
    raw_text: The raw text response from the LLM.

  Returns:
    A string containing the extracted JSON list or JSON object (as a fallback).

  Raises:
    ValueError: If valid JSON brackets [] or {} were not found in the response.
  """
  # Priority 1: Check for explicit markdown code block for list
  match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
  if match:
    return match.group(1)

  # Extract JSON array or object
  bracket_start = raw_text.find("[")
  brace_start = raw_text.find("{")

  if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
    end_idx = raw_text.rfind("]")
    if end_idx > bracket_start:
      return raw_text[bracket_start : end_idx + 1]
    raise ValueError(
        "Valid JSON closing bracket ] was not found in the response."
    )

  # Fallback to existing utility for objects
  return extract_json_from_llm_response(raw_text)


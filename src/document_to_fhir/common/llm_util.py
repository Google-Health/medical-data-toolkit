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


def extract_json_from_llm_response(raw_text: str) -> str:
  """Extracts ONLY the JSON object from the raw LLM response."""
  # Step 1: Find the index of the first opening brace
  start_idx = raw_text.find("{")
  # Step 2: Find the index of the last closing brace
  end_idx = raw_text.rfind("}")

  # Step 3: Slice the string to extract ONLY the JSON object
  if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
    return raw_text[start_idx : end_idx + 1]
  else:
    raise ValueError("Valid JSON brackets {} were not found in the response.")

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
"""Utility functions for string manipulations and loinc data handling."""

import collections
import json
import re

import pandas


class StringUtils:
  """Helper methods for string related functions."""

  @classmethod
  def get_similarity_ratio(cls, str1: str, str2: str) -> float:
    s1 = "".join(c.lower() for c in str(str1) if c.isalnum())
    s2 = "".join(c.lower() for c in str(str2) if c.isalnum())
    if not s1 or not s2:
      return 1.0
    c1 = collections.Counter(s1)
    c2 = collections.Counter(s2)
    all_chars = set(c1.keys()) | set(c2.keys())
    diff = sum(abs(c1[c] - c2[c]) for c in all_chars)
    denominator = min(len(s1), len(s2))
    return 1.0 if denominator == 0 else diff / denominator

  @classmethod
  def compute_anagram_signature(cls, text: str) -> str:
    """Computes the anagram signature: sorted alphanumeric characters, lowercased."""
    if not text:
      return ""
    cleaned = sorted(char.lower() for char in text if char.isalnum())
    return "".join(cleaned)

  @classmethod
  def extract_json_list_from_llm_response(cls, text: str) -> list[str]:
    """Extracts a JSON list of strings from LLM response."""
    if not text:
      return []
    # Find JSON list using regex
    match = re.search(r"\[\s*.*?\s*\]", text, re.DOTALL)
    if not match:
      return []
    try:
      res = json.loads(match.group(0))
      return res if isinstance(res, list) else []
    except json.JSONDecodeError:
      return []


def read_loinc_kb(
    file_path: str, max_rank: int, limit: int = 0
) -> pandas.DataFrame:
  """Reads the LOINC knowledge base from a CSV file.

  The function reads the LOINC data from the provided file path.
  It filters the DataFrame to include only rows with 'CLASSTYPE' in [1, 2]
  and 'COMMON_TEST_RANK' greater than 0 and less than or equal to `max_rank`.
  Optionally, it limits the number of rows based on `limit`.

  Args:
    file_path: Path to the LOINC CSV file.
    max_rank: The maximum value of COMMON_TEST_RANK to include.
    limit: If > 0, the maximum number of rows to return.

  Returns:
    A pandas DataFrame containing the filtered LOINC data, or None if the
    file is not found.
  """
  df_loinc = pandas.read_csv(file_path, low_memory=False, encoding="utf-8-sig")

  if (
      df_loinc is not None
      and "CLASSTYPE" in df_loinc.columns
      and "COMMON_TEST_RANK" in df_loinc.columns
  ):
    df_loinc = df_loinc[df_loinc["CLASSTYPE"].isin([1, 2])]
    df_loinc = df_loinc[
        pandas.to_numeric(df_loinc["COMMON_TEST_RANK"], errors="coerce") > 0
    ]
    df_loinc = df_loinc[
        pandas.to_numeric(df_loinc["COMMON_TEST_RANK"], errors="coerce")
        <= max_rank
    ]
    if limit > 0 and len(df_loinc) > limit:
      df_loinc = df_loinc.head(limit)
  print(f"Read {len(df_loinc)} LOINC rows.")
  return df_loinc

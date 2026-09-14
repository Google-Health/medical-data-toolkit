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

import json
import logging
import re
from typing import Any

import pandas
import stringdist

from src.document_to_fhir.common import file_util

file_exists = file_util.file_exists
open_file = file_util.open_file
read_text_file = file_util.read_text_file
read_prompt = file_util.read_prompt


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
NORMALIZED_LEVENSHTEIN_THRESHOLD = 0.15


def read_csv(filepath: str, **kwargs: Any) -> pandas.DataFrame:
  """Reads a CSV into a pandas DataFrame using central open_file."""
  with open_file(filepath, "r") as f:
    return pandas.read_csv(f, **kwargs)


class StringUtils:
  """Helper methods for string related functions."""

  @classmethod
  def levenshtein_distance(cls, s1: str, s2: str) -> int:
    """Computes the Levenshtein distance between two strings."""
    return stringdist.levenshtein(s1, s2)

  @classmethod
  def normalized_levenshtein_distance(cls, s1: str, s2: str) -> float:
    """Computes the normalized Levenshtein distance between two strings (0.0 to 1.0)."""
    if not s1 and not s2:
      return 0.0
    if not s1 or not s2:
      return 1.0
    return stringdist.levenshtein_norm(s1, s2)

  @classmethod
  def tokenize(cls, text: str) -> list[str]:
    """Extracts lowercased alphanumeric tokens from text."""
    if not text:
      return []
    return _TOKEN_PATTERN.findall(str(text).lower())

  @classmethod
  def compute_fuzzy_token_overlap(
      cls,
      query: str,
      target: str,
      threshold: float = NORMALIZED_LEVENSHTEIN_THRESHOLD,
  ) -> float:
    """Computes directional fuzzy token overlap of query in target (0.0 to 1.0).

    Extracts alphanumeric tokens from query and calculates the fraction that
    match any token in target (either exactly, or within a normalized
    Levenshtein distance threshold).
    """
    q_tokens = cls.tokenize(query)
    if not q_tokens:
      return 0.0
    t_tokens = set(cls.tokenize(target))
    if not t_tokens:
      return 0.0

    matched_count = 0
    for q in q_tokens:
      if q in t_tokens:
        matched_count += 1
        continue
      # Note: Early-terminates upon finding the first valid match <= threshold.
      # Target tokens are not consumed (many-to-one matching). Potential future
      # improvement: 1-to-1 bipartite matching or greedy best-match assignment
      # if duplicate query tokens ever require distinct target counterparts.
      if any(
          abs(len(q) - len(t)) / max(len(q), len(t)) <= threshold
          and cls.normalized_levenshtein_distance(q, t) <= threshold
          for t in t_tokens
      ):
        matched_count += 1
    return matched_count / len(q_tokens)

  @classmethod
  def get_similarity_ratio(cls, str1: str, str2: str) -> float:
    """Returns distance ratio between str1 (query) and str2 (target).

    Defined as 1.0 - fuzzy_token_overlap(str1, str2) so 0.0 indicates a perfect
    match.
    """
    return 1.0 - cls.compute_fuzzy_token_overlap(str1, str2)

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
    file_path: str,
    max_rank: int,
    limit: int = 0,
    additional_loinc_codes: list[str] | None = None,
) -> pandas.DataFrame:
  """Reads the LOINC knowledge base from a CSV file.

  The function reads the LOINC data from the provided file path.
  It filters the DataFrame to include only rows with 'CLASSTYPE' in [1, 2].
  If `additional_loinc_codes` is provided, it processes codes up to max_rank
  AND the specific codes in the file (union).

  Args:
    file_path: Path to the LOINC CSV file.
    max_rank: The maximum value of COMMON_TEST_RANK to include.
    limit: If > 0, the maximum number of rows to return.
    additional_loinc_codes: Optional list of additional LOINC codes to include
      irrespective of the max_rank.

  Returns:
    A pandas DataFrame containing the filtered LOINC data.
  """
  df_loinc = read_csv(file_path, low_memory=False, encoding="utf-8-sig")

  if "CLASSTYPE" in df_loinc.columns:
    df_loinc = df_loinc[df_loinc["CLASSTYPE"].isin([1, 2])]

  if "LOINC_NUM" in df_loinc.columns:
    df_loinc = df_loinc.drop_duplicates(subset=["LOINC_NUM"])

  rank_mask = pandas.Series(False, index=df_loinc.index)
  if "COMMON_TEST_RANK" in df_loinc.columns:
    common_test_rank = pandas.to_numeric(
        df_loinc["COMMON_TEST_RANK"], errors="coerce"
    )
    rank_mask = (common_test_rank.fillna(0) > 0) & (
        common_test_rank.fillna(float("inf")) <= max_rank
    )

  if additional_loinc_codes is not None:
    code_mask = df_loinc["LOINC_NUM"].isin(additional_loinc_codes)
    df_loinc = df_loinc[rank_mask | code_mask]
  elif "COMMON_TEST_RANK" in df_loinc.columns:
    df_loinc = df_loinc[rank_mask]

  if limit > 0 and len(df_loinc) > limit:
    df_loinc = df_loinc.head(limit)

  logging.info("Read %d LOINC rows.", len(df_loinc))
  return df_loinc

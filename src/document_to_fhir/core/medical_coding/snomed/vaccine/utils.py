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
"""Utility functions for SNOMED vaccine mapping."""

from collections.abc import Sequence
import re

import pandas as pd

_VACCINE_KEYWORDS_PATTERN = (
    r"\b(vaccines?|vax(?:es)?|antigens?|virus|bacteria|"
    r"paediatric|pediatric|adult|ped)\b"
)
_VACCINE_KEYWORDS_RE = re.compile(
    _VACCINE_KEYWORDS_PATTERN, flags=re.IGNORECASE
)


def remove_vaccine_keywords(text: str) -> str:
  """Removes common vaccine, pathogen, and age qualifier words from text.

  Args:
    text: Input vaccine description text.

  Returns:
    Cleaned text with matching keyword tokens removed and spacing normalized.
  """
  cleaned = _VACCINE_KEYWORDS_RE.sub("", text)

  # Clean up resulting extra spaces around punctuation and words.
  cleaned = re.sub(r"\s+", " ", cleaned)
  cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
  cleaned = re.sub(r"(^,\s*|,\s*$)", "", cleaned)
  return cleaned.strip()


def parse_and_normalize_vaccines(text: str) -> list[str]:
  """Extracts and normalizes vaccine names from text into normalized tokens.

  Removes vaccine keywords, expands parenthesized subterms into comma-separated
  terms, removes hyphens, and lowercases each token.

  Args:
    text: Input vaccine string.

  Returns:
    List of normalized vaccine name tokens.
  """
  cleaned = remove_vaccine_keywords(text)
  cleaned = re.sub(r"\s*\(([^)]+)\)", r", \1", cleaned)

  normalized_list = []
  for item in cleaned.split(", "):
    stripped = item.strip()
    # Replace all hyphens and spaces with empty strings, and convert to
    # lowercase. This is done after splitting to avoid merging terms. The
    # reason for doing this to handle combination vaccines that are typically
    # written with a hyphen or space between them, e.g. "MMR-Vax" or
    # "Varicella Zoster". By removing the hyphen and space, we can normalize
    # these to "mmrvax" and "varicellazoster" respectively which allows us to
    # match them to the similarly normalized SNOMED terms.
    normalized = stripped.replace("-", "").replace(" ", "").lower()
    normalized_terms = normalized.split("/")
    normalized_list.extend(normalized_terms)

  return normalized_list


def get_top_parent_code(
    df: pd.DataFrame,
    excluded_codes: Sequence[str | int],
    concept_col: str = "snomed_code",
    parent_col: str = "parent_snomed_code",
) -> str | None:
  """Traverses a parent concept chain upward and returns the top parent code.

  Args:
    df: DataFrame containing concept codes and parent concept codes.
    excluded_codes: Concept codes to stop before or exclude from traversal.
    concept_col: Column name for concept identifiers.
    parent_col: Column name for parent concept identifiers.

  Returns:
    The top non-excluded parent code in the ancestor chain, or None.
  """
  excluded_set = {str(code) for code in excluded_codes}
  parent_map = dict(
      zip(df[concept_col].astype(str), df[parent_col].astype(str))
  )

  # Pick any starting concept that isn't already excluded.
  valid_starts = [code for code in parent_map if code not in excluded_set]
  if not valid_starts:
    return None

  current = valid_starts[0]
  visited = set()

  # Traverse upward until reaching an excluded code or the top of the chain.
  while current in parent_map and current not in visited:
    visited.add(current)
    parent = parent_map[current]

    # Stop right before stepping into an excluded code or missing parent.
    if (
        parent in excluded_set
        or pd.isna(parent)
        or not parent
        or parent in ("None", "nan", "N/A", "")
    ):
      break

    current = parent

  return current

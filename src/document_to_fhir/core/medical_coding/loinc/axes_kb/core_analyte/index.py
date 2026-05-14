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
"""Provides an index for LOINC Analyte harmonization.

This module defines the `AnalytesIndex` class, which allows for loading,
indexing, and searching LOINC data, including handling of core analytes
and their synonyms.
"""

import ast
import collections
import os
import re
from typing import Any, Iterable

import pandas

from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import normalize
from src.document_to_fhir.core.medical_coding.loinc.common import schema


_SEMANTIC_SYMBOLS_RE = re.compile(r"\s*([/:%#+])\s*")
_NON_SEMANTIC_PUNCTUATION_RE = re.compile(r"[,\.\-]")


def _file_exists(csv_path: str) -> bool:
  """Returns True if the file exists, False otherwise."""
  return os.path.exists(csv_path)


class AnalytesIndex:
  """A readable, API-driven index for LOINC Analyte harmonization."""

  def __init__(self):
    self._signature_to_records_index: dict[str, list[schema.LoincRow]] = (
        collections.defaultdict(list)
    )
    self._loinc_to_core_analyte_map: dict[str, str] = {}
    self._core_analyte_to_synonyms_map: dict[str, set[str]] = (
        collections.defaultdict(set)
    )

  @classmethod
  def from_csv(cls, csv_path: str) -> "AnalytesIndex":
    """Loads and builds an AnalytesIndex from a CSV file.

    Args:
      csv_path: The path to the input CSV file.

    Returns:
      An instance of AnalytesIndex populated with data from the CSV.

    Raises:
      FileNotFoundError: If the specified csv_path does not exist.
    """
    if not _file_exists(csv_path):
      raise FileNotFoundError(f"File not found: {csv_path}")

    print(f"Loading Index from {csv_path}...")
    df = pandas.read_csv(csv_path)

    index = cls()
    # Convert 'synonyms' column upfront if it exists.
    if config.SYNONYMS_KEY in df.columns:
      df[config.SYNONYMS_KEY] = df[config.SYNONYMS_KEY].apply(
          index._parse_synonyms
      )

    index.load_data(df.to_dict("records"))
    print(f"Loaded {len(index._signature_to_records_index)} terms.")
    return index

  def load_data(self, records: Iterable[dict[str, Any]]):
    """Loads and indexes LOINC analyte data from a list of records.

    This method processes the input records in two steps:
    1.  Harvest core analytes and their associated synonyms.
    2.  Build lookup tables and search index.

    Args:
      records: An iterable of dictionaries representing LOINC records.
    """
    records_with_core = self._harvest_synonyms(records)
    self._build_search_index(records_with_core)

    print("✅ Indexing Complete.")
    print(f"   - Loaded {len(records_with_core)} LOINC rows.")
    print(
        "   - Harmonized synonyms for"
        f" {len(self._core_analyte_to_synonyms_map)} unique Core Analytes."
    )

  def search_by_analyte(self, analyte_name: str) -> list[schema.LoincRow]:
    """Searches the index for records matching the given analyte name."""
    signature = self._create_analyte_signature(analyte_name)
    return self._signature_to_records_index.get(signature, [])

  def get_core_analyte(self, loinc_number: str) -> str | None:
    """Returns the core analyte mapped to a specific LOINC number."""
    return self._loinc_to_core_analyte_map.get(str(loinc_number))

  def get_all_analytes(self, loinc_number: str) -> set[str]:
    """Returns the core analyte and all its synonyms for a LOINC number."""
    core = self.get_core_analyte(loinc_number)
    if not core:
      return set()
    synonyms = self._core_analyte_to_synonyms_map.get(core, set())
    return {core}.union(synonyms)

  def _harvest_synonyms(
      self, records: Iterable[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Pass 1: Collect core analytes and their synonyms."""
    valid_records = []
    for record in records:
      core_analyte = record.get(config.CORE_ANALYTE_KEY)
      if not core_analyte or pandas.isna(core_analyte):
        print(f"Core is empty: {record}")
        continue

      record[config.CORE_ANALYTE_KEY] = str(core_analyte)
      valid_records.append(record)

      synonyms = self._parse_synonyms(record.get(config.SYNONYMS_KEY))
      self._core_analyte_to_synonyms_map[str(core_analyte)].update(synonyms)

    return valid_records

  def _build_search_index(self, records: list[dict[str, Any]]):
    """Pass 2: Populate lookup tables and mapping signatures to records."""
    loinc_to_record = {}
    signature_to_loincs = collections.defaultdict(set)

    for record in records:
      core = record[config.CORE_ANALYTE_KEY]
      loinc_num = str(record.get(config.LOINC_NUM_KEY))
      loinc_to_record[loinc_num] = schema.LoincRow.model_validate(record)

      self._loinc_to_core_analyte_map[loinc_num] = core
      all_terms = self._core_analyte_to_synonyms_map[core] | {core}

      for term in all_terms:
        if term:
          signature = self._create_analyte_signature(term)
          signature_to_loincs[signature].add(loinc_num)

    # Finalize search index by mapping signatures to record objects.
    for signature, loinc_nums in signature_to_loincs.items():
      self._signature_to_records_index[signature] = [
          loinc_to_record[num] for num in loinc_nums
      ]

  def _create_analyte_signature(self, analyte_name: str) -> str:
    """Standardizes clinical lab text into a normalized signature.

    This method transforms clinical lab text into a robust signature suitable
    for database mapping (like LOINC) by performing several normalization steps.

    Args:
      analyte_name: The input clinical lab text.

    Returns:
      A normalized signature string.
    """
    if not analyte_name:
      return ""

    # 1. Normalize analyte using domain-specific rules.
    analyte_name = normalize.normalize_analyte(analyte_name)

    # 2. Convert to lowercase.
    analyte_name = str(analyte_name).lower()

    # 3. Normalize Semantic Symbols
    # Remove spaces around critical symbols (/, :, %, #, +) so they bind to the
    # analyte. Examples: "CD4 / CD8" -> "cd4/cd8", "Na +" -> "na+"
    analyte_name = _SEMANTIC_SYMBOLS_RE.sub(r"\1", analyte_name)

    # 4. Strip Non-Semantic Punctuation
    # Replace commas, periods, and hyphens with spaces.
    # This solves the "1,25-Dihydroxy" vs "1 25 Dihydroxy" problem.
    analyte_name = _NON_SEMANTIC_PUNCTUATION_RE.sub(" ", analyte_name)

    # 5. Tokenize, de-duplicate, and sort for a stable signature.
    signature_tokens = sorted(list(set(analyte_name.split())))

    # 6. Join with an underscore for readability and uniqueness.
    return "_".join(signature_tokens)

  def _parse_synonyms(self, raw_synonyms: Any) -> list[str]:
    """Parses raw synonyms into a list of strings."""
    if isinstance(raw_synonyms, list):
      return [str(s) for s in raw_synonyms]
    if isinstance(raw_synonyms, str):
      try:
        parsed = ast.literal_eval(raw_synonyms)
        return [str(s) for s in parsed] if isinstance(parsed, list) else []
      except (ValueError, SyntaxError):
        return []
    return []

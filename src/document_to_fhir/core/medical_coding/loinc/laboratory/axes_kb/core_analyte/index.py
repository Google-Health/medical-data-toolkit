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
import logging
from typing import Any, Iterable

import pandas

from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.common import schema
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import normalize


def _file_exists(csv_path: str) -> bool:
  """Returns True if the file exists, False otherwise."""
  return utils.file_exists(csv_path)


class AnalytesIndex:
  """A readable, API-driven index for LOINC Analyte harmonization."""

  def __init__(self):
    self._strict_signature_to_records: dict[str, list[schema.LoincRow]] = (
        collections.defaultdict(list)
    )
    self._lenient_signature_to_records: dict[str, list[schema.LoincRow]] = (
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

    logging.info("Loading Index from %s...", csv_path)
    df = pandas.read_csv(csv_path)

    index = cls()
    if config.STRICT_SYNONYMS_KEY not in df.columns:
      raise ValueError(
          f"Missing required column '{config.STRICT_SYNONYMS_KEY}' in"
          f" {csv_path}. Please use an updated Core Analyte KB CSV generated"
          " with the latest core_analyte_builder_main."
      )
    df[config.STRICT_SYNONYMS_KEY] = df[config.STRICT_SYNONYMS_KEY].apply(
        index._parse_synonyms
    )
    if config.LENIENT_SYNONYMS_KEY in df.columns:
      df[config.LENIENT_SYNONYMS_KEY] = df[config.LENIENT_SYNONYMS_KEY].apply(
          index._parse_synonyms
      )

    index.load_data(df.to_dict("records"))
    logging.info(
        "Loaded %d strict terms, %d lenient terms.",
        len(index._strict_signature_to_records),
        len(index._lenient_signature_to_records),
    )
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

    logging.info("✅ Indexing Complete.")
    logging.info("   - Loaded %d LOINC rows.", len(records_with_core))
    logging.info(
        "   - Harmonized synonyms for %d unique Core Analytes.",
        len(self._core_analyte_to_synonyms_map),
    )

  def search_by_analyte(
      self, analyte_name: str, allow_lenient: bool = True
  ) -> list[schema.LoincRow]:
    """Searches the index for records matching the given analyte name.

    First checks the strict index. If no candidates are found and allow_lenient
    is True, falls back to the lenient index.

    Args:
      analyte_name: The analyte name or synonym to search for.
      allow_lenient: Whether to fall back to the lenient index if no strict
        match is found.

    Returns:
      A list of matching LoincRow records.
    """
    signature = normalize.create_analyte_signature(analyte_name)
    results = self._strict_signature_to_records.get(signature, [])
    if not results and allow_lenient:
      results = self._lenient_signature_to_records.get(signature, [])
    return results

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
        logging.warning("Core is empty for record: %s", record)
        continue

      record[config.CORE_ANALYTE_KEY] = str(core_analyte)
      for syn_key in (
          config.STRICT_SYNONYMS_KEY,
          config.LENIENT_SYNONYMS_KEY,
      ):
        if syn_key in record:
          record[syn_key] = self._parse_synonyms(record[syn_key])
      valid_records.append(record)

      strict_syns = record.get(config.STRICT_SYNONYMS_KEY) or []
      self._core_analyte_to_synonyms_map[str(core_analyte)].update(strict_syns)

    return valid_records

  def _build_search_index(self, records: list[dict[str, Any]]):
    """Pass 2: Populate lookup tables and mapping signatures to records."""
    loinc_to_record = {}
    strict_sig_to_loincs = collections.defaultdict(set)
    lenient_sig_to_loincs = collections.defaultdict(set)

    for record in records:
      core = record[config.CORE_ANALYTE_KEY]
      loinc_num = str(record.get(config.LOINC_NUM_KEY))
      loinc_to_record[loinc_num] = schema.LoincRow.model_validate(record)

      self._loinc_to_core_analyte_map[loinc_num] = core

      # 1. Index strict terms (core analyte + harmonized strict synonyms for
      # this exact core analyte).
      strict_terms = {core} | self._core_analyte_to_synonyms_map[core]
      for term in strict_terms:
        if term:
          signature = normalize.create_analyte_signature(term)
          strict_sig_to_loincs[signature].add(loinc_num)

      # 2. Index lenient terms (broad parent substances, categories)
      lenient_terms = set(record.get(config.LENIENT_SYNONYMS_KEY) or [])
      for term in lenient_terms:
        if term:
          signature = normalize.create_analyte_signature(term)
          if loinc_num not in strict_sig_to_loincs.get(signature, set()):
            lenient_sig_to_loincs[signature].add(loinc_num)

    # Finalize search index by mapping signatures to record objects.
    for signature, loinc_nums in strict_sig_to_loincs.items():
      self._strict_signature_to_records[signature] = [
          loinc_to_record[num] for num in loinc_nums
      ]
    for signature, loinc_nums in lenient_sig_to_loincs.items():
      self._lenient_signature_to_records[signature] = [
          loinc_to_record[num] for num in loinc_nums
      ]

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

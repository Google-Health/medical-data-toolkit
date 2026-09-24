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
"""Builds a Knowledge Base for LOINC properties using LLM.

This module contains the `PropertyKBBuilder` class, which is responsible for
generating synonyms for LOINC properties by querying a Gemini model and saving
the results to a CSV file.
"""

import concurrent.futures
import json
from typing import Any

from absl import logging
import pandas
import tqdm.auto

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import prompt as prompt_lib


class PropertyKBBuilder:
  """Builds the Property Synonym Knowledge Base using LLM."""

  def __init__(
      self,
      df_loinc: pandas.DataFrame,
      client: model_client.LLMClient,
      workers: int = 10,
      max_attempts: int = 4,
  ):
    self.df_loinc = df_loinc
    self.client = client
    self.workers = workers
    self.max_attempts = max_attempts

  def _process_single_unit(self, unit: str) -> tuple[str, list[str]]:
    """Queries Gemini for synonyms of a single unit."""
    prompt = prompt_lib.PROMPT_TEMPLATE.format(unit=unit)
    for attempt in range(max(1, self.max_attempts)):
      try:
        response = self.client.generate_content(contents=[prompt])
        synonyms = json.loads(
            llm_util.extract_json_list_from_llm_response(response.text)
        )
        if unit not in synonyms:
          synonyms.append(unit)
        return unit, synonyms
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Attempt %d/%d failed for unit %s: %s",
            attempt + 1,
            self.max_attempts,
            unit,
            e,
        )
    logging.error(
        "Failed to process unit %s after %d attempts.", unit, self.max_attempts
    )
    return unit, []

  def _extract_unit_to_properties(self) -> dict[str, set[str]]:
    """Extracts mapping from UCUM unit to canonical LOINC properties."""
    df_filtered = self.df_loinc[[config.PROPERTY_KEY, config.UNIT_KEY]].dropna(
        subset=[config.PROPERTY_KEY]
    )
    df_filtered[config.UNIT_KEY] = df_filtered[config.UNIT_KEY].fillna("")

    unit_to_properties = {}
    for _, row in df_filtered.iterrows():
      prop = row[config.PROPERTY_KEY]
      units_str = row[config.UNIT_KEY]
      units = [u.strip() for u in units_str.split(";")] if units_str else [""]
      for u in units:
        if u not in unit_to_properties:
          unit_to_properties[u] = set()
        unit_to_properties[u].add(prop)
    return unit_to_properties

  def _load_existing_records(
      self, save_csv_path: str
  ) -> tuple[list[dict[str, Any]], set[str]]:
    """Loads existing records for resuming KB build from a CSV file."""
    try:
      with utils.open_file(save_csv_path, "r") as f:
        df_existing = pandas.read_csv(f)
    except (
        OSError,
        IOError,
        pandas.errors.EmptyDataError,
        pandas.errors.ParserError,
    ) as e:
      logging.warning(
          "Failed to read existing CSV for resume: %s. Starting from scratch.",
          e,
      )
      return [], set()

    existing_records = df_existing.to_dict(orient="records")
    processed_synonyms = set(df_existing["synonym"].dropna())
    logging.info(
        "Found %d existing records in %s",
        len(existing_records),
        save_csv_path,
    )
    return existing_records, processed_synonyms

  def _parse_units_in_parallel(
      self, units_to_process: list[str]
  ) -> dict[str, list[str]]:
    """Queries LLM in parallel to generate synonyms for unique units."""
    if not units_to_process:
      return {}

    unit_to_synonyms = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=self.workers
    ) as executor:
      futures = {
          executor.submit(self._process_single_unit, unit): unit
          for unit in units_to_process
      }
      for future in tqdm.auto.tqdm(
          concurrent.futures.as_completed(futures),
          total=len(units_to_process),
          desc="Building Property KB",
          unit="unit",
      ):
        unit, synonyms = future.result()
        unit_to_synonyms[unit] = synonyms

    return unit_to_synonyms

  def _save_mapping_to_csv(
      self,
      unit_to_properties: dict[str, set[str]],
      unit_to_synonyms: dict[str, list[str]],
      existing_records: list[dict[str, Any]],
      save_csv_path: str,
  ):
    """Constructs final mapping and saves results to CSV."""
    results = []
    for unit, properties in unit_to_properties.items():
      if not unit:
        synonyms = [""]
      elif unit == "-":
        synonyms = ["", "-"]
      else:
        synonyms = unit_to_synonyms.get(unit, [unit])
      for prop in properties:
        for syn in synonyms:
          results.append({"property": prop, "synonym": syn})

    if existing_records:
      results = existing_records + results

    df_results = pandas.DataFrame(results).drop_duplicates()
    logging.info(
        "Saving mapping (%d rows) to %s", len(df_results), save_csv_path
    )
    with utils.open_file(save_csv_path, "w") as f:
      df_results.to_csv(f, index=False)

    logging.info("Finished! Saved mapping to %s", save_csv_path)

  def build_kb(self, save_csv_path: str, resume: bool = False):
    """Extracts property synonyms, parses them via LLM, and saves to CSV."""
    if config.PROPERTY_KEY not in self.df_loinc.columns:
      raise ValueError(f"Column {config.PROPERTY_KEY} not found in DataFrame")
    if config.UNIT_KEY not in self.df_loinc.columns:
      raise ValueError(f"Column {config.UNIT_KEY} not found in DataFrame")

    logging.info(
        "Building Property KB by mapping EXAMPLE_UCUM_UNITS to PROPERTY "
        "via LLM..."
    )

    unit_to_properties = self._extract_unit_to_properties()
    unique_units = list(unit_to_properties.keys())
    unique_units_to_process = [u for u in unique_units if u]
    logging.info(
        "Found %d unique UCUM units (non-empty: %d) to parse.",
        len(unique_units),
        len(unique_units_to_process),
    )

    existing_records = []
    if resume and save_csv_path and utils.file_exists(save_csv_path):
      existing_records, processed_synonyms = self._load_existing_records(
          save_csv_path
      )
      if processed_synonyms:
        remaining_units = [
            u for u in unique_units_to_process if u not in processed_synonyms
        ]
        logging.info(
            "Filtered unique units from %d to %d (resuming)",
            len(unique_units_to_process),
            len(remaining_units),
        )
        unique_units_to_process = remaining_units

    unit_to_synonyms = self._parse_units_in_parallel(unique_units_to_process)
    self._save_mapping_to_csv(
        unit_to_properties=unit_to_properties,
        unit_to_synonyms=unit_to_synonyms,
        existing_records=existing_records,
        save_csv_path=save_csv_path,
    )

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
"""Builds a knowledge base of analytes from LOINC data using an LLM.

This module contains functions and classes to read LOINC data, enrich it
with analyte information (like core analytes and synonyms) using an
`AnalyteGenerator`, and build an index for efficient searching.
"""

import concurrent.futures
import json
import logging
from typing import Any

import pandas
import tqdm.auto

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import index
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import prompt

_PROMPT_COLS = [
    config.LOINC_NUM_KEY,
    config.COMPONENT_KEY,
    config.PROPERTY_KEY,
    config.TIME_ASPCT_KEY,
    config.SYSTEM_KEY,
    config.SCALE_TYPE_KEY,
    config.METHOD_TYP_KEY,
    config.RELATEDNAMES_KEY,
    config.SHORTNAME_KEY,
    config.EXAMPLE_UNITS_KEY,
    config.UNIT_KEY,
    config.DISPLAY_NAME_KEY,
    config.LONG_COMMON_NAME_KEY,
]


def _clean_row(row: pandas.Series) -> dict[str, Any]:
  return {k: v if pandas.notna(v) else None for k, v in row.to_dict().items()}


class AnalyteKBBuilder:
  """Builds a knowledge base of analytes by enriching LOINC data.

  This class takes a DataFrame of LOINC codes and uses an `AnalyteGenerator`
  to generate additional analyte information, such as core analytes and
  synonyms. It can save the enriched data and build an `AnalyteIndex`
  for efficient lookups.
  """

  def __init__(
      self,
      df_loinc: pandas.DataFrame,
      client_inst: model_client.LLMClient,
      workers: int,
      batch_size: int = 5,
      max_attempts: int = 4,
  ):
    if df_loinc is None:
      raise ValueError("df_loinc cannot be None")
    if client_inst is None:
      raise ValueError("client cannot be None")

    self.df_loinc = df_loinc
    self.client = client_inst
    self.workers = workers
    # Note: batch_size should be set according to the limitations of the LLM
    # client being used.
    self.batch_size = batch_size
    self.max_attempts = max_attempts

  def build_kb(
      self, save_csv_path: str = "", resume: bool = False
  ) -> index.AnalytesIndex:
    """Builds the Analyte Knowledge Base by enriching LOINC data.

    This method orchestrates the process of generating enriched analyte
    information using the provided LLM client. It first generates
    the data, optionally saves the results to a CSV, and then builds
    an `AnalyteIndex` for efficient lookups.

    Args:
      save_csv_path: An optional path to save the enriched LOINC records as a
        CSV file. If None, the records are not saved.
      resume: If True, resumes building by loading existing records from
        save_csv_path and only processing remaining LOINCs.

    Returns:
      An `AnalyteIndex` instance populated with the enriched analyte data.
    """
    logging.info(
        "--- Starting KB Build (Workers: %d, LOINCS: %d) ---",
        self.workers,
        len(self.df_loinc),
    )

    original_total = len(self.df_loinc)
    existing_records = []
    already_processed_count = 0

    file_exists = utils.file_exists(save_csv_path) if save_csv_path else False

    if resume and save_csv_path and file_exists:
      try:
        with utils.open_file(save_csv_path, "r") as f:
          df_existing = pandas.read_csv(f)

        existing_records = df_existing.to_dict(orient="records")
        processed_loincs = set(df_existing[config.LOINC_NUM_KEY].dropna())
        logging.info(
            "Found %d existing records in %s",
            len(existing_records),
            save_csv_path,
        )

        # Filter df_loinc to only include non-processed LOINCs
        self.df_loinc = self.df_loinc[
            ~self.df_loinc[config.LOINC_NUM_KEY].isin(processed_loincs)
        ]
        already_processed_count = original_total - len(self.df_loinc)
        logging.info(
            "Filtered input from %d to %d records (resuming)",
            original_total,
            len(self.df_loinc),
        )
      except Exception:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Failed to read existing CSV for resume. Starting from scratch.",
            exc_info=True,
        )

    if resume:
      logging.info(
          "Remaining LOINC codes to be processed: %d", len(self.df_loinc)
      )

    # 1. Generate (The heavy lifting) with Interrupt Safety
    new_records, errors = self._generate_enriched_records()

    if errors:
      logging.error("%d rows failed processing.", len(errors))
      logging.error("Sample errors:")
      for err in errors[:5]:
        logging.error(
            "    - LOINC %s: %s",
            err.get("LOINC_NUM", "N/A"),
            err.get("error"),
        )
      if len(errors) > 5:
        logging.error("    ...")

    records = existing_records + new_records

    if not records:
      logging.warning("No valid records were processed.")
      return index.AnalytesIndex()

    if not new_records:
      logging.info(
          "No new records processed, but returning index with existing %d"
          " records.",
          len(existing_records),
      )

    # 2. Save Valid Records
    if save_csv_path:
      try:
        with utils.open_file(save_csv_path, "w") as f:
          pandas.DataFrame(records).to_csv(f, index=False)
        logging.info(
            "Enriched KB (%d records) saved to: %s",
            len(records),
            save_csv_path,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Failed to save CSV cache: %s", e)

    # 3. Populate Index
    logging.info("[Phase 2] Building Search Index...")
    analyte_index = index.AnalytesIndex()
    analyte_index.load_data(records)

    successful_count = already_processed_count + len(new_records)
    logging.info(
        "--- Build Complete. Processed %d / %d rows successfully. ---",
        successful_count,
        original_total,
    )
    return analyte_index

  def _generate_enriched_records(
      self,
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generates enriched analyte records using the LLM client.

    This method processes batches of the `df_loinc` DataFrame in parallel
    using a ThreadPoolExecutor. It calls `_process_batch` for each batch of
    LOINC entries to generate enriched analyte information, including core
    analytes and synonyms. It includes error handling and graceful interruption.

    Returns:
      A tuple containing two lists:
        - enriched_results: A list of dictionaries, where each dictionary
          represents a successfully enriched LOINC record.
        - error_records: A list of dictionaries for records that failed
          processing, containing error details.
    """
    logging.info("[Phase 1] Generating Analyte Data via LLM...")
    enriched_results = []

    # Use a dictionary to keep track of remaining work by LOINC_NUM
    remaining_work = {
        row[config.LOINC_NUM_KEY]: row for _, row in self.df_loinc.iterrows()
    }
    latest_errors = {}

    max_attempts = self.max_attempts
    initial_batch_size = self.batch_size

    for attempt in range(max_attempts):
      if not remaining_work:
        break

      # Use configured batch size for first attempt, 1 for retries
      current_batch_size = initial_batch_size if attempt == 0 else 1

      if attempt > 0:
        logging.info(
            "Retrying failed records. Attempt %d/%d. Records to process: %d"
            " (Batch Size: %d)",
            attempt,
            max_attempts - 1,
            len(remaining_work),
            current_batch_size,
        )

      process_args = list(remaining_work.values())
      batches = [
          process_args[i : i + current_batch_size]
          for i in range(0, len(process_args), current_batch_size)
      ]
      total_tasks = len(batches)

      executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
      future_to_batch = {
          executor.submit(self._process_batch, batch): batch
          for batch in batches
      }

      try:
        with tqdm.auto.tqdm(
            total=total_tasks,
            desc=(
                "Building Core Analyte KB (Attempt"
                f" {attempt + 1}/{max_attempts})"
            ),
            unit="batch",
        ) as pbar:
          for future in concurrent.futures.as_completed(future_to_batch):
            try:
              results = future.result()
              for result in results:
                if not result:
                  continue
                loinc_num = result.get(config.LOINC_NUM_KEY)
                if "error" not in result:
                  enriched_results.append(result)
                  if loinc_num in remaining_work:
                    del remaining_work[loinc_num]
                  if loinc_num in latest_errors:
                    del latest_errors[loinc_num]
                else:
                  if loinc_num:
                    latest_errors[loinc_num] = result
              pbar.update(1)
            except Exception as e:  # pylint: disable=broad-exception-caught
              batch = future_to_batch[future]
              for row in batch:
                loinc_num = row.get(config.LOINC_NUM_KEY, "N/A")
                logging.error(
                    "Unexpected error processing LOINC %s: %s",
                    loinc_num,
                    e,
                )
                error_record = {
                    "error": repr(e),
                    config.LOINC_NUM_KEY: loinc_num,
                }
                if loinc_num != "N/A":
                  latest_errors[loinc_num] = error_record
              pbar.update(1)

      except KeyboardInterrupt:
        logging.warning("User interrupt detected. Stopping gracefully...")
        logging.warning("Captured %d records so far.", len(enriched_results))
        for future in future_to_batch:
          future.cancel()
        executor.shutdown(wait=False)
        return enriched_results, list(latest_errors.values())
      finally:
        executor.shutdown(wait=True)

    return enriched_results, list(latest_errors.values())

  def _process_batch(self, batch: list[pandas.Series]) -> list[dict[str, Any]]:
    """Processes a batch of LOINC rows to generate enriched analyte information.

    This method uses the LLM client to enrich a batch of LOINC data,
    extracting or generating a core analyte, synonyms, and other key fields.
    It handles potential errors during the generation process.

    Args:
      batch: A list of pandas Series representing rows from the LOINC DataFrame.

    Returns:
      A list of dictionaries containing the enriched analyte information. If an
      error occurs, the dictionary will contain an "error" key with details.
    """

    loinc_records = [_clean_row(row) for row in batch]
    try:
      analyte_infos = self._process_analyte_batch(loinc_records)

      if not isinstance(analyte_infos, list):
        analyte_infos = [analyte_infos]

      # Map LLM outputs by their LOINC_NUM
      analyte_map = {
          info.get(config.LOINC_NUM_KEY): info for info in analyte_infos
      }

      results = []
      for record in loinc_records:
        loinc_num = record.get(config.LOINC_NUM_KEY)
        analyte_info = analyte_map.get(loinc_num)

        if not analyte_info:
          results.append({
              "error": "Missing or reordered result from LLM",
              config.LOINC_NUM_KEY: loinc_num,
          })
          continue

        core = analyte_info.get(config.CORE_ANALYTE_KEY)

        if not core or core == "ERROR":
          results.append({
              "error": "Core is empty or ERROR",
              "error_reasoning": analyte_info.get("analysis_reasoning", "N/A"),
              config.LOINC_NUM_KEY: record.get(config.LOINC_NUM_KEY),
          })
        else:
          strict_synonyms = analyte_info.get(config.STRICT_SYNONYMS_KEY, [])
          lenient_synonyms = analyte_info.get(config.LENIENT_SYNONYMS_KEY, [])
          results.append({
              # LLM Derived Data
              config.CORE_ANALYTE_KEY: core,
              config.STRICT_SYNONYMS_KEY: strict_synonyms,
              config.LENIENT_SYNONYMS_KEY: lenient_synonyms,
              # Original Metadata (Needed for filtering in the Query Engine)
              config.LOINC_NUM_KEY: record.get(config.LOINC_NUM_KEY),
              config.COMPONENT_KEY: record.get(config.COMPONENT_KEY),
              config.TIME_ASPCT_KEY: record.get(config.TIME_ASPCT_KEY),
              config.SCALE_TYPE_KEY: record.get(config.SCALE_TYPE_KEY),
              config.CLASS_TYPE_KEY: record.get(config.CLASS_TYPE_KEY),
              config.RANK_KEY: record.get(config.RANK_KEY),
              config.SYSTEM_KEY: record.get(config.SYSTEM_KEY),
              config.PROPERTY_KEY: record.get(config.PROPERTY_KEY),
              config.METHOD_TYP_KEY: record.get(config.METHOD_TYP_KEY),
              config.LONG_COMMON_NAME_KEY: record.get(
                  config.LONG_COMMON_NAME_KEY
              ),
              config.UNIT_KEY: record.get(config.UNIT_KEY),
          })
      return results
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Catching broad Exception to ensure that any failure in processing a
      # batch does not halt the entire multi-threaded generation process.
      # Errors are logged in the returned dictionaries.
      return [
          {
              "error": repr(e),
              config.LOINC_NUM_KEY: record.get(config.LOINC_NUM_KEY),
          }
          for record in loinc_records
      ]

  def _process_analyte_batch(
      self, loinc_records: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Queries the LLM for a batch of analyte rows."""
    filtered_records = [
        {k: record[k] for k in _PROMPT_COLS if k in record}
        for record in loinc_records
    ]
    input_data = json.dumps(filtered_records, indent=2)
    prompt_text = prompt.PROMPT_ANALYTE_INFO_TEMPLATE.format(
        input_data=input_data
    )
    response = self.client.generate_content([prompt_text])
    json_string = llm_util.extract_json_list_from_llm_response(response.text)
    return json.loads(json_string)

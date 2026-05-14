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
from typing import Any

import pandas
import tqdm.auto

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import index
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import prompt


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
  ):
    if df_loinc is None:
      raise ValueError("df_loinc cannot be None")
    if client_inst is None:
      raise ValueError("client cannot be None")

    self.df_loinc = df_loinc
    self.client = client_inst
    self.workers = workers

  def build_kb(self, save_csv_path: str = "") -> index.AnalytesIndex:
    """Builds the Analyte Knowledge Base by enriching LOINC data.

    This method orchestrates the process of generating enriched analyte
    information using the provided LLM client. It first generates
    the data, optionally saves the results to a CSV, and then builds
    an `AnalyteIndex` for efficient lookups.

    Args:
      save_csv_path: An optional path to save the enriched LOINC records as a
        CSV file. If None, the records are not saved.

    Returns:
      An `AnalyteIndex` instance populated with the enriched analyte data.
    """
    print(
        f"--- Starting KB Build (Workers: {self.workers}, LOINCS:"
        f" {len(self.df_loinc)}) ---"
    )

    # 1. Generate (The heavy lifting) with Interrupt Safety
    records, errors = self._generate_enriched_records()

    if errors:
      print(f"  [Info] {len(errors)} rows failed processing.")
      print("  [Debug] Sample errors:")
      for i, err in enumerate(errors[:5]):
        print(f"    - LOINC {err.get('LOINC_NUM', 'N/A')}: {err.get('error')}")
        if i == 4 and len(errors) > 5:
          print("    ...")

    if not records:
      print("  [Warning] No valid records were processed.")
      return index.AnalytesIndex()

    # 2. Save Valid Records
    if save_csv_path:
      try:
        pandas.DataFrame(records).to_csv(save_csv_path, index=False)
        print(
            f"  [Checkpoint] Enriched KB ({len(records)} records) saved to:"
            f" {save_csv_path}"
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"  [Warning] Failed to save CSV cache: {e}")

    # 3. Populate Index
    print("  [Phase 2] Building Search Index...")
    analyte_index = index.AnalytesIndex()
    analyte_index.load_data(records)

    print(f"--- Build Complete. Indexed {len(records)} records. ---")
    return analyte_index

  def _generate_enriched_records(
      self,
  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generates enriched analyte records using the LLM client.

    This method processes each row of the `df_loinc` DataFrame in parallel
    using a ThreadPoolExecutor. It calls `_process_single_row` for each LOINC
    entry to generate enriched analyte information, including core analytes
    and synonyms. It includes error handling and graceful interruption.

    Returns:
      A tuple containing two lists:
        - enriched_results: A list of dictionaries, where each dictionary
          represents a successfully enriched LOINC record.
        - error_records: A list of dictionaries for records that failed
          processing, containing error details.
    """
    print("  [Phase 1] Generating Analyte Data via LLM...")
    enriched_results = []
    error_records = []
    process_args = [(idx, row) for idx, row in self.df_loinc.iterrows()]
    total_tasks = len(process_args)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
    future_to_row = {
        executor.submit(self._process_single_row, row): idx
        for idx, row in process_args
    }

    try:
      with tqdm.auto.tqdm(
          total=total_tasks, desc="Processing LOINC Rows", unit="row"
      ) as pbar:
        for future in concurrent.futures.as_completed(future_to_row):
          try:
            result = future.result()
            if result and "error" not in result:
              enriched_results.append(result)
            elif result:
              error_records.append(result)
            pbar.update(1)
          except Exception as e:  # pylint: disable=broad-exception-caught
            row_idx = future_to_row[future]
            loinc_num = self.df_loinc.iloc[row_idx].get(
                config.LOINC_NUM_KEY, "N/A"
            )
            print(
                "  [Row Error] Unexpected error processing LOINC"
                f" {loinc_num}: {e}"
            )
            error_records.append(
                {"error": str(e), config.LOINC_NUM_KEY: loinc_num}
            )
            pbar.update(1)
    except KeyboardInterrupt:
      print("\n\n" + "!" * 60)
      print("  [USER INTERRUPT DETECTED] Stopping gracefully...")
      print(f"  Captured {len(enriched_results)} records so far.")
      print("!" * 60 + "\n")
      for f in future_to_row:
        f.cancel()
      executor.shutdown(wait=False)
      return enriched_results, error_records
    finally:
      executor.shutdown(wait=True)

    return enriched_results, error_records

  def _process_single_row(self, row) -> dict[str, Any]:
    """Processes a single LOINC row to generate enriched analyte information.

    This method uses the LLM client to enrich a single row of LOINC data,
    extracting or generating a core analyte, synonyms, and other key fields.
    It handles potential errors during the generation process.

    Args:
      row: A pandas Series representing a single row from the LOINC DataFrame.

    Returns:
      A dictionary containing the enriched analyte information. If an error
      occurs, the dictionary will contain an "error" key with details.
    """
    row_dict = {
        k: v if pandas.notna(v) else None for k, v in row.to_dict().items()
    }
    try:
      analyte_info = self._process_single_analyte(row_dict)
      core = analyte_info.get(config.CORE_ANALYTE_KEY)

      if not core or core == "ERROR":
        return {
            "error": "Core is empty or ERROR",
            "error_reasoning": analyte_info.get("analysis_reasoning", "N/A"),
            config.LOINC_NUM_KEY: row_dict.get(config.LOINC_NUM_KEY),
        }

      return {
          # LLM Derived Data
          config.CORE_ANALYTE_KEY: core,
          config.SYNONYMS_KEY: analyte_info.get(config.SYNONYMS_KEY, []),
          # Original Metadata (Needed for filtering in the Query Engine)
          config.LOINC_NUM_KEY: row_dict.get(config.LOINC_NUM_KEY),
          config.COMPONENT_KEY: row_dict.get(config.COMPONENT_KEY),
          config.TIME_ASPCT_KEY: row_dict.get(config.TIME_ASPCT_KEY),
          config.SCALE_TYPE_KEY: row_dict.get(config.SCALE_TYPE_KEY),
          config.CLASS_TYPE_KEY: row_dict.get(config.CLASS_TYPE_KEY),
          config.RANK_KEY: row_dict.get(config.RANK_KEY),
          config.SYSTEM_KEY: row_dict.get(config.SYSTEM_KEY),
          config.PROPERTY_KEY: row_dict.get(config.PROPERTY_KEY),
          config.METHOD_TYP_KEY: row_dict.get(config.METHOD_TYP_KEY),
          config.LONG_COMMON_NAME_KEY: row_dict.get(
              config.LONG_COMMON_NAME_KEY
          ),
          config.UNIT_KEY: row_dict.get(config.UNIT_KEY),
      }
    except Exception as e:  # pylint: disable=broad-exception-caught
      # Catching broad Exception to ensure that any failure in processing a
      # single row does not halt the entire multi-threaded generation process.
      # Errors are logged in the returned dictionary.
      return {
          "error": str(e),
          config.LOINC_NUM_KEY: row_dict.get(config.LOINC_NUM_KEY),
      }

  def _process_single_analyte(self, row_dict: dict[str, Any]) -> dict[str, Any]:
    """Queries the LLM for a single analyte row."""
    input_data = json.dumps(row_dict, indent=2)
    prompt_text = prompt.PROMPT_ANALYTE_INFO_TEMPLATE.format(
        input_data=input_data
    )
    response = self.client.generate_content([prompt_text])
    json_string = llm_util.extract_json_from_llm_response(response.text)
    return json.loads(json_string)

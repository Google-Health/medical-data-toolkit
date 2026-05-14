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

from absl import logging
import pandas
import tqdm.auto

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.property import prompt as prompt_lib
from src.document_to_fhir.core.medical_coding.loinc.common import utils


class PropertyKBBuilder:
  """Builds the Property Synonym Knowledge Base."""

  def __init__(
      self,
      df_loinc: pandas.DataFrame,
      client: model_client.LLMClient,
      workers: int,
  ):
    self.df_loinc = df_loinc
    self.client = client
    self.workers = workers

  def build_kb(self, save_csv_path: str):
    """Generates property synonyms and saves to CSV."""
    if config.PROPERTY_KEY not in self.df_loinc.columns:
      raise ValueError(f"Column {config.PROPERTY_KEY} not found in DataFrame")

    unique_properties = (
        self.df_loinc[config.PROPERTY_KEY].dropna().unique().tolist()
    )
    logging.info(
        "Found %d unique properties to process.", len(unique_properties)
    )

    results = []
    logging.info(
        "Starting Gemini prediction for %d properties...",
        len(unique_properties),
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
    future_to_prop = {
        executor.submit(self._process_single_property, prop): prop
        for prop in unique_properties
    }

    try:
      with tqdm.auto.tqdm(
          total=len(unique_properties),
          desc="Processing Properties",
          unit="prop",
      ) as pbar:
        for future in concurrent.futures.as_completed(future_to_prop):
          prop = future_to_prop[future]
          try:
            units = future.result()
            for unit in units:
              results.append({"property": prop, "synonym": unit})
            pbar.update(1)
          except Exception as e:  # pylint: disable=broad-except
            logging.exception("Error processing %s: %s", prop, e)
            pbar.update(1)
    finally:
      executor.shutdown(wait=True)

    df_results = pandas.DataFrame(results)
    logging.info(
        "Saving mapping (%d rows) to %s", len(df_results), save_csv_path
    )
    df_results.to_csv(save_csv_path, index=False)

    logging.info("Finished! Saved mapping to %s", save_csv_path)

  def _process_single_property(self, prop: str) -> list[str]:
    """Queries Gemini for a single property."""
    prompt = prompt_lib.PROMPT_TEMPLATE.format(item=prop)
    response = self.client.generate_content(contents=[prompt])
    result = utils.StringUtils.extract_json_list_from_llm_response(
        response.text
    )
    return result

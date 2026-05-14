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
"""Builds a Knowledge Base for System synonyms using an LLM."""

import concurrent.futures
import json

from absl import logging
import pandas
import tqdm.auto

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc import config
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.system import prompt as prompt_lib
from src.document_to_fhir.core.medical_coding.loinc.common import utils



class SystemKBBuilder:
  """Builds the System Synonym Knowledge Base."""

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
    """Generates system synonyms and saves to CSV."""
    if config.SYSTEM_KEY not in self.df_loinc.columns:
      raise ValueError(f"Column {config.SYSTEM_KEY} not found in DataFrame")

    unique_systems = self.df_loinc[config.SYSTEM_KEY].dropna().unique().tolist()
    logging.info("Found %d unique systems to process.", len(unique_systems))

    results = []
    logging.info(
        "Starting Gemini prediction for %d systems...",
        len(unique_systems),
    )

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.workers)
    future_to_system = {
        executor.submit(self._process_single_system, system): system
        for system in unique_systems
    }

    try:
      with tqdm.auto.tqdm(
          total=len(unique_systems),
          desc="Processing Systems",
          unit="system",
      ) as pbar:
        for future in concurrent.futures.as_completed(future_to_system):
          system = future_to_system[future]
          try:
            synonyms = future.result()
            for synonym in synonyms:
              results.append({"canonical": system, "synonym": synonym})
            pbar.update(1)
          except Exception as e:  # pylint: disable=broad-except
            logging.exception("Error processing %s: %s", system, e)
            pbar.update(1)
    finally:
      executor.shutdown(wait=True)

    df_results = pandas.DataFrame(results)
    logging.info(
        "Saving mapping (%d rows) to %s", len(df_results), save_csv_path
    )
    df_results.to_csv(save_csv_path, index=False)

    logging.info("Finished! Saved mapping to %s", save_csv_path)

  def _process_single_system(self, system: str) -> list[str]:
    """Queries Gemini for a single system."""
    prompt = prompt_lib.PROMPT_TEMPLATE.format(item=system)
    response = self.client.generate_content(contents=[prompt])
    result = utils.StringUtils.extract_json_list_from_llm_response(
        response.text
    )
    return result

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
"""Sequential 3-stage LLM Pipeline Builder for Vaccine Knowledge Base.

This module executes 3 LLM prompt stages sequentially at temperature = 0:
  Stage 1: Parent-child hierarchy extraction.
  Stage 2: Clinical variations and brand names generation.
  Stage 3: Antigen constituents extraction.

The first prompt takes an initial CSV (as a DataFrame) along with a text prompt.
The LLM response is parsed into a DataFrame and fed as input context to the next
prompt, repeated across 3 stages in total. The final output passes through a
post-processing stage (display name mapping and patient type determination)
before optional saving.
"""

import dataclasses

import io
import json
import re
from typing import Any

from absl import logging
import numpy as np
import pandas as pd

from src.document_to_fhir.common import file_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine.axes_kb import prompts

_DEFAULT_TEMPERATURE: float = 0.0
_DEFAULT_MAX_ATTEMPTS: int = 3
_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:csv)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE
)
_THOUGHT_PATTERN = re.compile(
    r"<\|channel>thought\n(.*?)\n<channel\|>", re.DOTALL
)
_N_SNOMED_NDHM_VACCINE_CODES = 179


def extract_csv_from_response(raw_text: str) -> str:
  """Extracts clean CSV text from an LLM response.

  Handles responses wrapped in markdown code blocks (e.g. ```csv ... ```),
  thought channels, or raw CSV text.

  Args:
    raw_text: The raw string response from the LLM.

  Returns:
    The extracted CSV text string.

  Raises:
    ValueError: If the extracted CSV text is empty.
  """
  clean_text = raw_text.strip()
  clean_text = _THOUGHT_PATTERN.sub("", clean_text).strip()

  matches = _CODE_BLOCK_PATTERN.findall(clean_text)
  if matches:
    # Use the last code block if multiple exist, or the largest one
    clean_text = max(matches, key=len).strip()

  if not clean_text:
    raise ValueError(
        f"No CSV content could be extracted from LLM response: {raw_text!r}"
    )

  return clean_text


def parse_csv_to_dataframe(csv_text: str) -> pd.DataFrame:
  """Parses a CSV text string into a pandas DataFrame.

  Args:
    csv_text: CSV formatted string.

  Returns:
    Parsed pandas DataFrame.
  """
  return pd.read_csv(io.StringIO(csv_text))


def parse_llm_response_to_dataframe(response_text: str) -> pd.DataFrame:
  """Parses LLM response into a pandas DataFrame.

  Supports both JSON-encoded responses and CSV-formatted responses (wrapped in
  markdown code blocks, thought channels, or raw CSV text).

  Args:
    response_text: Raw string response from LLM.

  Returns:
    Parsed pandas DataFrame.
  """
  clean_text = response_text.strip()
  if clean_text.startswith(("[", "{")):
    try:
      parsed_json = json.loads(clean_text)
      df = pd.DataFrame(parsed_json)
      if not df.empty:
        return df
    except (json.JSONDecodeError, ValueError):
      pass

  csv_text = extract_csv_from_response(clean_text)
  return parse_csv_to_dataframe(csv_text)


@dataclasses.dataclass(frozen=True)
class VaccinesKBColumnNames:
  """Column names configuration for the Vaccine Knowledge Base."""

  snomed_code_col_name: str
  preferred_term_col_name: str
  parent_snomed_code_col_name: str
  parent_preferred_display_term_col_name: str
  patient_type_col_name: str
  clinical_variations_col_name: str
  antigens_col_name: str


class VaccinesKBBuilder:
  """Builds Vaccine Knowledge Base via a 3-stage sequential LLM pipeline.

  Each stage loads a prompt template from `prompts.py`, formats it with the
  current DataFrame serialized to CSV, and queries the LLM at temperature = 0.
  The resulting DataFrame is parsed and fed into the next stage.
  """

  def __init__(
      self,
      df: pd.DataFrame,
      client: model_client.LLMClient,
      input_snomed_code_col_name: str,
      input_preferred_term_col_name: str,
      column_names: VaccinesKBColumnNames,
      n_snomed_codes: int = _N_SNOMED_NDHM_VACCINE_CODES,
      temperature: float = _DEFAULT_TEMPERATURE,
      max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
  ):
    """Initializes the builder.

    Args:
      df: Initial pandas DataFrame containing SNOMED vaccine codes.
      client: The LLM client instance to use for querying the model.
      input_snomed_code_col_name: Name of column containing SNOMED codes in the
        input dataframe.
      input_preferred_term_col_name: Name of column containing preferred terms
        in the input dataframe.
      column_names: Optional configuration of column names for the knowledge
        base.
      n_snomed_codes: Expected number of SNOMED codes (defaults to
        _N_SNOMED_NDHM_VACCINE_CODES).
      temperature: Sampling temperature for LLM generation (default: 0.0).
      max_attempts: Maximum retry attempts per LLM stage on failure.
    """
    if client is None:
      raise ValueError("LLM `client` cannot be None.")
    if df is None:
      raise ValueError("Input `df` cannot be None.")

    # Fix the column names in the final dataframe - this is an important
    # normalization step that ensures that the column names are consistent with
    # the column names used in the prompts.
    self.snomed_code_col_name = column_names.snomed_code_col_name
    self.preferred_term_col_name = column_names.preferred_term_col_name
    self.parent_snomed_code_col_name = (
        column_names.parent_snomed_code_col_name
    )
    self.parent_preferred_display_term_col_name = (
        column_names.parent_preferred_display_term_col_name
    )
    self.patient_type_col_name = column_names.patient_type_col_name
    self.clinical_variations_col_name = (
        column_names.clinical_variations_col_name
    )
    self.antigens_col_name = column_names.antigens_col_name

    # Pre-processing: Standardization of input dataframe column names
    df.rename(
        columns={
            input_snomed_code_col_name: self.snomed_code_col_name,
            input_preferred_term_col_name: self.preferred_term_col_name,
        },
        inplace=True,
    )
    self.df = df

    self._n_expected_snomed_codes = n_snomed_codes
    self.client = client
    self.temperature = temperature
    self.max_attempts = max_attempts

  def _format_prompt(self, template: str, df: pd.DataFrame) -> str:
    """Formats a prompt template by inserting DataFrame CSV content.

    Replaces `{input_csv}` or `{input_data}` placeholders with the CSV
    representation of the DataFrame. If neither placeholder is present,
    appends the CSV to the end of the template.

    Args:
      template: The prompt template text.
      df: The DataFrame to format as CSV input context.

    Returns:
      The formatted prompt string ready to pass to the LLM.
    """
    csv_str = df.to_csv(index=False)

    template = template.replace("{input_csv}", csv_str)
    template = template.replace(
        "{expected_snomed_codes_count}", str(self._n_expected_snomed_codes)
    )
    template = template.replace(
        "{snomed_code_col_name}", self.snomed_code_col_name
    )
    template = template.replace(
        "{parent_snomed_code_col_name}", self.parent_snomed_code_col_name
    )
    template = template.replace(
        "{preferred_term_col_name}", self.preferred_term_col_name
    )
    template = template.replace(
        "{clinical_variations_col_name}", self.clinical_variations_col_name
    )
    template = template.replace("{antigens_col_name}", self.antigens_col_name)

    return template

  def _generate_llm_response(self, prompt_text: str) -> str:
    """Queries the LLM client with the given prompt text at temperature = 0.

    Args:
      prompt_text: Complete prompt text sent to the model.

    Returns:
      The model's textual response.
    """
    response = self.client.generate_content(
        contents=[prompt_text],
        config={"temperature": self.temperature},
    )
    return response.text

  def _determine_patient_type(
      self, preferred_name: str
  ) -> resources.PatientType:
    """Determines the age bracket of the patient from the preferred name."""
    preferred_name = preferred_name.lower()

    if "adult" in preferred_name:
      return resources.PatientType.ADULT
    elif "paediatric" in preferred_name or "pediatric" in preferred_name:
      return resources.PatientType.PAEDIATRIC
    else:
      return resources.PatientType.UNKNOWN

  def _fetch_preferred_display_name_for_snomed_codes(
      self, df: pd.DataFrame, snomed_codes: Any
  ) -> str:
    """Fetches preferred display names for one or more SNOMED codes.

    Args:
      df: DataFrame containing the SNOMED codes and their preferred display
        names.
      snomed_codes: Comma-separated string of SNOMED codes (e.g. "123, 456"),
        numeric code (int or float), "N/A", or missing/null value.

    Returns:
      A pipe-delimited string of matching preferred display names, or "N/A" if
      no matches are found or if the code is invalid.
    """
    if snomed_codes is None or pd.isna(snomed_codes):
      return "N/A"

    if (
        isinstance(snomed_codes, (float, np.floating))
        and snomed_codes.is_integer()
    ):
      snomed_codes_str = str(int(snomed_codes))
    else:
      snomed_codes_str = str(snomed_codes)

    cleaned_codes = []
    for code in snomed_codes_str.split(","):
      code = code.strip()
      if not code or code.upper() in ("N/A", "NAN", "NONE", "<NA>"):
        continue
      if code.endswith(".0") and code[:-2].isdigit():
        code = code[:-2]
      cleaned_codes.append(code)

    if not cleaned_codes:
      return "N/A"

    unique_codes = np.unique(cleaned_codes).tolist()
    lookup_series = (
        df[self.snomed_code_col_name]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )
    matches = df[lookup_series.isin(unique_codes)][self.preferred_term_col_name]

    if matches.empty:
      return "N/A"

    return "|".join(matches)

  def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
    """Post-processing stage on the final DataFrame.

    This stage will handle:
      1. Validation: Validate schema, mandatory columns, and data integrity.
      2. String manipulation: Text normalization, trimming whitespace, and
         canonical formatting.

    Args:
      df: The final DataFrame output from the 3rd LLM stage.

    Returns:
      The post-processed DataFrame.
    """
    logging.info("--- [Phase: Post-Processing] ---")
    processed_df = df.copy()
    processed_df[self.parent_preferred_display_term_col_name] = processed_df[
        self.parent_snomed_code_col_name
    ].apply(
        lambda x: self._fetch_preferred_display_name_for_snomed_codes(
            processed_df,
            x,
        ),
    )

    processed_df[self.patient_type_col_name] = processed_df[
        self.preferred_term_col_name
    ].apply(self._determine_patient_type)
    return processed_df

  def infer_parent_child_relationships(
      self, df: pd.DataFrame | None = None
  ) -> pd.DataFrame:
    """Executes Stage 1: Informs the LLM to infer parent-child relationships.

    Formats the Stage 1 prompt with the provided DataFrame (or self.df if None)
    and queries the LLM to extract parent SNOMED codes. Retries up to
    self.max_attempts times on failure or row-count mismatch.

    Args:
      df: Optional input DataFrame containing SNOMED vaccine codes and preferred
        terms. Defaults to self.df.copy().

    Returns:
      DataFrame with the parent_snomed_code column populated.

    Raises:
      ValueError: If generation fails after max_attempts.
    """
    snomed_base_df = self.df.copy() if df is None else df.copy()
    snomed_relationships_df = pd.DataFrame()

    # Stage 1: Prompt the model to infer parent-child relationships in the
    # provided set of SNOMED-CT vaccination codes.
    template = prompts.PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY
    prompt_text = self._format_prompt(template, snomed_base_df)

    for attempt in range(1, self.max_attempts + 1):
      try:
        logging.info(
            "Executing Stage 1/3 to infer SNOMED parent-child relationships"
            " - Attempt %d/%d...",
            attempt,
            self.max_attempts,
        )
        response_text = self._generate_llm_response(prompt_text)
        snomed_relationships_df = parse_llm_response_to_dataframe(response_text)

        if len(snomed_relationships_df) == self._n_expected_snomed_codes:
          logging.info(
              "Stage 1 returned %d rows, expected %d. Proceeding to the next"
              " stage.",
              len(snomed_relationships_df),
              self._n_expected_snomed_codes,
          )
          break
        else:
          logging.warning(
              "Stage 1 returned %d rows, expected %d. Retrying...",
              len(snomed_relationships_df),
              self._n_expected_snomed_codes,
          )

      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Attempt %d failed while generating SNOMED parent-child"
            " relationships: %s",
            attempt,
            e,
        )

    if snomed_relationships_df.empty:
      raise ValueError(
          "Failed to generate SNOMED parent-child relationships after"
          f" {self.max_attempts} attempts."
      )

    return snomed_relationships_df

  def generate_clinical_variations(self, df: pd.DataFrame) -> pd.DataFrame:
    """Executes Stage 2: Generates clinical variations and brand names.

    Formats the Stage 2 prompt with the provided DataFrame (from Stage 1)
    and queries the LLM to generate clinical variations and brand names. Retries
    up to self.max_attempts times on failure or row-count mismatch.

    Args:
      df: Input DataFrame containing SNOMED vaccine codes, preferred terms, and
        parent SNOMED codes.

    Returns:
      DataFrame with the clinical_variations column populated.

    Raises:
      ValueError: If generation fails after max_attempts.
    """
    snomed_clinical_variations_df = pd.DataFrame()

    # Stage 2: Generate clinical variations for each vaccine in the SNOMED-CT
    # codes.
    template = prompts.PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS
    prompt_text = self._format_prompt(template, df)

    for attempt in range(1, self.max_attempts + 1):
      try:
        logging.info(
            "Executing Stage 2/3 to generate clinical variations and brand"
            " names for each vaccine in the SNOMED-CT codes"
            " - Attempt %d/%d...",
            attempt,
            self.max_attempts,
        )
        response_text = self._generate_llm_response(prompt_text)
        snomed_clinical_variations_df = parse_llm_response_to_dataframe(
            response_text
        )
        if len(snomed_clinical_variations_df) == self._n_expected_snomed_codes:
          logging.info(
              "Stage 2 returned %d rows, expected %d. Proceeding to the next"
              " stage.",
              len(snomed_clinical_variations_df),
              self._n_expected_snomed_codes,
          )
          break
        else:
          logging.warning(
              "Stage 2 returned %d rows, expected %d. Retrying...",
              len(snomed_clinical_variations_df),
              self._n_expected_snomed_codes,
          )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Attempt %d failed while generating SNOMED clinical variations: %s",
            attempt,
            e,
        )

    if snomed_clinical_variations_df.empty:
      raise ValueError(
          "Failed to generate SNOMED clinical variations after"
          f" {self.max_attempts} attempts."
      )

    return snomed_clinical_variations_df

  def generate_antigens(self, df: pd.DataFrame) -> pd.DataFrame:
    """Executes Stage 3: Generates list of antigens in each vaccine.

    Formats the Stage 3 prompt with the provided DataFrame (from Stage 2)
    and queries the LLM to extract antigen constituents. Retries up to
    self.max_attempts times on failure or row-count mismatch.

    Args:
      df: Input DataFrame containing SNOMED vaccine codes, preferred terms,
        parent codes, and clinical variations.

    Returns:
      DataFrame with the antigens column populated.

    Raises:
      ValueError: If generation fails after max_attempts.
    """
    snomed_final_df = pd.DataFrame()

    # Stage 3: Generate list of antigens in each vaccine.
    template = prompts.PROMPT_SNOMED_VACCINES_ANTIGENS
    prompt_text = self._format_prompt(template, df)

    for attempt in range(1, self.max_attempts + 1):
      try:
        logging.info(
            "Executing Stage 3/3 to generate list of antigens in each vaccine"
            " - Attempt %d/%d...",
            attempt,
            self.max_attempts,
        )
        response_text = self._generate_llm_response(prompt_text)

        snomed_final_df = parse_llm_response_to_dataframe(response_text)
        if len(snomed_final_df) == self._n_expected_snomed_codes:
          logging.info(
              "Stage 3 returned %d rows, expected %d. Proceeding to post-"
              "processing.",
              len(snomed_final_df),
              self._n_expected_snomed_codes,
          )
          break
        else:
          logging.warning(
              "Stage 3 returned %d rows, expected %d. Retrying...",
              len(snomed_final_df),
              self._n_expected_snomed_codes,
          )

      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Attempt %d failed while generating SNOMED vaccine antigens: %s",
            attempt,
            e,
        )

    if snomed_final_df.empty:
      raise ValueError(
          "Failed to generate SNOMED vaccine antigens after"
          f" {self.max_attempts} attempts."
      )

    return snomed_final_df

  def build_kb(self, save_tsv_path: str = "") -> pd.DataFrame:
    """Executes the full 3-stage sequential pipeline and post-processing.

    Passes the output DataFrame of each stage as input context to the next
    stage, runs the post-processing stage on the 3rd stage's response, and
    optionally saves the final DataFrame to disk.

    Args:
      save_tsv_path: Optional filepath to save the final post-processed
        DataFrame as a TSV.

    Returns:
      The final post-processed DataFrame.
    """
    logging.info("--- Starting Sequential Vaccine KB Build ---")
    snomed_base_df = self.df.copy()

    # Stage 1: Prompt the model to infer parent-child relationships in the
    # provided set of SNOMED-CT vaccination codes.
    snomed_relationships_df = self.infer_parent_child_relationships(
        snomed_base_df
    )

    # Stage 2: Generate clinical variations for each vaccine in the SNOMED-CT
    # codes.
    snomed_clinical_variations_df = self.generate_clinical_variations(
        snomed_relationships_df
    )

    # Stage 3: Generate list of antigens in each vaccine.
    snomed_final_df = self.generate_antigens(snomed_clinical_variations_df)

    # Post-processing Stage: Validate schema, mandatory columns, and data
    # integrity.
    logging.info("--- [Phase: Post-Processing] ---")
    processed_df = self._post_process(snomed_final_df)

    if save_tsv_path:
      logging.info(
          "Saving final KB (%d rows, %d cols) to: %s",
          len(processed_df),
          len(processed_df.columns),
          save_tsv_path,
      )
      with file_util.open_file(save_tsv_path, "w") as f:
        processed_df.to_csv(f, sep="\t", index=False)
      logging.info("Saved final KB successfully to %s", save_tsv_path)

    logging.info("--- Vaccine KB Build Complete ---")
    return processed_df

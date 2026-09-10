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
"""CLI binary to run the Vaccine Axes KB Builder using an LLM client."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from src.document_to_fhir.common import file_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.snomed.vaccine.axes_kb import builder

_INPUT_CSV_PATH = flags.DEFINE_string(
    "input_csv_path",
    None,
    "Path to the input CSV file to be processed. This should contain the SNOMED"
    " codes and preferred display terms and can be downloaded from the official"
    " website.",
    required=True,
)
_INPUT_SNOMED_CODE_COL_NAME = flags.DEFINE_string(
    "input_snomed_code_col_name",
    None,
    "Name of the column in the input CSV containing SNOMED codes",
    required=True,
)
_INPUT_PREFERRED_TERM_COL_NAME = flags.DEFINE_string(
    "input_preferred_term_col_name",
    None,
    "Name of the column in the input CSV containing preferred display terms.",
    required=True,
)
_CLIENT_TYPE = flags.DEFINE_string(
    "client_type",
    None,
    "Type of LLM client to use (gemini, gemma, litellm).",
    required=True,
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    None,
    "LLM model name.",
    required=True,
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    os.environ.get("API_KEY") or os.environ.get("GEMINI_API_KEY", ""),
    "API Key for the Gemini LLM client. Defaults to API_KEY / GEMINI_API_KEY"
    " env vars.",
)
_OUTPUT_TSV_PATH = flags.DEFINE_string(
    "output_tsv_path",
    None,
    "Filepath where the final post-processed TSV will be saved.",
    required=True,
)
_SNOMED_CODE_COL_NAME = flags.DEFINE_string(
    "snomed_code_col_name",
    "snomed_code",
    "Name of the column containing the SNOMED code.",
)
_PREFERRED_TERM_COL_NAME = flags.DEFINE_string(
    "preferred_term_col_name",
    "preferred_display_term",
    "Name of the column containing the preferred display term.",
)
_PARENT_SNOMED_CODE_COL_NAME = flags.DEFINE_string(
    "parent_snomed_code_col_name",
    "parent_snomed_code",
    "Name of the column that contains the parent SNOMED code.",
)
_PARENT_PREFERRED_DISPLAY_TERM_COL_NAME = flags.DEFINE_string(
    "parent_preferred_display_term_col_name",
    "parent_preferred_display_term",
    "Name of the column that contains the parent preferred display term.",
)
_PATIENT_TYPE_COL_NAME = flags.DEFINE_string(
    "patient_type_col_name",
    "patient_type",
    "Name of the column that contains the patient type.",
)
_CLINICAL_VARIATIONS_COL_NAME = flags.DEFINE_string(
    "clinical_variations_col_name",
    "clinical_variations",
    "Name of the column that contains clinical variations.",
)
_ANTIGENS_COL_NAME = flags.DEFINE_string(
    "antigens_col_name",
    "antigens",
    "Name of the column that contains antigens.",
)
_MAX_ATTEMPTS = flags.DEFINE_integer(
    "max_attempts",
    3,
    "Maximum retry attempts per stage on failure.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logging.info(
      "Initializing %s client using model: %s",
      _CLIENT_TYPE.value,
      _MODEL_NAME.value,
  )
  try:
    llm_client = model_client.create_llm_client(
        client_type=_CLIENT_TYPE.value,
        model_name=_MODEL_NAME.value,
        api_key=_API_KEY.value,  # pyrefly: ignore[bad-argument-type]
    )
  except ValueError as e:
    raise app.UsageError(repr(e)) from e

  with file_util.open_file(_INPUT_CSV_PATH.value, "r") as f:
    snomed_vaccine_df = pd.read_csv(f)

  if snomed_vaccine_df is None or snomed_vaccine_df.empty:
    raise app.UsageError("Failed to load any records. Check path and criteria.")

  input_snomed_code_col_name = _INPUT_SNOMED_CODE_COL_NAME.value
  input_preferred_term_col_name = _INPUT_PREFERRED_TERM_COL_NAME.value

  column_names = builder.VaccinesKBColumnNames(
      snomed_code_col_name=_SNOMED_CODE_COL_NAME.value,
      preferred_term_col_name=_PREFERRED_TERM_COL_NAME.value,
      parent_snomed_code_col_name=_PARENT_SNOMED_CODE_COL_NAME.value,
      parent_preferred_display_term_col_name=(
          _PARENT_PREFERRED_DISPLAY_TERM_COL_NAME.value
      ),
      patient_type_col_name=_PATIENT_TYPE_COL_NAME.value,
      clinical_variations_col_name=_CLINICAL_VARIATIONS_COL_NAME.value,
      antigens_col_name=_ANTIGENS_COL_NAME.value,
  )

  kb_builder = builder.VaccinesKBBuilder(
      df=snomed_vaccine_df,
      client=llm_client,
      input_snomed_code_col_name=input_snomed_code_col_name,
      input_preferred_term_col_name=input_preferred_term_col_name,
      column_names=column_names,
      n_snomed_codes=len(snomed_vaccine_df),
      max_attempts=_MAX_ATTEMPTS.value,
      temperature=0.0,
  )

  logging.info("Starting sequential build across 3 prompt stages...")
  final_df = kb_builder.build_kb(save_tsv_path=_OUTPUT_TSV_PATH.value)
  logging.info(
      "Pipeline finished successfully! Final output shape: %s saved to %s",
      final_df.shape,
      _OUTPUT_TSV_PATH.value,
  )


if __name__ == "__main__":
  app.run(main)

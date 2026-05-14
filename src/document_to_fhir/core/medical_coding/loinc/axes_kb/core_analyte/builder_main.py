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
"""CLI binary for building the LOINC Analyte Knowledge Base."""

import os

from absl import app
from absl import flags
from absl import logging

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import builder
from src.document_to_fhir.core.medical_coding.loinc.common import utils


_LOINC_CSV_PATH = flags.DEFINE_string(
    "loinc_csv_path",
    None,
    "Path to the input LOINC CSV file.",
    required=True,
)
_OUTPUT_CSV_FOLDER = flags.DEFINE_string(
    "output_csv_folder",
    None,
    "Folder path where output CSV will be saved.",
    required=True,
)

_CLIENT_TYPE = flags.DEFINE_string(
    "client_type",
    None,
    "Type of LLM client to use (gemini, gemma, litellm).",
    required=True,
)

_API_KEY = flags.DEFINE_string(
    "api_key",
    os.environ.get("API_KEY", None),
    "API Key for the LLM client. Defaults to the API_KEY environment variable.",
    required=True,
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name", None, "LLM model name.", required=True,
)

_MAX_RANK = flags.DEFINE_integer(
    "max_rank", 2000, "Maximum LOINC rank."
)
_LIMIT = flags.DEFINE_integer(
    "limit", 0, "Limit number of rows (0 for none)."
)
_WORKERS = flags.DEFINE_integer(
    "workers", 10, "Number of parallel workers."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Construct filename based on max_rank and limit
  if _LIMIT.value > 0:
    filename = f"analyte_records_top_{_MAX_RANK.value}_limit_{_LIMIT.value}.csv"
  else:
    filename = f"analyte_records_top_{_MAX_RANK.value}.csv"

  output_path = os.path.join(_OUTPUT_CSV_FOLDER.value, filename)
  logging.info("Target output path resolved to: %s", output_path)

  logging.info("Reading LOINC KB from: %s", _LOINC_CSV_PATH.value)
  df_loinc = utils.read_loinc_kb(
      file_path=_LOINC_CSV_PATH.value,
      max_rank=_MAX_RANK.value,
      limit=_LIMIT.value,
  )

  if df_loinc is None or df_loinc.empty:
    logging.error("Failed to load any records. Check path and criteria.")

  logging.info(
      "Initializing %s client using model: %s",
      _CLIENT_TYPE.value,
      _MODEL_NAME.value,
  )
  try:
    llm_client = model_client.create_llm_client(
        client_type=_CLIENT_TYPE.value,
        model_name=_MODEL_NAME.value,
        api_key=_API_KEY.value,
    )
  except ValueError as e:
    raise app.UsageError(str(e)) from e

  logging.info("Starting build with %d workers...", _WORKERS.value)
  builder_instance = builder.AnalyteKBBuilder(
      df_loinc=df_loinc,
      client_inst=llm_client,
      workers=_WORKERS.value,
  )

  builder_instance.build_kb(save_csv_path=output_path)
  logging.info("Execution completed.")


if __name__ == "__main__":
  app.run(main)

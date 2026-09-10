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
"""Main script to build the LOINC Axis Property Knowledge Base.

This script reads LOINC data, uses a Gemini model to extract properties,
and saves the results to a CSV file.
"""

import datetime
import os
import pathlib

from absl import app
from absl import flags
from absl import logging

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import builder


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
    "gemini",
    "LLM Client Type (e.g., gemini, vertex_ai_http, vertex_ai_grpc).",
)
_API_KEY = flags.DEFINE_string(
    "api_key",
    "",
    "API Key for the LLM client. Defaults to the GEMINI_API_KEY environment"
    " variable.",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "gemini-flash-latest",
    "Model name to use (e.g. gemini-flash-latest).",
)
_MAX_RANK = flags.DEFINE_integer("max_rank", 2000, "Maximum LOINC rank.")
_LIMIT = flags.DEFINE_integer(
    "limit", 0, "Limit number of rows (0 for none)."
)
_WORKERS = flags.DEFINE_integer("workers", 10, "Number of parallel workers.")
_FILENAME_SUFFIX = flags.DEFINE_string(
    "filename_suffix",
    "",
    "Suffix to add to the output filename.",
)
_RESUME = flags.DEFINE_bool(
    "resume", False, "Resume build from existing output CSV if it exists."
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  limit_str = f"_limit_{_LIMIT.value}" if _LIMIT.value > 0 else ""
  suffix_str = f"_{_FILENAME_SUFFIX.value}" if _FILENAME_SUFFIX.value else ""
  filename = (
      f"loinc_axis_property_kb_top_{_MAX_RANK.value}{limit_str}_{now_str}{suffix_str}.csv"
  )

  output_path = pathlib.Path(_OUTPUT_CSV_FOLDER.value) / filename
  logging.info("Target output path resolved to: %s", output_path)

  logging.info("Reading LOINC KB from: %s", _LOINC_CSV_PATH.value)
  df_loinc = utils.read_loinc_kb(
      file_path=_LOINC_CSV_PATH.value,
      max_rank=_MAX_RANK.value,
      limit=_LIMIT.value,
  )

  if df_loinc is None or df_loinc.empty:
    logging.error("Failed to load any records. Check path and criteria.")
    return

  # Initialize LLM client
  api_key = _API_KEY.value or os.environ.get("GEMINI_API_KEY", "")
  if not api_key:
    logging.warning("API key not found. Gemini calls might fail.")

  logging.info(
      "Initializing LLM client of type %s with model %s",
      _CLIENT_TYPE.value,
      _MODEL_NAME.value,
  )
  client = model_client.create_llm_client(
      client_type=_CLIENT_TYPE.value,
      model_name=_MODEL_NAME.value,
      api_key=api_key,
  )

  logging.info("Initializing PropertyKBBuilder...")
  builder_instance = builder.PropertyKBBuilder(
      df_loinc=df_loinc,
      client=client,
      workers=_WORKERS.value,
  )

  builder_instance.build_kb(
      save_csv_path=str(output_path), resume=_RESUME.value
  )
  logging.info("Execution completed.")


if __name__ == "__main__":
  app.run(main)

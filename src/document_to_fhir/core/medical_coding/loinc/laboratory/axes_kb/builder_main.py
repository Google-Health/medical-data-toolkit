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
"""CLI binary for building the LOINC Axis Knowledge Bases."""

import datetime
import os
import pathlib

from absl import app
from absl import flags
from absl import logging
import pandas

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import builder
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import builder as property_builder
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system import builder as system_builder


_SUPPORTED_AXES = ("core_analyte", "property", "system")

_AXES = flags.DEFINE_list(
    "axes",
    ["core_analyte"],
    "List of axes to build ('core_analyte', 'property', 'system', or 'all').",
)
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
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    5,
    "Batch size for LLM processing. This should be set according to the"
    " limitations of the LLM client being used.",
)
_MAX_ATTEMPTS = flags.DEFINE_integer(
    "max_attempts",
    4,
    "Maximum number of attempts for processing LOINC batches.",
)
_RESUME = flags.DEFINE_bool(
    "resume", False, "Resume build from existing output CSV if it exists."
)
_ADDITIONAL_LOINC_CODES_FILE = flags.DEFINE_string(
    "additional_loinc_codes_file",
    None,
    "Path to a file containing a list of additional LOINC codes to process.",
)
_FILENAME_SUFFIX = flags.DEFINE_string(
    "filename_suffix",
    "",
    "Suffix to add to the output filename.",
)


def _resolve_axes(requested_axes: list[str]) -> list[str]:
  """Validates and expands requested axes, failing fast on unrecognized values."""
  normalized = [axis.strip().lower() for axis in requested_axes if axis.strip()]
  if not normalized:
    raise app.UsageError("At least one axis must be specified via --axes.")

  valid_options = set(_SUPPORTED_AXES) | {"all"}
  invalid_axes = [axis for axis in normalized if axis not in valid_options]
  if invalid_axes:
    raise app.UsageError(
        f"Unknown axis: {', '.join(invalid_axes)}. Supported:"
        f" {', '.join(_SUPPORTED_AXES)}, all"
    )

  if "all" in normalized:
    return list(_SUPPORTED_AXES)
  return list(dict.fromkeys(normalized))


def _get_output_filename(axis: str, date_str: str) -> str:
  """Constructs standard output filename for a given axis."""
  input_basename = pathlib.Path(_LOINC_CSV_PATH.value).stem
  sanitized_model = _MODEL_NAME.value.replace("/", "_").replace(":", "_")

  if axis == "core_analyte":
    filename_parts = [
        "analyte_records",
        input_basename,
        f"top_{_MAX_RANK.value}",
    ]
  else:
    filename_parts = [f"loinc_axis_{axis}_kb_top_{_MAX_RANK.value}"]

  if _LIMIT.value > 0:
    filename_parts.append(f"limit_{_LIMIT.value}")
  filename_parts.extend([_CLIENT_TYPE.value, sanitized_model, date_str])
  if _FILENAME_SUFFIX.value:
    filename_parts.append(_FILENAME_SUFFIX.value)
  return "__".join(filename_parts) + ".csv"


def _build_axis(
    axis: str,
    df_loinc: pandas.DataFrame,
    llm_client: model_client.LLMClient,
    date_str: str,
) -> None:
  """Builds a single LOINC axis knowledge base using the configured builder."""
  if axis not in _SUPPORTED_AXES:
    raise app.UsageError(
        f"Unknown axis: {axis}. Supported: {', '.join(_SUPPORTED_AXES)}, all"
    )

  filename = _get_output_filename(axis, date_str)
  output_path = pathlib.Path(_OUTPUT_CSV_FOLDER.value) / filename
  logging.info(
      "Target output path for '%s' resolved to: %s", axis, output_path
  )

  logging.info(
      "Starting '%s' KB build with %d workers...", axis, _WORKERS.value
  )
  if axis == "core_analyte":
    builder_instance = builder.AnalyteKBBuilder(
        df_loinc=df_loinc,
        client_inst=llm_client,
        workers=_WORKERS.value,
        batch_size=_BATCH_SIZE.value,
        max_attempts=_MAX_ATTEMPTS.value,
    )
    builder_instance.build_kb(
        save_csv_path=str(output_path), resume=_RESUME.value
    )
  elif axis == "property":
    builder_instance = property_builder.PropertyKBBuilder(
        df_loinc=df_loinc,
        client=llm_client,
        workers=_WORKERS.value,
        max_attempts=_MAX_ATTEMPTS.value,
    )
    builder_instance.build_kb(
        save_csv_path=str(output_path), resume=_RESUME.value
    )
  elif axis == "system":
    builder_instance = system_builder.SystemKBBuilder(
        df_loinc=df_loinc,
        client=llm_client,
        workers=_WORKERS.value,
        max_attempts=_MAX_ATTEMPTS.value,
    )
    builder_instance.build_kb(save_csv_path=str(output_path))


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  axes_to_run = _resolve_axes(_AXES.value)

  # Construct filename based on flags and date
  date_str = datetime.datetime.now().strftime("%Y%m%d")

  additional_loinc_codes = None
  if _ADDITIONAL_LOINC_CODES_FILE.value:
    logging.info(
        "Reading additional LOINC codes from file: %s",
        _ADDITIONAL_LOINC_CODES_FILE.value,
    )
    with utils.open_file(_ADDITIONAL_LOINC_CODES_FILE.value, "r") as f:
      additional_loinc_codes = [line.strip() for line in f if line.strip()]
    logging.info(
        "Found %d codes in additional file.", len(additional_loinc_codes)
    )

  logging.info("Reading LOINC KB from: %s", _LOINC_CSV_PATH.value)
  df_loinc = utils.read_loinc_kb(
      file_path=_LOINC_CSV_PATH.value,
      max_rank=_MAX_RANK.value,
      limit=_LIMIT.value,
      additional_loinc_codes=additional_loinc_codes,
  )

  if df_loinc is None or df_loinc.empty:
    logging.error("Failed to load any records. Check path and criteria.")
    return

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

  for axis in axes_to_run:
    _build_axis(axis, df_loinc, llm_client, date_str)
  logging.info("Execution completed.")


if __name__ == "__main__":
  app.run(main)

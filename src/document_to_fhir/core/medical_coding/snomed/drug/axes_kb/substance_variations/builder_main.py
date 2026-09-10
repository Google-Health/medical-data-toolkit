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
"""CLI binary for building the SNOMED Drug Substance Variations Knowledge Base.

This script reads SNOMED drug substance and generic data, clusters substance
variations (including salt forms in combination drugs), and generates a CSV
file containing the Substance Variations KB.
"""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags
from absl import logging

from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.axes_kb.substance_variations import builder

_SUBSTANCE_KB_PATH = flags.DEFINE_string(
    "substance_kb_path",
    None,
    "Path to the input SNOMED drug substance KB file (SubstanceMaster.txt).",
    required=True,
)
_GENERIC_KB_PATH = flags.DEFINE_string(
    "generic_kb_path",
    None,
    "Path to the input SNOMED drug generic KB file (GenericMaster.txt).",
    required=True,
)
_OUTPUT_FOLDER = flags.DEFINE_string(
    "output_folder",
    None,
    "Folder path where output CSV will be saved.",
    required=True,
)
_FILENAME = flags.DEFINE_string(
    "filename",
    "SubstanceVariations.txt",
    "Output TXT filename.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_path = pathlib.Path(_OUTPUT_FOLDER.value) / _FILENAME.value
  logging.info("Target output path resolved to: %s", output_path)

  logging.info("Reading SNOMED Substance KB from: %s", _SUBSTANCE_KB_PATH.value)
  substance_df = utils.read_snomed_drug_kb(_SUBSTANCE_KB_PATH.value)
  if substance_df is None or substance_df.empty:
    raise RuntimeError(
        f"Failed to load substance records from: {_SUBSTANCE_KB_PATH.value}"
    )

  logging.info("Reading SNOMED Generic KB from: %s", _GENERIC_KB_PATH.value)
  generic_df = utils.read_snomed_drug_kb(_GENERIC_KB_PATH.value)
  if generic_df is None or generic_df.empty:
    raise RuntimeError(
        f"Failed to load generic records from: {_GENERIC_KB_PATH.value}"
    )

  logging.info("Initializing SubstanceVariationsKBBuilder...")
  builder_instance = builder.SubstanceVariationsKBBuilder(
      substance_df=substance_df,
      generic_df=generic_df,
  )
  builder_instance.build_kb(
      save_txt_path=str(output_path)
  )
  logging.info("Execution completed.")

if __name__ == "__main__":
  app.run(main)

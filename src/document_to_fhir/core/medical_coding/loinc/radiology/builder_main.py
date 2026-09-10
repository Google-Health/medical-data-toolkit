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
"""CLI entry point for building the offline RSNA Playbook knowledge base."""

from collections.abc import Sequence
import csv

from absl import app
from absl import flags
from absl import logging

from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.radiology import builder
from src.document_to_fhir.core.medical_coding.loinc.radiology import config

_INPUT_PLAYBOOK_CSV = flags.DEFINE_string(
    "input_playbook_csv",
    None,
    "Path to official LoincRsnaRadiologyPlaybook.csv accessory table.",
    required=True,
)
_MASTER_LOINC_CSV = flags.DEFINE_string(
    "master_loinc_csv",
    None,
    "Optional path to master LoincTable.csv to read COMMON_TEST_RANK metadata.",
)
_MAX_COMMON_TEST_RANK = flags.DEFINE_integer(
    "max_common_test_rank",
    None,
    "Optional limit to filter and include only top-N ranked LOINC codes"
    " (e.g. 500 or 1000).",
)
_OUTPUT_CSV = flags.DEFINE_string(
    "output_csv",
    None,
    "Output destination path where the normalized RSNA Playbook dataset"
    " should be stored.",
    required=True,
)


def load_ranks_from_master_table(master_csv_path: str) -> dict[str, int]:
  """Reads COMMON_TEST_RANK map from master LoincTable.csv."""
  ranks: dict[str, int] = {}
  with utils.open_file(master_csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
      loinc_raw = (
          row.get("LOINC_NUM") or row.get(config.COL_LOINC_NUMBER) or ""
      )
      loinc = str(loinc_raw).strip()
      rank_raw = row.get(config.FIELD_COMMON_TEST_RANK) or ""
      rank_str = str(rank_raw).strip()
      if loinc and rank_str.isdigit():
        ranks[loinc] = int(rank_str)
  return ranks


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_path = _INPUT_PLAYBOOK_CSV.value
  output_path = _OUTPUT_CSV.value
  max_rank = _MAX_COMMON_TEST_RANK.value

  with utils.open_file(input_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

  if not rows:
    logging.error("No rows found in %s", input_path)
    return

  builder_instance = builder.RsnaPlaybookBuilder()

  # Load COMMON_TEST_RANK map if master table is provided
  rank_map: dict[str, int] | None = None
  if _MASTER_LOINC_CSV.value:
    logging.info(
        "Loading COMMON_TEST_RANK from %s...", _MASTER_LOINC_CSV.value
    )
    rank_map = load_ranks_from_master_table(_MASTER_LOINC_CSV.value)
    logging.info("Loaded %d ranked codes.", len(rank_map))

  logging.info(
      "Processing official LOINC/RSNA Playbook accessory table (%d rows)...",
      len(rows),
  )
  processed = builder_instance.process_official_playbook_table(
      rows, common_test_rank_map=rank_map, max_rank=max_rank
  )

  logging.info(
      "Saving %d normalized RSNA Playbook records to %s...",
      len(processed),
      output_path,
  )
  builder_instance.save_to_csv(processed, output_path)
  logging.info("Done!")


if __name__ == "__main__":
  app.run(main)

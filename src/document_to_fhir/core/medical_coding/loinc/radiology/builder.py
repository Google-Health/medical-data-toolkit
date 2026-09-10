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
"""Builder for offline RSNA Playbook dataset from official LOINC/RSNA tables."""

import collections
import csv
import re
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.radiology import config


def normalize_text(text: str | None) -> str:
  """Normalizes string for search indexing."""
  if not text:
    return ""
  return (
      re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip().lower()
  )


def _combine_parts(first: str | None, second: str | None) -> str:
  """Combines two related LOINC parts (e.g. Region + Focus).

  Args:
    first: The primary part value.
    second: The secondary part value.

  Returns:
    The combined non-redundant string representation.
  """
  f = (first or "").strip()
  s = (second or "").strip()
  if f and s:
    if f.lower() == s.lower() or s.lower() in f.lower():
      return f
    if f.lower() in s.lower():
      return s
    return f"{f} {s}"
  return f or s


class RsnaPlaybookBuilder:
  """Builds normalized RSNA Playbook dataset from official tables."""

  def process_official_playbook_table(
      self,
      playbook_rows: list[dict[str, str | None]],
      common_test_rank_map: dict[str, int] | None = None,
      max_rank: int | None = None,
  ) -> list[dict[str, str]]:
    """Builds records from official accessory table with optional rank filtering."""
    grouped_parts: dict[str, dict[str, str]] = collections.defaultdict(dict)
    loinc_names: dict[str, str] = {}
    rpid_map: dict[str, str] = {}

    for row in playbook_rows:
      loinc = (row.get(config.COL_LOINC_NUMBER) or "").strip()
      if not loinc:
        continue

      long_name = (row.get(config.COL_LONG_COMMON_NAME) or "").strip()
      if long_name:
        loinc_names[loinc] = long_name

      rpid = (row.get(config.COL_RPID) or "").strip()
      if rpid:
        rpid_map[loinc] = rpid

      part_type = (row.get(config.COL_PART_TYPE_NAME) or "").strip()
      part_name = (row.get(config.COL_PART_NAME) or "").strip()
      if part_type and part_name:
        grouped_parts[loinc][part_type] = part_name

    processed: list[dict[str, str]] = []
    for loinc, parts in grouped_parts.items():
      long_name = loinc_names.get(loinc, "")

      # Check COMMON_TEST_RANK limit if provided
      rank_val = (
          common_test_rank_map.get(loinc) if common_test_rank_map else None
      )
      if max_rank is not None:
        if rank_val is None or rank_val > max_rank:
          continue

      modality_type = parts.get(config.PART_TYPE_MODALITY_TYPE, "")
      modality_subtype = parts.get(config.PART_TYPE_MODALITY_SUBTYPE, "")
      region = _combine_parts(
          parts.get(config.PART_TYPE_REGION_IMAGED),
          parts.get(config.PART_TYPE_IMAGING_FOCUS),
      )
      contrast = _combine_parts(
          parts.get(config.PART_TYPE_TIMING),
          parts.get(config.PART_TYPE_SUBSTANCE_GIVEN),
      )
      view_type = _combine_parts(
          parts.get(config.PART_TYPE_VIEW_TYPE),
          parts.get(config.PART_TYPE_VIEW_AGGREGATION),
      )
      laterality = parts.get(config.PART_TYPE_LATERALITY, "")

      rpid = rpid_map.get(loinc, "")
      search_text = normalize_text(
          f"{long_name} {modality_type} {modality_subtype} {region} {contrast}"
          f" {view_type} {laterality}"
      )

      processed.append({
          config.FIELD_LOINC_NUM: loinc,
          config.FIELD_LONG_COMMON_NAME: long_name,
          config.FIELD_MODALITY_TYPE: modality_type,
          config.FIELD_MODALITY_SUBTYPE: modality_subtype,
          config.FIELD_REGION_IMAGED: region,
          config.FIELD_CONTRAST_USED: contrast,
          config.FIELD_VIEW_TYPE: view_type,
          config.FIELD_LATERALITY: laterality,
          config.FIELD_COMMON_TEST_RANK: (
              str(rank_val) if rank_val is not None else ""
          ),
          config.FIELD_RPID: rpid,
          config.FIELD_SEARCH_TEXT: search_text,
      })

    return processed

  def save_to_csv(self, records: list[dict[str, str]], filepath: str) -> None:
    with utils.open_file(filepath, "w") as f:
      writer = csv.DictWriter(f, fieldnames=config.INDEX_FIELDNAMES)
      writer.writeheader()
      writer.writerows(records)

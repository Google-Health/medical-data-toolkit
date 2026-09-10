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
"""Runtime engine for resolving Radiology reports to RSNA LOINC codes."""

import collections.abc
import csv
import dataclasses
import re

from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.radiology import config
from src.document_to_fhir.core.medical_coding.loinc.radiology import mappings

_NORM_MODALITY = "_norm_modality"
_NORM_SUBTYPE = "_norm_subtype"
_NORM_REGION = "_norm_region"
_NORM_CONTRAST = "_norm_contrast"
_NORM_VIEW = "_norm_view"
_NORM_LATERALITY = "_norm_laterality"


@dataclasses.dataclass(frozen=True)
class _NormalizedRsnaQuery:
  """Container for pre-normalized radiology query attributes."""
  modality: str
  subtype: str
  region: str
  contrast: str
  view: str
  laterality: str


def normalize_text(text: str | None) -> str:
  """Normalizes string for searching and comparison."""
  if not text:
    return ""
  return (
      re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip().lower()
  )


_COMPILED_PATTERN_CACHE: dict[int, tuple[tuple[re.Pattern[str], str], ...]] = {}


def normalize_attribute(
    text: str | None,
    pattern_dict: collections.abc.Mapping[str, str],
) -> str:
  """Normalizes free-text attributes using cached pre-compiled regex patterns."""
  if not text or not text.strip():
    return ""

  dict_id = id(pattern_dict)
  if dict_id not in _COMPILED_PATTERN_CACHE:
    _COMPILED_PATTERN_CACHE[dict_id] = tuple(
        (re.compile(pattern), normalized_val)
        for pattern, normalized_val in pattern_dict.items()
    )

  search_text = text.lower()
  for compiled_pattern, normalized_value in _COMPILED_PATTERN_CACHE[dict_id]:
    if compiled_pattern.search(search_text):
      return normalized_value
  return text.strip()


class RsnaPlaybookQueryEngine:
  """Maps RadiologyReport fields to procedure LOINC codes."""

  def __init__(self, records: list[dict[str, str]]):
    self._records = records
    self._tuple_index: dict[
        tuple[str, str, str, str, str, str], dict[str, str]
    ] = {}
    self._build_index()

  @classmethod
  def from_csv(cls, filepath: str) -> "RsnaPlaybookQueryEngine":
    if not utils.file_exists(filepath):
      raise FileNotFoundError(f"RSNA Playbook CSV file not found: {filepath}")
    with utils.open_file(filepath, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      records = list(reader)
    return cls(records)

  def map_report(self, report: medical_documents.RadiologyReport) -> None:
    """Maps procedure LOINC code on a RadiologyReport in-place."""
    if not report.procedure_loinc_code:
      loinc_number, loinc_name = self.match_procedure(
          modality_type=report.modality_type,
          modality_subtype=report.modality_subtype,
          region_imaged=report.region_imaged,
          contrast_used=report.contrast_used,
          view_type=report.view_type,
          laterality=report.laterality,
          exam_name=report.exam_name,
      )
      if loinc_number:
        report.procedure_loinc_code = loinc_number
        report.procedure_loinc_common_name = loinc_name

  def match_procedure(
      self,
      modality_type: str | None,
      modality_subtype: str | None,
      region_imaged: str | None,
      contrast_used: str | None,
      view_type: str | None,
      laterality: str | None,
      exam_name: str,
  ) -> tuple[str | None, str | None]:
    """Resolves procedure LOINC code via 3-tier matching with robust normalization."""
    query = self._normalize_query_attributes(
        modality_type=modality_type,
        modality_subtype=modality_subtype,
        region_imaged=region_imaged,
        contrast_used=contrast_used,
        view_type=view_type,
        laterality=laterality,
    )

    # Tier 1: Exact Tuple Match
    loinc_num, loinc_name = self._match_exact_tuple(query)
    if loinc_num:
      return loinc_num, loinc_name

    # Tier 1b: Match Modality (+ Subtype) + Region + View (Primary for XR)
    if query.view:
      loinc_num, loinc_name = self._match_by_view(query)
      if loinc_num:
        return loinc_num, loinc_name

    # Tier 1c: Match Modality (+ Subtype) + Region + Contrast (Primary for CT/MR)
    if query.contrast:
      loinc_num, loinc_name = self._match_by_contrast(query)
      if loinc_num:
        return loinc_num, loinc_name

    # Tier 2: Candidate filtering by modality + region + text similarity scoring
    loinc_num, loinc_name = self._match_by_similarity_score(query, exam_name)
    if loinc_num:
      return loinc_num, loinc_name

    # Tier 3: Modality fallback match
    if query.modality:
      return self._match_by_modality_fallback(query.modality)

    return None, None

  def _build_index(self) -> None:
    for record in self._records:
      modality = normalize_text(record.get(config.FIELD_MODALITY_TYPE, ""))
      subtype = normalize_text(record.get(config.FIELD_MODALITY_SUBTYPE, ""))
      region = normalize_text(record.get(config.FIELD_REGION_IMAGED, ""))
      contrast = normalize_text(record.get(config.FIELD_CONTRAST_USED, ""))
      view = normalize_text(record.get(config.FIELD_VIEW_TYPE, ""))
      laterality = normalize_text(record.get(config.FIELD_LATERALITY, ""))

      record[_NORM_MODALITY] = modality
      record[_NORM_SUBTYPE] = subtype
      record[_NORM_REGION] = region
      record[_NORM_CONTRAST] = contrast
      record[_NORM_VIEW] = view
      record[_NORM_LATERALITY] = laterality

      index_key = (modality, subtype, region, contrast, view, laterality)
      if index_key not in self._tuple_index:
        self._tuple_index[index_key] = record

  def _normalize_query_attributes(
      self,
      modality_type: str | None,
      modality_subtype: str | None,
      region_imaged: str | None,
      contrast_used: str | None,
      view_type: str | None,
      laterality: str | None,
  ) -> _NormalizedRsnaQuery:
    """Normalizes extracted procedure attributes to canonical search terms."""
    normalized_modality = normalize_attribute(
        modality_type, mappings.MODALITY_PATTERNS
    )
    normalized_subtype = normalize_attribute(
        modality_subtype, mappings.MODALITY_SUBTYPE_PATTERNS
    )
    canonical_region = normalize_attribute(
        region_imaged, mappings.REGION_PATTERNS
    )
    canonical_contrast = normalize_attribute(
        contrast_used, mappings.CONTRAST_PATTERNS
    )
    canonical_view = normalize_attribute(view_type, mappings.VIEW_PATTERNS)
    canonical_laterality = normalize_attribute(
        laterality, mappings.LATERALITY_PATTERNS
    )
    return _NormalizedRsnaQuery(
        modality=normalize_text(normalized_modality),
        subtype=normalize_text(normalized_subtype),
        region=normalize_text(canonical_region),
        contrast=normalize_text(canonical_contrast),
        view=normalize_text(canonical_view),
        laterality=normalize_text(canonical_laterality),
    )

  def _match_exact_tuple(
      self, query: _NormalizedRsnaQuery
  ) -> tuple[str | None, str | None]:
    """Tier 1: Exact Tuple Match in RSNA index."""
    exact_key = (
        query.modality,
        query.subtype,
        query.region,
        query.contrast,
        query.view,
        query.laterality,
    )
    if exact_key in self._tuple_index:
      matched_record = self._tuple_index[exact_key]
      return (
          matched_record[config.FIELD_LOINC_NUM],
          matched_record[config.FIELD_LONG_COMMON_NAME],
      )

    # If subtype wasn't specified but matched with empty subtype
    if not query.subtype:
      fallback_key = (
          query.modality,
          "",
          query.region,
          query.contrast,
          query.view,
          query.laterality,
      )
      if fallback_key in self._tuple_index:
        matched_record = self._tuple_index[fallback_key]
        return (
            matched_record[config.FIELD_LOINC_NUM],
            matched_record[config.FIELD_LONG_COMMON_NAME],
        )

    return None, None

  def _match_by_view(
      self, query: _NormalizedRsnaQuery
  ) -> tuple[str | None, str | None]:
    """Tier 1b: Match Modality (+ Subtype) + Region + View (Primary for XR)."""
    # 1. Exact view match first
    for record in self._records:
      if (
          record[_NORM_MODALITY] == query.modality
          and (not query.subtype or query.subtype == record[_NORM_SUBTYPE])
          and record[_NORM_REGION] == query.region
          and query.view == record[_NORM_VIEW]
          and (
              not query.laterality
              or record[_NORM_LATERALITY] == query.laterality
          )
      ):
        return (
            record[config.FIELD_LOINC_NUM],
            record[config.FIELD_LONG_COMMON_NAME],
        )

    # 2. Partial view match fallback
    for record in self._records:
      record_region = record[_NORM_REGION]
      record_view = record[_NORM_VIEW]
      if (
          record[_NORM_MODALITY] == query.modality
          and (not query.subtype or query.subtype == record[_NORM_SUBTYPE])
          and (
              record_region == query.region
              or query.region in record_region
              or record_region in query.region
          )
          and (query.view in record_view or record_view in query_view)
          and (
              not query.laterality
              or record[_NORM_LATERALITY] == query.laterality
          )
      ):
        return (
            record[config.FIELD_LOINC_NUM],
            record[config.FIELD_LONG_COMMON_NAME],
        )

    return None, None

  def _match_by_contrast(
      self, query: _NormalizedRsnaQuery
  ) -> tuple[str | None, str | None]:
    """Tier 1c: Match Modality (+ Subtype) + Region + Contrast (Primary for CT/MR)."""
    for record in self._records:
      record_region = record[_NORM_REGION]
      record_contrast = record[_NORM_CONTRAST]
      if (
          record[_NORM_MODALITY] == query.modality
          and (not query.subtype or query.subtype == record[_NORM_SUBTYPE])
          and (
              record_region == query.region
              or query.region in record_region
              or record_region in query.region
          )
          and (
              query.contrast in record_contrast
              or record_contrast in query.contrast
          )
          and (
              not query.laterality
              or record[_NORM_LATERALITY] == query.laterality
          )
      ):
        return (
            record[config.FIELD_LOINC_NUM],
            record[config.FIELD_LONG_COMMON_NAME],
        )
    return None, None

  def _match_by_similarity_score(
      self, query: _NormalizedRsnaQuery, exam_name: str
  ) -> tuple[str | None, str | None]:
    """Tier 2: Candidate filtering by modality + region + text similarity scoring."""
    candidates = []
    for record in self._records:
      record_modality = record[_NORM_MODALITY]
      record_subtype = record[_NORM_SUBTYPE]
      record_region = record[_NORM_REGION]
      if query.modality and query.modality != record_modality:
        continue
      if query.subtype and query.subtype != record_subtype:
        continue
      if query.region and (
          not record_region
          or (
              query.region not in record_region
              and record_region not in query.region
          )
      ):
        continue
      candidates.append(record)

    if not candidates:
      return None, None

    query_exam_name = normalize_text(exam_name)
    best_candidate = None
    best_score = -1.0
    for candidate in candidates:
      candidate_search_text = candidate.get(config.FIELD_SEARCH_TEXT, "")
      # Compute token overlap score
      score = float(
          sum(
              1
              for word in query_exam_name.split()
              if word in candidate_search_text
          )
      )
      # Prioritize lower COMMON_TEST_RANK if available
      common_rank_str = candidate.get(config.FIELD_COMMON_TEST_RANK, "")
      if common_rank_str and common_rank_str.isdigit():
        score += 1.0 / (int(common_rank_str) + 1)
      if score > best_score:
        best_score = score
        best_candidate = candidate

    if best_candidate:
      return (
          best_candidate[config.FIELD_LOINC_NUM],
          best_candidate[config.FIELD_LONG_COMMON_NAME],
      )
    return None, None

  def _match_by_modality_fallback(
      self, query_modality: str
  ) -> tuple[str | None, str | None]:
    """Tier 3: Broadest fallback match by modality."""
    for record in self._records:
      if record[_NORM_MODALITY] == query_modality:
        return (
            record[config.FIELD_LOINC_NUM],
            record[config.FIELD_LONG_COMMON_NAME],
        )
    return None, None

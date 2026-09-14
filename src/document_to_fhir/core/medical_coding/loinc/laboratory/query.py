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
"""Engine for querying the LOINC Knowledgebase based on lab test criteria."""

import abc
import math

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import index as core_analyte_index_lib
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import mapper as property_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.scale_type import mapper as scale_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system import mapper as systems_mapper
from src.document_to_fhir.core.medical_coding.loinc.common import schema
from src.document_to_fhir.core.medical_coding.loinc.common import utils

_DEFAULT_RANK = 99999


def _get_rank(record: schema.LoincRow) -> int:
  """Returns integer LOINC rank or default rank if missing/invalid."""
  if record.common_test_rank is not None and int(record.common_test_rank) > 0:
    return int(record.common_test_rank)
  return _DEFAULT_RANK


class BaseLOINCQueryEngine(abc.ABC):
  """Abstract base class for LOINC query engines."""

  @abc.abstractmethod
  def query(self, lab_test: resources.LabTest) -> list[schema.LoincRow]:
    """Queries LOINC codes based on lab test criteria."""

  @abc.abstractmethod
  def query_batch(
      self, lab_tests: list[resources.LabTest]
  ) -> list[list[schema.LoincRow]]:
    """Queries LOINC codes for a batch of lab tests."""


class LaboratoryLoincQueryEngine(BaseLOINCQueryEngine):
  """Responsible for searching the AnalyteIndex.

  Handles query validation, filtering logic, and ranking.
  Implements 'Soft Filtering': filters are applied only if they do not
  reduce the result set to zero.
  """

  def __init__(
      self,
      analyte_index: core_analyte_index_lib.AnalytesIndex,
      system_mapper: systems_mapper.SpecimenToSystemMapper | None = None,
      scale_mapper: scale_mapper_lib.ScaleMapper | None = None,
      property_mapper: property_mapper_lib.UnitToPropertyMapper | None = None,
  ):
    """Initializes the query engine.

    Args:
      analyte_index: An instance of AnalytesIndex.
      system_mapper: Optional instance of SpecimenToSystemMapper for system
        filtering.
      scale_mapper: Optional instance of ScaleMapper for scale filtering.
      property_mapper: Optional instance of UnitToPropertyMapper for property
        filtering.
    """
    self.analyte_index = analyte_index
    self.system_mapper = system_mapper
    self.scale_mapper = scale_mapper
    self.property_mapper = property_mapper

  def query(self, lab_test: resources.LabTest) -> list[schema.LoincRow]:
    """Queries the AnalyteIndex for LOINC codes based on a lab test.

    Performs an exact match search using the 'core_analyte' from the
    `lab_test` and then applies filters and ranking.

    Args:
      lab_test: A `resources.LabTest` object representing a lab test.

    Returns:
      A list of schema.LoincRow objects representing filtered and ranked LOINC
      records.
    """
    input_term = lab_test.core_analyte
    if not input_term:
      return []

    # Search by signature, handles case-insensitivity and word order
    candidates = self.analyte_index.search_by_analyte(input_term)

    if not candidates:
      return []

    filtered_records = self._apply_filters_and_rank(candidates, lab_test)
    return filtered_records

  def query_batch(
      self, lab_tests: list[resources.LabTest]
  ) -> list[list[schema.LoincRow]]:
    """Queries LOINC codes for a batch of lab tests sequentially."""
    return [self.query(test) for test in lab_tests]

  def _apply_filters_and_rank(
      self,
      records: list[schema.LoincRow],
      criteria: resources.LabTest,
  ) -> list[schema.LoincRow]:
    """Applies filters and ranks LOINC records based on criteria.

    Args:
      records: A list of schema.LoincRow objects.
      criteria: A LabTest object containing filtering criteria like 'name' and
        'specimen'.

    Returns:
      A list of filtered and ranked schema.LoincRow objects.
    """
    # 1. Apply System Filter using mapper.py if available
    raw_specimen = criteria.specimen
    if raw_specimen and self.system_mapper:
      # Get valid canonical systems from the mapper
      canonical_systems = self.system_mapper.get_canonical_systems(raw_specimen)
      if canonical_systems:
        # Filter candidates directly based on the 'SYSTEM' field
        filtered_records = [
            rec for rec in records if rec.system in canonical_systems
        ]
        # Soft Filtering: only use filtered records if not empty
        if filtered_records:
          records = filtered_records

    # 2. Apply Scale Filter (Soft Filter)
    raw_result = criteria.result
    if raw_result and self.scale_mapper:
      canonical_scales = self.scale_mapper.get_canonical_scales(raw_result)
      if canonical_scales:
        filtered_by_scale = [
            rec for rec in records if rec.scale_typ in canonical_scales
        ]
        if filtered_by_scale:
          records = filtered_by_scale

    # 3. Apply Property Filter (Soft Filter) based on Unit
    raw_unit = criteria.unit
    if raw_unit and self.property_mapper:
      canonical_properties = self.property_mapper.get_canonical_properties(
          raw_unit
      )
      if canonical_properties:
        filtered_by_property = [
            rec for rec in records if rec.property in canonical_properties
        ]
        if filtered_by_property:
          records = filtered_by_property

    # 4. Rank candidates
    return self._rank_candidates(records, criteria.name)

  def _rank_candidates(
      self, records: list[schema.LoincRow], raw_test_name: str
  ) -> list[schema.LoincRow]:
    """Sorts candidates by fuzzy token overlap with raw_test_name and damped log-rank prior."""
    if not raw_test_name:
      # If no test name, just sort by rank ascending
      return sorted(records, key=_get_rank)

    query_tokens = utils.StringUtils.tokenize(raw_test_name)
    num_query_tokens = max(1, len(query_tokens))
    # Bounded scaling factor guarantees rank_prior < 1 / num_query_tokens,
    # ensuring rank acts strictly as a tie-breaker regardless of query length.
    max_rank_boost = 0.05 / (num_query_tokens + 1)

    scored_records = []

    for rec in records:
      target_text = (
          f"{rec.core_analyte or ''} {rec.long_common_name or ''}".strip()
      )
      overlap = utils.StringUtils.compute_fuzzy_token_overlap(
          raw_test_name, target_text
      )
      rank = _get_rank(rec)
      rank_prior = max_rank_boost / math.log2(1 + rank)
      total_score = overlap + rank_prior
      scored_records.append((total_score, rank, rec))

    # Sort by total_score descending, tie-breaking by rank ascending
    scored_records.sort(key=lambda x: (-x[0], x[1]))

    return [item[2] for item in scored_records]

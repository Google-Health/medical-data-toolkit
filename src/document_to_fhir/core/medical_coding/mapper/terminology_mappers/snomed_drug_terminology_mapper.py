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
"""Terminology mapper for SNOMED drug codes."""

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.mapper import terminology_mapper
from src.document_to_fhir.core.medical_coding.snomed.drug.query import snomed_drug_query_engine


class SnomedDrugTerminologyMapper(terminology_mapper.ITerminologyMapper):
  """Terminology mapper for SNOMED drug codes."""

  def __init__(
      self,
      query_engine: snomed_drug_query_engine.SnomedDrugQueryEngine,
  ):
    self.query_engine = query_engine

  def map_inplace(self, medication: resources.Medication):
    """Maps a medication to a SNOMED drug code."""
    if medication.medication_code:
      return
    result = self.query_engine.query(medication)
    medication.medication_code = result.code
    medication.medication_code_name = result.display_name

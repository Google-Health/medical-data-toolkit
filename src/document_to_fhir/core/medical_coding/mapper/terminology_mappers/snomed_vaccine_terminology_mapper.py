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
"""Terminology mapper for SNOMED vaccine codes."""

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.mapper import terminology_mapper
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import snomed_vaccine_query_engine


class SnomedVaccineTerminologyMapper(terminology_mapper.ITerminologyMapper):
  """Terminology mapper for SNOMED vaccine codes."""

  def __init__(
      self,
      query_engine: snomed_vaccine_query_engine.SnomedVaccineQueryEngine,
  ):
    """Initializes the terminology mapper with a query engine."""
    self.query_engine = query_engine

  def map_inplace(self, immunization: resources.Immunization) -> None:
    """Maps an immunization to a SNOMED vaccine code."""
    if immunization.vaccine_code:
      return
    result = self.query_engine.query(immunization)
    immunization.vaccine_code = result.code
    immunization.vaccine_code_name = result.display_name

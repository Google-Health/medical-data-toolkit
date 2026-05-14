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
"""Terminology mapper for LOINC codes."""

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc import query
from src.document_to_fhir.core.medical_coding.mapper import terminology_mapper


class LoincTerminologyMapper(terminology_mapper.ITerminologyMapper):
  """Maps lab tests to LOINC codes using a LoincQueryEngine."""

  def __init__(self, query_engine: query.LoincQueryEngine):
    self.query_engine = query_engine

  def map_inplace(self, test: resources.LabTest):
    if test.loinc_code:
      return
    results = self.query_engine.query(test)
    if results:
      test.loinc_code = results[0].loinc_num
      test.loinc_common_name = results[0].long_common_name
    else:
      test.loinc_code = None
      test.loinc_common_name = None

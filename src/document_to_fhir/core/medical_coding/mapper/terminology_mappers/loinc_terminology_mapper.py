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

from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc.laboratory import query as lab_query
from src.document_to_fhir.core.medical_coding.loinc.radiology import query as rsna_query
from src.document_to_fhir.core.medical_coding.mapper import terminology_mapper


class LoincTerminologyMapper(terminology_mapper.ITerminologyMapper):
  """Maps lab tests and radiology reports to LOINC codes."""

  def __init__(
      self,
      lab_query_engine: lab_query.LaboratoryLoincQueryEngine,
      radiology_query_engine: rsna_query.RsnaPlaybookQueryEngine,
  ):
    self.lab_query_engine = lab_query_engine
    self.radiology_query_engine = radiology_query_engine

  def map_inplace(self, model: resources.MedicalData) -> None:
    """Maps LOINC codes in-place on the provided medical data model."""
    if isinstance(model, resources.LabTest):
      self.map_lab_test(model)
    elif isinstance(model, medical_documents.LabReport):
      self.map_lab_report(model)
    elif isinstance(model, medical_documents.RadiologyReport):
      self.map_radiology_report(model)

  def map_lab_report(self, report: medical_documents.LabReport) -> None:
    """Maps LOINC codes in-place for all lab tests within a LabReport."""
    if report.lab_tests:
      for lab_test in report.lab_tests:
        self.map_lab_test(lab_test)

  def map_lab_test(self, lab_test: resources.LabTest) -> None:
    """Maps LOINC code in-place on a single LabTest."""
    rows = self.lab_query_engine.query(lab_test)
    if rows:
      lab_test.loinc_code = rows[0].loinc_num
      lab_test.loinc_common_name = rows[0].long_common_name

  def map_radiology_report(
      self, report: medical_documents.RadiologyReport
  ) -> None:
    """Maps procedure LOINC code in-place on a RadiologyReport."""
    self.radiology_query_engine.map_report(report)

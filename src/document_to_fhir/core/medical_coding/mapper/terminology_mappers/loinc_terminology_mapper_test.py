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
"""Unit tests for LOINC terminology mapper."""

from unittest import mock

from absl.testing import absltest

from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc.common import schema as loinc_schema
from src.document_to_fhir.core.medical_coding.loinc.laboratory import query as lab_query
from src.document_to_fhir.core.medical_coding.loinc.radiology import config
from src.document_to_fhir.core.medical_coding.loinc.radiology import query as rsna_query
from src.document_to_fhir.core.medical_coding.mapper.terminology_mappers import loinc_terminology_mapper

TEST_RADIOLOGY_RECORDS = [{
    config.FIELD_LOINC_NUM: "30932-8",
    config.FIELD_LONG_COMMON_NAME: "CT Chest with contrast",
    config.FIELD_MODALITY_TYPE: "CT",
    config.FIELD_MODALITY_SUBTYPE: "",
    config.FIELD_REGION_IMAGED: "Chest",
    config.FIELD_CONTRAST_USED: "W contrast",
    config.FIELD_VIEW_TYPE: "",
    config.FIELD_LATERALITY: "",
    config.FIELD_SEARCH_TEXT: "ct chest with contrast",
}]


class LoincTerminologyMapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_lab_engine = mock.MagicMock(
        spec=lab_query.LaboratoryLoincQueryEngine
    )
    self.radiology_engine = rsna_query.RsnaPlaybookQueryEngine(
        TEST_RADIOLOGY_RECORDS
    )
    self.mapper = loinc_terminology_mapper.LoincTerminologyMapper(
        lab_query_engine=self.mock_lab_engine,
        radiology_query_engine=self.radiology_engine,
    )

  def test_map_radiology_report_rsna_matching(self):
    report = medical_documents.RadiologyReport(
        exam_name="CT Chest with contrast",
        modality_type="CT",
        region_imaged="Chest",
        contrast_used="W contrast",
        findings="Lungs are clear without focal consolidation.",
        conclusion="Normal CT examination of the chest.",
        patient=resources.Patient(name="John Doe"),
    )
    self.mapper.map_inplace(report)

    self.assertEqual(report.procedure_loinc_code, "30932-8")

  def test_map_lab_test_with_query_engine(self):
    mock_row = loinc_schema.LoincRow(
        loinc_num="718-7",
        core_analyte="Hemoglobin",
        property="MCnc",
        system="Bld",
        scale_typ="Qn",
        long_common_name="Hemoglobin [Mass/volume] in Blood",
    )
    self.mock_lab_engine.query.return_value = [mock_row]

    test = resources.LabTest(
        core_analyte="Hemoglobin", name="Hemoglobin", result="14.2", unit="g/dL"
    )
    self.mapper.map_inplace(test)
    self.assertEqual(test.loinc_code, "718-7")
    self.assertEqual(
        test.loinc_common_name, "Hemoglobin [Mass/volume] in Blood"
    )


if __name__ == "__main__":
  absltest.main()

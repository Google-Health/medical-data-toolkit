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
"""Unit tests for snomed_vaccine_terminology_mapper."""

from unittest import mock

from absl.testing import absltest

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.mapper.terminology_mappers import snomed_vaccine_terminology_mapper
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import base
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import snomed_vaccine_query_engine


class SnomedVaccineTerminologyMapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_query_engine = mock.create_autospec(
        snomed_vaccine_query_engine.SnomedVaccineQueryEngine, instance=True
    )
    self.mapper = (
        snomed_vaccine_terminology_mapper.SnomedVaccineTerminologyMapper(
            query_engine=self.mock_query_engine
        )
    )

  def test_map_inplace_maps_code_and_display_name(self):
    self.mock_query_engine.query.return_value = base.VaccineResult(
        code="1119349007",
        display_name="Covaxin vaccine",
        code_system="SNOMED-CT",
    )
    immunization = resources.Immunization(
        dosage_reason="",
        vaccine_name="Covaxin",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )

    self.mapper.map_inplace(immunization)

    self.mock_query_engine.query.assert_called_once_with(immunization)
    self.assertEqual(immunization.vaccine_code, "1119349007")
    self.assertEqual(immunization.vaccine_code_name, "Covaxin vaccine")

  def test_map_inplace_skips_when_already_coded(self):
    immunization = resources.Immunization(
        dosage_reason="",
        vaccine_name="Covaxin",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
        vaccine_code="1119349007",
        vaccine_code_name="Covaxin vaccine",
    )

    self.mapper.map_inplace(immunization)

    self.mock_query_engine.query.assert_not_called()
    self.assertEqual(immunization.vaccine_code, "1119349007")
    self.assertEqual(immunization.vaccine_code_name, "Covaxin vaccine")

  def test_map_inplace_handles_no_match(self):
    self.mock_query_engine.query.return_value = base.VaccineResult(
        code=None,
        display_name=None,
        code_system=None,
    )
    immunization = resources.Immunization(
        dosage_reason="",
        vaccine_name="Unknown Vaccine",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )

    self.mapper.map_inplace(immunization)

    self.mock_query_engine.query.assert_called_once_with(immunization)
    self.assertIsNone(immunization.vaccine_code)
    self.assertIsNone(immunization.vaccine_code_name)


if __name__ == "__main__":
  absltest.main()

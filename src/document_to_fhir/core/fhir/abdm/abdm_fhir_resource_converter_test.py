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
import datetime

from absl.testing import absltest
from absl.testing import parameterized

from google.fhir.r4.proto.core import datatypes_pb2
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir.abdm import abdm_fhir_resource_converter


class AbdmFhirResourceConverterTest(parameterized.TestCase):

  def test_create_patient(self):
    patient_data = resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
        dob=datetime.date(1990, 1, 1),
        gender="male",
    )
    patient_fhir = abdm_fhir_resource_converter.create_patient(
        patient_data, "patient-id"
    )
    self.assertEqual(patient_fhir.id.value, "patient-id")
    self.assertEqual(
        patient_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient",
    )
    self.assertEqual(patient_fhir.identifier[0].value.value, "MR123")

  def test_create_patient_missing_identifier(self):
    patient_data = resources.Patient(
        name="John Doe",
        dob=datetime.date(1990, 1, 1),
        gender="male",
    )
    patient_fhir = abdm_fhir_resource_converter.create_patient(
        patient_data, "patient-id"
    )
    self.assertEqual(patient_fhir.id.value, "patient-id")
    self.assertEqual(patient_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        patient_fhir.identifier[0].system.value, "http://hospital.smarthealthit.org"
    )

  def test_create_practitioner(self):
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD123"),
    )
    practitioner_fhir = abdm_fhir_resource_converter.create_practitioner(
        practitioner_data, "practitioner-id"
    )
    self.assertEqual(practitioner_fhir.id.value, "practitioner-id")
    self.assertEqual(
        practitioner_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner",
    )

  def test_create_practitioner_missing_identifier(self):
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
    )
    practitioner_fhir = abdm_fhir_resource_converter.create_practitioner(
        practitioner_data, "practitioner-id"
    )
    self.assertEqual(practitioner_fhir.id.value, "practitioner-id")
    self.assertEqual(practitioner_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        practitioner_fhir.identifier[0].system.value, "https://doctor.ndhm.gov.in"
    )

  def test_create_organization(self):
    org_data = resources.Organization(
        name="General Hospital",
        identifiers=abdm_resources.AbdmOrganizationIdentifiers(prn="PRNORG"),
    )
    org_fhir = abdm_fhir_resource_converter.create_organization(
        org_data, "org-id"
    )
    self.assertEqual(org_fhir.id.value, "org-id")
    self.assertEqual(
        org_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Organization",
    )
    self.assertEqual(org_fhir.identifier[0].value.value, "PRNORG")

  def test_create_organization_missing_identifier(self):
    org_data = resources.Organization(
        name="General Hospital",
    )
    org_fhir = abdm_fhir_resource_converter.create_organization(
        org_data, "org-id"
    )
    self.assertEqual(org_fhir.id.value, "org-id")
    self.assertEqual(org_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        org_fhir.identifier[0].system.value, "https://facility.ndhm.gov.in"
    )

  def test_create_encounter(self):
    start_dt = datatypes_pb2.DateTime(value_us=12345)
    encounter_fhir = abdm_fhir_resource_converter.create_encounter(
        "encounter-id", "Patient/patient-id", start_dt
    )
    self.assertEqual(encounter_fhir.id.value, "encounter-id")
    self.assertEqual(encounter_fhir.subject.uri.value, "Patient/patient-id")

  def test_create_lab_observation(self):
    lab_test = resources.LabTest(
        core_analyte="Glucose",
        name="Glucose",
        result="100",
        unit="mg/dL",
        loinc_code="2345-7",
    )
    effective_dt = datatypes_pb2.DateTime(value_us=12345)
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, effective_dt
    )
    self.assertEqual(obs_fhir.id.value, "obs-id")
    self.assertEqual(
        obs_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation",
    )

  def test_create_diagnostic_report(self):
    effective_dt = datatypes_pb2.DateTime(value_us=12345)
    report_fhir = abdm_fhir_resource_converter.create_diagnostic_report(
        "report-id",
        "Patient/patient-id",
        "Practitioner/prac-id",
        "Organization/org-id",
        "Encounter/enc-id",
        ["Observation/obs-id"],
        ["Specimen/spec-id"],
        effective_dt,
    )
    self.assertEqual(report_fhir.id.value, "report-id")
    self.assertEqual(
        report_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportLab",
    )

  @parameterized.named_parameters(
      (
          "zero_prefix_normalization",
          "Eosinophils",
          "Absolute Eosinophil Count",
          "00",
          "cells/mcL",
          [resources.ReferenceRange(low="00", high="450", text="00-450")],
          "0",
          "0",
          "450",
      ),
      (
          "threshold_and_decimal_normalization",
          "Glucose",
          "Glucose",
          "01.0",
          "mg/dL",
          [resources.ReferenceRange(high="01", text="< 01")],
          "1.0",
          None,
          "1",
      ),
  )
  def test_create_lab_observation_normalization(
      self,
      core_analyte,
      name,
      result,
      unit,
      reference_range,
      expected_value,
      expected_low,
      expected_high,
  ):
    lab_test = resources.LabTest(
        core_analyte=core_analyte,
        name=name,
        result=result,
        unit=unit,
        reference_range=reference_range,
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertEqual(obs_fhir.value.quantity.value.value, expected_value)
    self.assertLen(obs_fhir.reference_range, 1)
    if expected_low is not None:
      self.assertEqual(
          obs_fhir.reference_range[0].low.value.value, expected_low
      )
    else:
      self.assertFalse(obs_fhir.reference_range[0].low.HasField("value"))

    if expected_high is not None:
      self.assertEqual(
          obs_fhir.reference_range[0].high.value.value, expected_high
      )
    else:
      self.assertFalse(obs_fhir.reference_range[0].high.HasField("value"))

  @parameterized.named_parameters(
      ("unparseable_text", "M: 1 - 7 / F: 3 - 12", "M: 1 - 7 / F: 3 - 12"),
      ("invalid_decimals", "1.2.3-4.5.6", "1.2.3-4.5.6"),
  )
  def test_create_lab_observation_invalid_range_fallback(
      self, reference_range_str, expected_text
  ):
    lab_test = resources.LabTest(
        core_analyte="ESR",
        name="ESR",
        result="23",
        unit="mm",
        reference_range=[resources.ReferenceRange(text=reference_range_str)],
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertLen(obs_fhir.reference_range, 1)
    self.assertFalse(obs_fhir.reference_range[0].low.HasField("value"))
    self.assertFalse(obs_fhir.reference_range[0].high.HasField("value"))
    self.assertEqual(obs_fhir.reference_range[0].text.value, expected_text)

  @parameterized.named_parameters(
      ("empty_string", ""),
      ("whitespace_only", "   "),
  )
  def test_create_lab_observation_empty_result(self, result):
    lab_test = resources.LabTest(
        core_analyte="WBC",
        name="White Blood Cell Count",
        result=result,
        unit="Thousand/uL",
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertFalse(obs_fhir.HasField("value"))

  def test_create_lab_observation_reference_range_fields(self):
    lab_test = resources.LabTest(
        core_analyte="HbA1c",
        name="HbA1c",
        result="5.5",
        unit="%",
        reference_range=[
            resources.ReferenceRange(
                low="4.5",
                high="6.0",
                applies_to=resources.ReferenceRangeAppliesTo.MALE,
                age=resources.AgeRange(low=18.0, high=120.0),
                text="Male 18+ 4.5-6.0",
            )
        ],
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertLen(obs_fhir.reference_range, 1)
    rr = obs_fhir.reference_range[0]
    self.assertEqual(rr.low.value.value, "4.5")
    self.assertEqual(rr.high.value.value, "6.0")
    self.assertEqual(rr.text.value, "Male 18+ 4.5-6.0")
    self.assertLen(rr.applies_to, 1)
    self.assertEqual(rr.applies_to[0].text.value, "Male")
    self.assertEqual(rr.age.low.value.value, "18.0")
    self.assertEqual(rr.age.high.value.value, "120.0")
    self.assertEqual(rr.age.low.unit.value, "a")
    self.assertEqual(rr.age.high.unit.value, "a")


if __name__ == "__main__":
  absltest.main()

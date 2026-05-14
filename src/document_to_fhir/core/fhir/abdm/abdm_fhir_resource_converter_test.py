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

from google.fhir.r4.proto.core import datatypes_pb2
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir.abdm import abdm_fhir_resource_converter


class AbdmFhirResourceConverterTest(absltest.TestCase):

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


if __name__ == "__main__":
  absltest.main()

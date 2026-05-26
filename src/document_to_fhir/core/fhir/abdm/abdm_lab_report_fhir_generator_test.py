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
import json
import os
from unittest import mock
import uuid

from absl.testing import absltest

from google.fhir.r4.proto.core import codes_pb2
from google.fhir.r4.proto.core.resources import bundle_and_contained_resource_pb2
from google.fhir.r4 import json_format
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_medical_documents
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir import fhir_test_utils
from src.document_to_fhir.core.fhir.abdm import abdm_lab_report_fhir_generator

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data/lab_report")


class AbdmLabReportFhirGeneratorTest(
    absltest.TestCase
):

  def _create_default_patient(self) -> resources.Patient:
    return resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
        dob=None,
        gender="male",
    )

  def _create_default_organization(
      self,
  ) -> resources.Organization[abdm_resources.AbdmOrganizationIdentifiers]:
    return resources.Organization[abdm_resources.AbdmOrganizationIdentifiers](
        name="General Hospital",
        identifiers=None,
        address=None,
        contact=None,
    )

  def test_generate_fhir(self):
    generator = abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator()

    # Create mock medical document
    patient_data = resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
    )
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD123"),
    )
    org_data = resources.Organization(name="General Hospital")
    lab_test = resources.LabTest(
        core_analyte="Glucose",
        name="Glucose",
        result="100",
        unit="mg/dL",
        loinc_code="2345-7",
    )

    doc = abdm_medical_documents.AbdmLabReport(
        patient=patient_data,
        practitioner=practitioner_data,
        service_provider=org_data,
        lab_tests=[lab_test],
        sample_collection_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )

    bundle = generator.generate_fhir(doc)

    self.assertIsInstance(bundle, bundle_and_contained_resource_pb2.Bundle)
    self.assertEqual(bundle.type.value, codes_pb2.BundleTypeCode.DOCUMENT)
    self.assertEqual(
        bundle.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle",
    )

    # Verify first entry is Composition
    self.assertTrue(bundle.entry[0].resource.HasField("composition"))
    composition = bundle.entry[0].resource.composition
    self.assertEqual(composition.title.value, "Diagnostic Report Record")

  def test_generate_fhir_with_panel(self):
    generator = abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator()

    # Create mock medical document with panels
    patient_data = resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
    )
    lab_test1 = resources.LabTest(
        core_analyte="Glucose",
        name="Glucose",
        result="100",
        unit="mg/dL",
        loinc_code="2345-7",
        panel_name="Complete Blood Count",
        panel_loinc_code="58410-2",
    )
    lab_test2 = resources.LabTest(
        core_analyte="Hemoglobin",
        name="Hemoglobin",
        result="14",
        unit="g/dL",
        loinc_code="718-7",
        panel_name="Complete Blood Count",
        panel_loinc_code="58410-2",
    )
    lab_test_unpanelled = resources.LabTest(
        core_analyte="Cholesterol",
        name="Cholesterol",
        result="200",
        unit="mg/dL",
        loinc_code="2093-3",
    )

    doc = abdm_medical_documents.AbdmLabReport(
        patient=patient_data,
        lab_tests=[lab_test1, lab_test2, lab_test_unpanelled],
        sample_collection_time=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )

    bundle = generator.generate_fhir(doc)

    # Verify structure
    # 1 DiagnosticReport
    # 1 Panel Observation (Master)
    # 3 Leaf Observations (Glucose, Hemoglobin, Cholesterol)

    observations = []
    diagnostic_reports = []

    for entry in bundle.entry:
      if entry.resource.HasField("observation"):
        observations.append(entry.resource.observation)
      elif entry.resource.HasField("diagnostic_report"):
        diagnostic_reports.append(entry.resource.diagnostic_report)

    self.assertLen(diagnostic_reports, 1)
    self.assertLen(observations, 4) # 3 tests + 1 panel

    # Find the panel observation
    panel_obs = None
    for obs in observations:
      if obs.code.coding[0].code.value == "58410-2":
        panel_obs = obs
        break

    self.assertIsNotNone(panel_obs)
    self.assertLen(panel_obs.has_member, 2)

    # Verify DiagnosticReport references the panel and the unpanelled test
    report = diagnostic_reports[0]
    self.assertLen(report.result, 2) # Panel + Cholesterol

  def test_generate_fhir_missing_collection_time_date_fallback(self):
    # Mock datetime.datetime in the generator module
    fixed_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    class MockDateTime(datetime.datetime):

      @classmethod
      def now(cls, tz=None):
        return fixed_time

    patient_data = self._create_default_patient()
    org_data = self._create_default_organization()
    doc = abdm_medical_documents.AbdmLabReport(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        lab_tests=[],
        sample_collection_time=None,
    )

    generator = abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator()

    with mock.patch.object(
        abdm_lab_report_fhir_generator.datetime, "datetime", MockDateTime
    ):
      bundle = generator.generate_fhir(doc)

    comp = bundle.entry[0].resource.composition
    expected_us = int(fixed_time.timestamp() * 1_000_000)
    self.assertEqual(comp.date.value_us, expected_us)
    self.assertEqual(comp.date.timezone, "UTC")

  def test_generate_fhir_missing_practitioner_author_fallback(self):
    patient_data = self._create_default_patient()
    org_data = self._create_default_organization()
    doc = abdm_medical_documents.AbdmLabReport(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        lab_tests=[],
        sample_collection_time=datetime.datetime(
            2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
        ),
    )

    generator = abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator()
    bundle = generator.generate_fhir(doc)

    comp = bundle.entry[0].resource.composition
    self.assertLen(comp.author, 1)
    author_ref = comp.author[0].uri.value
    self.assertStartsWith(author_ref, "urn:uuid:")

    org_id = None
    for entry in bundle.entry:
      if entry.resource.HasField("organization"):
        org_id = entry.resource.organization.id.value
        break

    self.assertIsNotNone(org_id, "Organization not found in bundle")
    self.assertEqual(author_ref, f"urn:uuid:{org_id}")


if __name__ == "__main__":
  absltest.main()

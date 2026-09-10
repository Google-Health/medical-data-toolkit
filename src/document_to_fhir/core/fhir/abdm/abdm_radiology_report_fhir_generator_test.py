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
"""Tests for verifying ABDM FHIR generation for radiology reports."""

import datetime
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
from src.document_to_fhir.core.fhir.abdm import abdm_radiology_report_fhir_generator

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data/radiology_report")


class AbdmRadiologyReportFhirGeneratorTest(
    absltest.TestCase,
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

  def test_generate_fhir_success(self):
    generator = (
        abdm_radiology_report_fhir_generator.AbdmRadiologyReportFhirGenerator()
    )

    patient_data = self._create_default_patient()
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD123"),
    )
    org_data = self._create_default_organization()

    doc = abdm_medical_documents.AbdmRadiologyReport(
        patient=patient_data,
        practitioner=practitioner_data,
        service_provider=org_data,
        exam_name="CT Brain",
        modality_type="CT",
        findings="No abnormalities detected.",
        conclusion="Normal CT Brain.",
        study_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
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
    self.assertEqual(
        composition.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportRecord",
    )

    # Verify DiagnosticReport is present and has correct profile
    report = None
    for entry in bundle.entry:
      if entry.resource.HasField("diagnostic_report"):
        report = entry.resource.diagnostic_report
        break

    self.assertIsNotNone(report)
    self.assertEqual(
        report.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportImaging",
    )
    self.assertEqual(report.code.text.value, "CT Brain")
    self.assertEqual(
        report.conclusion.value,
        "Findings: No abnormalities detected.\nConclusion: Normal CT Brain.",
    )

    # Verify category is SNOMED CT for service
    self.assertLen(report.category, 1)
    category_coding = report.category[0].coding[0]
    self.assertEqual(category_coding.system.value, "http://snomed.info/sct")
    self.assertEqual(
        category_coding.code.value, "721981007"
    )  # Diagnostic studies report

    # Verify Patient is present
    patient = None
    for entry in bundle.entry:
      if entry.resource.HasField("patient"):
        patient = entry.resource.patient
        break
    self.assertIsNotNone(patient)
    patient_ref = f"urn:uuid:{patient.id.value}"

    # Verify Practitioner is present
    practitioner = None
    for entry in bundle.entry:
      if entry.resource.HasField("practitioner"):
        practitioner = entry.resource.practitioner
        break
    self.assertIsNotNone(practitioner)
    self.assertEqual(practitioner.name[0].text.value, "Dr. Smith")
    practitioner_ref = f"urn:uuid:{practitioner.id.value}"

    # Verify Organization is present
    organization = None
    for entry in bundle.entry:
      if entry.resource.HasField("organization"):
        organization = entry.resource.organization
        break
    self.assertIsNotNone(organization)
    self.assertEqual(organization.name.value, "General Hospital")
    organization_ref = f"urn:uuid:{organization.id.value}"

    # Verify Encounter is present and has correct references
    encounter = None
    for entry in bundle.entry:
      if entry.resource.HasField("encounter"):
        encounter = entry.resource.encounter
        break
    self.assertIsNotNone(encounter)
    self.assertEqual(encounter.subject.uri.value, patient_ref)
    self.assertEqual(
        encounter.participant[0].individual.uri.value, practitioner_ref
    )
    self.assertEqual(encounter.service_provider.uri.value, organization_ref)

    # Verify encounter period start matches study_date
    assert doc.study_date is not None
    expected_study_date_us = int(doc.study_date.timestamp() * 1_000_000)
    self.assertEqual(encounter.period.start.value_us, expected_study_date_us)

    # Verify Composition Author
    self.assertLen(composition.author, 1)
    self.assertEqual(composition.author[0].uri.value, practitioner_ref)

    # Verify Bundle ID and Identifier
    self.assertTrue(bundle.id.value)
    self.assertTrue(bundle.identifier.value.value)
    self.assertEqual(bundle.id.value, bundle.identifier.value.value)

  def test_generate_fhir_with_loinc_and_modality_subtype(self):
    generator = (
        abdm_radiology_report_fhir_generator.AbdmRadiologyReportFhirGenerator()
    )

    patient_data = self._create_default_patient()
    org_data = self._create_default_organization()

    doc = abdm_medical_documents.AbdmRadiologyReport(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        exam_name="CTA Brain",
        modality_type="CT",
        modality_subtype="Angio",
        region_imaged="Head",
        contrast_used="W contrast",
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name="CT Head and Neck WO contrast",
        findings="No abnormalities detected.",
        conclusion="Normal CTA Brain.",
        study_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )

    bundle = generator.generate_fhir(doc)

    report = None
    for entry in bundle.entry:
      if entry.resource.HasField("diagnostic_report"):
        report = entry.resource.diagnostic_report
        break

    self.assertIsNotNone(report)
    self.assertEqual(report.code.text.value, "CTA Brain")
    self.assertLen(report.code.coding, 1)
    self.assertEqual(report.code.coding[0].system.value, "http://loinc.org")
    self.assertEqual(report.code.coding[0].code.value, "82692-5")
    self.assertEqual(
        report.code.coding[0].display.value,
        "CT Head and Neck WO contrast",
    )
    self.assertEqual(report.category[0].text.value, "CT Angio")

  def test_generate_fhir_with_loinc_no_common_name(self):
    generator = (
        abdm_radiology_report_fhir_generator.AbdmRadiologyReportFhirGenerator()
    )

    patient_data = self._create_default_patient()
    org_data = self._create_default_organization()

    doc = abdm_medical_documents.AbdmRadiologyReport(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        exam_name="CT Brain",
        modality_type="CT",
        modality_subtype=None,
        region_imaged="Head",
        contrast_used=None,
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name=None,
        findings="Normal.",
        conclusion="Normal.",
        study_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )

    bundle = generator.generate_fhir(doc)

    report = None
    for entry in bundle.entry:
      if entry.resource.HasField("diagnostic_report"):
        report = entry.resource.diagnostic_report
        break

    self.assertIsNotNone(report)
    self.assertEqual(report.code.text.value, "CT Brain")
    self.assertLen(report.code.coding, 1)
    self.assertEqual(report.code.coding[0].system.value, "http://loinc.org")
    self.assertEqual(report.code.coding[0].code.value, "82692-5")
    self.assertEqual(
        report.code.coding[0].display.value,
        "CT Brain",
    )
    self.assertEqual(report.category[0].text.value, "CT")

  def test_generate_fhir_missing_study_date_date_fallback(self):
    fixed_time = datetime.datetime(
        2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc
    )

    class MockDateTime(datetime.datetime):

      @classmethod
      def now(cls, tz=None):
        return fixed_time

    patient_data = self._create_default_patient()
    org_data = self._create_default_organization()
    doc = abdm_medical_documents.AbdmRadiologyReport(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        exam_name="Chest X-Ray",
        study_date=None,
    )

    generator = (
        abdm_radiology_report_fhir_generator.AbdmRadiologyReportFhirGenerator()
    )

    with mock.patch.object(
        abdm_radiology_report_fhir_generator.datetime, "datetime", MockDateTime
    ):
      bundle = generator.generate_fhir(doc)

    comp = bundle.entry[0].resource.composition
    expected_us = int(fixed_time.timestamp() * 1_000_000)
    self.assertEqual(comp.date.value_us, expected_us)
    self.assertEqual(comp.date.timezone, "UTC")


if __name__ == "__main__":
  absltest.main()

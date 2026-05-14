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

if __name__ == "__main__":
  absltest.main()

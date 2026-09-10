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
from src.document_to_fhir.core.fhir.abdm import abdm_prescription_fhir_generator

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data/prescription")


class AbdmPrescriptionFhirGeneratorTest(
    absltest.TestCase,
):

  def test_generate_fhir(self):
    generator = abdm_prescription_fhir_generator.AbdmPrescriptionFhirGenerator()

    patient_data = resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
    )
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD123"),
    )
    org_data = resources.Organization(name="General Hospital")
    medication = resources.Medication(
        name="Azithromycin 250 mg Tablet",
        brand_name="Azithromycin",
        strength="250 mg",
        medicine_form="Tablet",
        medication_code="1145423002",
        medication_code_name="Azithromycin 250 mg oral tablet",
        dosage_instructions=resources.Dosage(
            instruction="Take one tablet daily in the morning",
            dose_value=1.0,
            dose_unit="TAB",
            frequency=1,
            frequency_period=1.0,
            frequency_period_unit="d",
        ),
        route="Oral",
    )

    doc = abdm_medical_documents.AbdmPrescription(
        patient=patient_data,
        practitioner=practitioner_data,
        service_provider=org_data,
        medications=[medication],
        prescription_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
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
    self.assertEqual(composition.title.value, "Prescription record")
    self.assertEqual(
        composition.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/PrescriptionRecord",
    )

    # Verify MedicationRequest entry is present
    med_entries = [
        e for e in bundle.entry if e.resource.HasField("medication_request")
    ]
    self.assertLen(med_entries, 1)
    med_req = med_entries[0].resource.medication_request
    self.assertEqual(
        med_req.medication.codeable_concept.text.value,
        "Azithromycin 250 mg oral tablet",
    )

  def test_generate_fhir_missing_patient(self):
    generator = abdm_prescription_fhir_generator.AbdmPrescriptionFhirGenerator()
    doc = abdm_medical_documents.AbdmPrescription.model_construct(
        patient=None,
        medications=[],
        prescription_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )
    with self.assertRaises(ValueError):
      generator.generate_fhir(doc)

  def test_generate_fhir_invalid_type(self):
    generator = abdm_prescription_fhir_generator.AbdmPrescriptionFhirGenerator()
    with self.assertRaises(TypeError):
      generator.generate_fhir(mock.MagicMock())

  def test_generate_fhir_missing_date_fallback(self):
    generator = abdm_prescription_fhir_generator.AbdmPrescriptionFhirGenerator()
    patient_data = resources.Patient(name="John Doe")
    doc = abdm_medical_documents.AbdmPrescription.model_construct(
        patient=patient_data,
        medications=[],
        prescription_date=None,
    )
    bundle = generator.generate_fhir(doc)
    self.assertTrue(bundle.entry[0].resource.HasField("composition"))

  def test_generate_fhir_missing_practitioner_author_fallback(self):
    generator = abdm_prescription_fhir_generator.AbdmPrescriptionFhirGenerator()
    patient_data = resources.Patient(name="John Doe")
    org_data = resources.Organization(name="General Hospital")
    doc = abdm_medical_documents.AbdmPrescription(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        medications=[],
        prescription_date=datetime.datetime(2026, 1, 1, 10, 0, 0),
    )
    bundle = generator.generate_fhir(doc)
    composition = bundle.entry[0].resource.composition
    self.assertLen(composition.author, 1)


if __name__ == "__main__":
  absltest.main()

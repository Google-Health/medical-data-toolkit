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
from src.document_to_fhir.core.fhir.abdm import abdm_immunization_fhir_generator

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), "data/immunization")


class AbdmImmunizationFhirGeneratorTest(
    absltest.TestCase,
):

  def test_generate_fhir(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()

    patient_data = resources.Patient(
        name="Jane Doe",
        identifiers=resources.PatientIdentifiers(mr="MR456"),
    )
    practitioner_data = resources.Practitioner(
        name="Dr. Patel",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD456"),
    )
    org_data = resources.Organization(name="City Hospital")
    imm_1 = resources.Immunization(
        vaccine_name="Covaxin",
        date=datetime.date(2026, 1, 15),
        vaccine_code="1119349007",
        dosage="1"
    )
    imm_2 = resources.Immunization(
        vaccine_name="Polio Vaccine",
        date=datetime.date(2026, 2, 1),
        dosage="1",
    )

    doc = abdm_medical_documents.AbdmImmunizationRecord(
        patient=patient_data,
        practitioner=practitioner_data,
        service_provider=org_data,
        immunizations=[imm_1, imm_2],
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
    self.assertEqual(composition.title.value, "Immunization record")
    self.assertEqual(
        composition.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/ImmunizationRecord",
    )
    self.assertEqual(
        composition.type.coding[0].system.value,
        "http://snomed.info/sct",
    )
    self.assertEqual(
        composition.type.coding[0].code.value,
        "41000179103",
    )
    self.assertEqual(
        composition.type.coding[0].display.value,
        "Immunization record",
    )

    # Verify composition section
    self.assertLen(composition.section, 1)
    self.assertEqual(composition.section[0].title.value, "Immunization record")
    self.assertLen(composition.section[0].entry, 2)
    self.assertEqual(composition.section[0].entry[0].type.value, "Immunization")
    self.assertEqual(composition.section[0].entry[1].type.value, "Immunization")

    # Verify Immunization entries are present
    imm_entries = [
        e for e in bundle.entry if e.resource.HasField("immunization")
    ]
    self.assertLen(imm_entries, 2)
    self.assertEqual(
        imm_entries[0].resource.immunization.vaccine_code.text.value,
        "Covaxin",
    )
    self.assertEqual(
        imm_entries[1].resource.immunization.vaccine_code.text.value,
        "Polio Vaccine",
    )

    # Verify bundle resource ordering: Composition, Patient, Practitioner,
    # Organization, Encounter, Immunizations
    entry_types = [
        e.resource.WhichOneof("oneof_resource") for e in bundle.entry
    ]
    self.assertEqual(
        entry_types,
        [
            "composition",
            "patient",
            "practitioner",
            "organization",
            "encounter",
            "immunization",
            "immunization",
        ],
    )

  def test_generate_fhir_missing_patient(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()
    doc = abdm_medical_documents.AbdmImmunizationRecord.model_construct(
        patient=None,
        immunizations=[],
    )
    with self.assertRaises(ValueError):
      generator.generate_fhir(doc)

  def test_generate_fhir_invalid_type(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()
    with self.assertRaises(TypeError):
      generator.generate_fhir(mock.MagicMock())

  def test_generate_fhir_missing_date_fallback(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()
    patient_data = resources.Patient(name="Jane Doe")
    imm = resources.Immunization(
        vaccine_name="BCG",
        dosage="1",
    )
    doc = abdm_medical_documents.AbdmImmunizationRecord(
        patient=patient_data,
        immunizations=[imm],
    )
    bundle = generator.generate_fhir(doc)
    self.assertTrue(bundle.entry[0].resource.HasField("composition"))
    self.assertTrue(bundle.entry[0].resource.composition.HasField("date"))

  def test_generate_fhir_missing_practitioner_author_fallback(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()
    patient_data = resources.Patient(name="Jane Doe")
    org_data = resources.Organization(name="City Hospital")
    doc = abdm_medical_documents.AbdmImmunizationRecord(
        patient=patient_data,
        practitioner=None,
        service_provider=org_data,
        immunizations=[],
    )
    bundle = generator.generate_fhir(doc)
    composition = bundle.entry[0].resource.composition
    self.assertLen(composition.author, 1)
    self.assertEqual(
        composition.author[0].uri.value,
        f"urn:uuid:{bundle.entry[2].resource.organization.id.value}",
    )

  def test_generate_fhir_no_practitioner_no_organization(self):
    generator = abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator()
    patient_data = resources.Patient(name="Jane Doe")
    doc = abdm_medical_documents.AbdmImmunizationRecord(
        patient=patient_data,
        practitioner=None,
        service_provider=None,
        immunizations=[],
    )
    bundle = generator.generate_fhir(doc)
    composition = bundle.entry[0].resource.composition
    self.assertEmpty(composition.author)


if __name__ == "__main__":
  absltest.main()

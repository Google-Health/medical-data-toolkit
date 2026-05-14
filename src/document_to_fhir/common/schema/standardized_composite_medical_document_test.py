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
"""Tests for standardized_composite_medical_document.py."""

from absl.testing import absltest
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import standardized_composite_medical_document


class StandardizedCompositeMedicalDocumentTest(absltest.TestCase):

  def test_hydrate_medical_document_lab_report(self):
    data = {
        "document_type": document_types.MedicalDocumentType.LABORATORY_REPORT,
        "medical_document": {
            "patient": {"name": "John Doe"},
            "lab_tests": [
                {"core_analyte": "Glucose", "name": "Glucose", "result": "100"}
            ],
        },
    }
    doc = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext.model_validate(
        data
    )
    self.assertIsInstance(doc.medical_document, medical_documents.LabReport)
    self.assertEqual(doc.medical_document.patient.name, "John Doe")

  def test_hydrate_medical_document_non_medical(self):
    data = {
        "document_type": document_types.MedicalDocumentType.NON_MEDICAL,
        "medical_document": {"some_field": "some_value"},
    }
    doc = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext.model_validate(
        data
    )
    # NON_MEDICAL doesn't have a specific model class mapped in the match case,
    # so it should remain a dict.
    self.assertEqual(doc.medical_document, {"some_field": "some_value"})

  def test_hydrate_medical_document_unrecognized_type(self):
    data = {
        "document_type": (
            document_types.MedicalDocumentType.DISCHARGE_SUMMARY
        ),
        "medical_document": {"field": "val"},
    }
    doc = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext.model_validate(
        data
    )
    self.assertEqual(doc.medical_document, {"field": "val"})


if __name__ == "__main__":
  absltest.main()

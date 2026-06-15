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

import datetime

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

  def test_datetime_serialization_format(self):
    # Prepare input and expected datetime for serveral types of variations of
    # the input.
    tz_offset = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
    input_dob = datetime.date(1990, 1, 1)
    test_cases_with_sample_collection_time = [
        # 1. UTC with ms
        {
            "input_dt": datetime.datetime(
                2023, 1, 19, 13, 3, 0, 123000, tzinfo=datetime.timezone.utc
            ),
            "expected_dt_format": "2023-01-19T13:03:00.123Z",
            "name": "UTC with ms",
        },
        # 2. UTC without ms
        {
            "input_dt": datetime.datetime(
                2023, 1, 19, 13, 3, 0, tzinfo=datetime.timezone.utc
            ),
            "expected_dt_format": "2023-01-19T13:03:00.000Z",
            "name": "UTC without ms",
        },
        # 3. Naive (assumed UTC)
        {
            "input_dt": datetime.datetime(2023, 1, 19, 13, 3, 0),
            "expected_dt_format": "2023-01-19T13:03:00.000Z",
            "name": "Naive (assumed UTC)",
        },
        # 4. Offset converted to UTC
        {
            "input_dt": datetime.datetime(
                2023, 1, 19, 18, 33, 0, tzinfo=tz_offset
            ),
            "expected_dt_format": "2023-01-19T13:03:00.000Z",
            "name": "Offset converted to UTC",
        },
        # 5. None
        {"input_dt": None, "expected_dt_format": None, "name": "None"},
    ]

    # Create a list of standardized documents with different datetime formats.
    standardized_docs = []
    for tc in test_cases_with_sample_collection_time:
      data = {
          "document_type": document_types.MedicalDocumentType.LABORATORY_REPORT,
          "medical_document": {
              "sample_collection_time": tc["input_dt"],
              "patient": {"name": tc["name"], "dob": input_dob},
              "lab_tests": [{
                  "core_analyte": "Glucose",
                  "name": "Glucose",
                  "result": "100",
              }],
          },
      }

      # Execute the function under test.
      doc = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext.model_validate(
          data
      )
      standardized_docs.append(doc)

    composite_doc = standardized_composite_medical_document.StandardizedCompositeMedicalDocumentWithContext(
        standardized_medical_documents=standardized_docs
    )

    # Serialize the composite document to JSON.
    serialized = composite_doc.model_dump(mode="json")
    docs_serialized = serialized["standardized_medical_documents"]

    # Verify the serialization format.
    for i, tc in enumerate(test_cases_with_sample_collection_time):
      with self.subTest(name=tc["name"]):
        self.assertEqual(
            docs_serialized[i]["medical_document"]["sample_collection_time"],
            tc["expected_dt_format"],
        )
        self.assertEqual(
            docs_serialized[i]["medical_document"]["patient"]["dob"],
            "1990-01-01",
        )

  def test_invalid_datetime_cleared(self):
    # Test data for invalid datetime string.
    data = {
        "document_type": document_types.MedicalDocumentType.LABORATORY_REPORT,
        "medical_document": {
            "sample_collection_time": "invalid-datetime-string",
            "patient": {
                "name": "John Doe",
            },
            "lab_tests": [
                {"core_analyte": "Glucose", "name": "Glucose", "result": "100"}
            ],
        },
    }
    # Execute the function under test.
    doc = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext.model_validate(
        data
    )
    # Verify that the invalid datetime is cleared.
    self.assertIsNone(doc.medical_document.sample_collection_time)


if __name__ == "__main__":
  absltest.main()

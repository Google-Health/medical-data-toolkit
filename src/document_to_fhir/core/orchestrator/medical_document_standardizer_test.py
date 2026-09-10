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
"""Tests for medical_document_standardizer."""

from unittest import mock

from absl.testing import absltest

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.orchestrator import medical_document_standardizer


class DummyMedicalData(resources.MedicalData):
  pass


class MedicalDocumentStandardizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_extractor = mock.Mock()
    self.medical_document = DummyMedicalData()
    self.mock_extractor.extract.return_value = self.medical_document

    self.mock_mapper = mock.Mock()
    self.mock_mapper_registry = {DummyMedicalData: self.mock_mapper}

    self.mock_fhir_generator = mock.Mock()
    self.mock_fhir_bundle = {"resourceType": "Bundle"}
    self.mock_fhir_generator.generate_fhir.return_value = self.mock_fhir_bundle

  def test_standardize_all_provided(self):
    standardizer = medical_document_standardizer.MedicalDocumentStandardizer(
        extractor=self.mock_extractor,
        mapper_registry=self.mock_mapper_registry,
        fhir_generator=self.mock_fhir_generator,
    )

    doc, fhir, latencies = standardizer.standardize([b"image"])

    self.assertEqual(doc, self.medical_document)
    self.assertEqual(fhir, self.mock_fhir_bundle)
    self.mock_extractor.extract.assert_called_once()
    self.mock_mapper.map_inplace.assert_called_once_with(self.medical_document)
    self.mock_fhir_generator.generate_fhir.assert_called_once_with(
        self.medical_document
    )
    self.assertIn("extraction", latencies)
    self.assertIn("terminology_mapping", latencies)
    self.assertIn("fhir_generation", latencies)

  def test_standardize_no_mapper_registry(self):
    standardizer = medical_document_standardizer.MedicalDocumentStandardizer(
        extractor=self.mock_extractor,
        mapper_registry=None,
        fhir_generator=self.mock_fhir_generator,
    )

    doc, fhir, latencies = standardizer.standardize([b"image"])

    self.assertEqual(doc, self.medical_document)
    self.assertEqual(fhir, self.mock_fhir_bundle)
    self.mock_extractor.extract.assert_called_once()
    self.mock_mapper.map_inplace.assert_not_called()
    self.mock_fhir_generator.generate_fhir.assert_called_once_with(
        self.medical_document
    )
    self.assertEqual(latencies["terminology_mapping"], 0.0)

  def test_standardize_no_fhir_generator(self):
    standardizer = medical_document_standardizer.MedicalDocumentStandardizer(
        extractor=self.mock_extractor,
        mapper_registry=self.mock_mapper_registry,
        fhir_generator=None,
    )

    doc, fhir, latencies = standardizer.standardize([b"image"])

    self.assertEqual(doc, self.medical_document)
    self.assertIsNone(fhir)
    self.mock_extractor.extract.assert_called_once()
    self.mock_mapper.map_inplace.assert_called_once_with(self.medical_document)
    self.assertEqual(latencies["fhir_generation"], 0.0)

  def test_standardize_none_provided(self):
    standardizer = medical_document_standardizer.MedicalDocumentStandardizer(
        extractor=self.mock_extractor,
        mapper_registry=None,
        fhir_generator=None,
    )

    doc, fhir, latencies = standardizer.standardize([b"image"])

    self.assertEqual(doc, self.medical_document)
    self.assertIsNone(fhir)
    self.mock_extractor.extract.assert_called_once()
    self.mock_mapper.map_inplace.assert_not_called()
    self.assertEqual(latencies["terminology_mapping"], 0.0)
    self.assertEqual(latencies["fhir_generation"], 0.0)


if __name__ == "__main__":
  absltest.main()

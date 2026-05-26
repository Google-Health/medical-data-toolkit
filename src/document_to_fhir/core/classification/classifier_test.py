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
"""Tests for classifier.py."""

from unittest import mock

from absl.testing import absltest
from google.genai import types

from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import standardized_composite_medical_document
from src.document_to_fhir.core.classification import classifier


class ClassifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.Mock()
    self.mock_client.supports_pdf = True

  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.read_prompt"
  )
  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.DocumentClassifierBase._chunk_pdf_to_parts"
  )
  def test_multi_document_classifier_no_chunks(
      self, mock_chunk, mock_load_prompt
  ):
    mock_load_prompt.return_value = "EXPECTED_MULTI_PROMPT"
    mock_chunk.return_value = [[types.Part.from_text(text="fake_parts")]]
    mock_response = mock.Mock()
    mock_response.parsed = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                reasoning="Rx found",
                start_page=1,
                end_page=1,
            )
        ]
    )
    self.mock_client.generate_content.return_value = mock_response
    self.mock_client.supports_pdf = True

    c = classifier.MultiDocumentClassifier(self.mock_client)
    result = c.classify(b"fake_pdf_content", split_into_chunks=False)

    self.assertLen(result.segments, 1)
    self.assertEqual(
        result.segments[0].document_type,
        document_types.MedicalDocumentType.PRESCRIPTION,
    )
    _, kwargs = self.mock_client.generate_content.call_args
    self.assertIn(
        types.Part.from_text(text="EXPECTED_MULTI_PROMPT"), kwargs["contents"]
    )

  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.DocumentClassifierBase._chunk_pdf_to_parts"
  )
  def test_multi_document_classifier_with_chunks(self, mock_chunk):
    mock_chunk.return_value = [
        [types.Part.from_text(text="chunk1")],
        [types.Part.from_text(text="chunk2")],
    ]

    mock_response1 = mock.Mock()
    mock_response1.parsed = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                reasoning="Lab",
                start_page=1,
                end_page=2,
            )
        ]
    )

    mock_response2 = mock.Mock()
    mock_response2.parsed = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                reasoning="Lab",
                start_page=2,
                end_page=3,
            )
        ]
    )

    self.mock_client.generate_content.side_effect = [
        mock_response1,
        mock_response2,
    ]
    self.mock_client.supports_pdf = False

    c = classifier.MultiDocumentClassifier(self.mock_client)
    result = c.classify(
        b"fake_pdf_content", split_into_chunks=True, chunk_size=2
    )

    self.assertLen(result.segments, 1)
    self.assertEqual(result.segments[0].start_page, 1)
    # Merged logic should set the end page.
    self.assertEqual(result.segments[0].end_page, 3)

  def test_merge_outputs_overlap_is_standalone_document(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    document_classification_outputs = [
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=1,
                    end_page=2,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=3,
                    reasoning="",
                ),
            ]
        ),
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=3,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=4,
                    end_page=6,
                    reasoning="",
                ),
            ]
        ),
    ]
    expected_segments = [
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=1,
            end_page=2,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=3,
            end_page=3,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=4,
            end_page=6,
            reasoning="",
        ),
    ]
    actual_output = c._merge_outputs(document_classification_outputs)
    self.assertEqual(expected_segments, actual_output.segments)

  def test_merge_outputs_overlap_is_first_page_of_new_segment(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    document_classification_outputs = [
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=1,
                    end_page=2,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=3,
                    reasoning="",
                ),
            ]
        ),
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=5,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=6,
                    end_page=6,
                    reasoning="",
                ),
            ]
        ),
    ]
    expected_segments = [
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=1,
            end_page=2,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=3,
            end_page=5,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=6,
            end_page=6,
            reasoning="",
        ),
    ]
    actual_output = c._merge_outputs(document_classification_outputs)
    self.assertEqual(expected_segments, actual_output.segments)

  def test_merge_outputs_overlap_is_last_page_of_old_segment(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    document_classification_outputs = [
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=1,
                    end_page=2,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=5,
                    reasoning="",
                ),
            ]
        ),
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=5,
                    end_page=5,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=6,
                    end_page=6,
                    reasoning="",
                ),
            ]
        ),
    ]
    expected_segments = [
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=1,
            end_page=2,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=3,
            end_page=5,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=6,
            end_page=6,
            reasoning="",
        ),
    ]
    actual_output = c._merge_outputs(document_classification_outputs)
    self.assertEqual(expected_segments, actual_output.segments)

  def test_merge_outputs_overlap_is_in_middle_of_document(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    document_classification_outputs = [
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=1,
                    end_page=2,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=3,
                    end_page=5,
                    reasoning="",
                ),
            ]
        ),
        standardized_composite_medical_document.CompositeDocument(
            segments=[
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                    start_page=5,
                    end_page=7,
                    reasoning="",
                ),
                standardized_composite_medical_document.DocumentSegment(
                    document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                    start_page=6,
                    end_page=6,
                    reasoning="",
                ),
            ]
        ),
    ]
    expected_segments = [
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=1,
            end_page=2,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=3,
            end_page=7,
            reasoning="",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=6,
            end_page=6,
            reasoning="",
        ),
    ]
    actual_output = c._merge_outputs(document_classification_outputs)
    self.assertEqual(expected_segments, actual_output.segments)

  def test_sort_document_segments(self):
    doc = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                start_page=5,
                end_page=5,
                reasoning="Rx 1",
            ),
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                start_page=1,
                end_page=2,
                reasoning="Lab 1",
            ),
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                start_page=3,
                end_page=4,
                reasoning="Rx 2",
            ),
        ]
    )

    sorted_doc = classifier.sort_document_segments(doc)

    expected_segments = [
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
            start_page=1,
            end_page=2,
            reasoning="Lab 1",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=3,
            end_page=4,
            reasoning="Rx 2",
        ),
        standardized_composite_medical_document.DocumentSegment(
            document_type=document_types.MedicalDocumentType.PRESCRIPTION,
            start_page=5,
            end_page=5,
            reasoning="Rx 1",
        ),
    ]
    self.assertEqual(sorted_doc.segments, expected_segments)

if __name__ == "__main__":
  absltest.main()

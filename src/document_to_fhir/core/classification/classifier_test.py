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

import threading
from unittest import mock

from absl.testing import absltest
from google.genai import types

from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import standardized_composite_medical_document
from src.document_to_fhir.core.classification import classifier

token_usage_var = model_client.token_usage_var
LLMUsage = model_client.LLMUsage


class ClassifierTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_client = mock.Mock()
    self.mock_client.supports_pdf = True

  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.read_prompt"
  )
  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.DocumentClassifierBase._chunk_images_to_parts"
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
      "src.document_to_fhir.core.classification.classifier.DocumentClassifierBase._chunk_images_to_parts"
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

    def generate_content_side_effect(contents, **kwargs):
      del kwargs
      for part in contents:
        if hasattr(part, "text") and part.text:
          if "chunk1" in part.text:
            return mock_response1
          elif "chunk2" in part.text:
            return mock_response2
      return mock.DEFAULT

    self.mock_client.generate_content.side_effect = generate_content_side_effect
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

  def test_process_handwritten_medical_pages(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    doc = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                start_page=1,
                end_page=2,
                reasoning="",
                handwritten_content_percent=80,
            ),
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                start_page=3,
                end_page=4,
                reasoning="",
                handwritten_content_percent=50,
            ),
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.NON_MEDICAL,
                start_page=5,
                end_page=6,
                reasoning="",
                handwritten_content_percent=90,
            ),
        ]
    )

    actual = c.process_handwritten_medical_pages(
        doc, handwritten_percent_threshold=70
    )

    expected_types = [
        document_types.MedicalDocumentType.HANDWRITTEN,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
        document_types.MedicalDocumentType.NON_MEDICAL,
    ]
    actual_types = [seg.document_type for seg in actual.segments]

    self.assertEqual(expected_types, actual_types)

  def test_process_handwritten_medical_pages_default_threshold(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    doc = standardized_composite_medical_document.CompositeDocument(
        segments=[
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.PRESCRIPTION,
                start_page=1,
                end_page=2,
                reasoning="",
                handwritten_content_percent=35,
            ),
            standardized_composite_medical_document.DocumentSegment(
                document_type=document_types.MedicalDocumentType.LABORATORY_REPORT,
                start_page=3,
                end_page=4,
                reasoning="",
                handwritten_content_percent=30,
            ),
        ]
    )

    actual = c.process_handwritten_medical_pages(doc)

    expected_types = [
        document_types.MedicalDocumentType.HANDWRITTEN,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
    ]
    actual_types = [seg.document_type for seg in actual.segments]

    self.assertEqual(expected_types, actual_types)

  def test_chunk_images_to_parts(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    images = [b"img1", b"img2", b"img3"]
    chunk_size = 2
    overlap = 0

    expected = [
        [
            types.Part.from_text(text="==Start of Document==\n"),
            types.Part.from_text(text="==Screenshot for page 1==\n"),
            types.Part.from_bytes(data=b"img1", mime_type="image/png"),
            types.Part.from_text(text="==Screenshot for page 2==\n"),
            types.Part.from_bytes(data=b"img2", mime_type="image/png"),
            types.Part.from_text(text="\n==End of Document==\n\n"),
        ],
        [
            types.Part.from_text(text="==Start of Document==\n"),
            types.Part.from_text(text="==Screenshot for page 3==\n"),
            types.Part.from_bytes(data=b"img3", mime_type="image/png"),
            types.Part.from_text(text="\n==End of Document==\n\n"),
        ],
    ]

    actual = c._chunk_images_to_parts(images, chunk_size, overlap, "image/png")
    self.assertEqual(actual, expected)

  def test_chunk_images_to_parts_with_overlap(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    images = [b"img1", b"img2", b"img3"]
    chunk_size = 2
    overlap = 1

    expected = [
        [
            types.Part.from_text(text="==Start of Document==\n"),
            types.Part.from_text(text="==Screenshot for page 1==\n"),
            types.Part.from_bytes(data=b"img1", mime_type="image/png"),
            types.Part.from_text(text="==Screenshot for page 2==\n"),
            types.Part.from_bytes(data=b"img2", mime_type="image/png"),
            types.Part.from_text(text="\n==End of Document==\n\n"),
        ],
        [
            types.Part.from_text(text="==Start of Document==\n"),
            types.Part.from_text(text="==Screenshot for page 2==\n"),
            types.Part.from_bytes(data=b"img2", mime_type="image/png"),
            types.Part.from_text(text="==Screenshot for page 3==\n"),
            types.Part.from_bytes(data=b"img3", mime_type="image/png"),
            types.Part.from_text(text="\n==End of Document==\n\n"),
        ],
    ]

    actual = c._chunk_images_to_parts(images, chunk_size, overlap, "image/png")
    self.assertEqual(actual, expected)

  def test_chunk_images_to_parts_raises_value_error(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    with self.assertRaises(ValueError):
      c._chunk_images_to_parts(
          [b"img1"], chunk_size=2, overlap=2, mime_type="image/png"
      )

  def test_prepare_request_contents_images(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)
    images = [b"img1"]

    expected = [[
        types.Part.from_text(text="==Start of Document==\n"),
        types.Part.from_text(text="==Screenshot for page 1==\n"),
        types.Part.from_bytes(data=b"img1", mime_type="image/jpeg"),
        types.Part.from_text(text="\n==End of Document==\n\n"),
        types.Part.from_text(text="test prompt"),
    ]]

    actual = c._prepare_request_contents(
        images,
        "test prompt",
        mime_type="image/jpeg",
        chunk_size=15,
        overlap=1,
    )
    self.assertEqual(actual, expected)

  @mock.patch(
      "src.document_to_fhir.core.classification.classifier.DocumentClassifierBase._chunk_images_to_parts"
  )
  def test_multi_document_classifier_partial_failure_and_metadata(
      self, mock_chunk
  ):
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

    def generate_content_side_effect(contents, **kwargs):
      del kwargs
      usage_list = token_usage_var.get()
      for part in contents:
        if hasattr(part, "text") and part.text:
          if "chunk1" in part.text:
            if usage_list is not None:
              usage_list.append(
                  LLMUsage(
                      prompt_tokens=10, completion_tokens=5, total_tokens=15
                  )
              )
            return mock_response1
          elif "chunk2" in part.text:
            raise ValueError("Simulated LLM Failure")
      return mock.DEFAULT

    self.mock_client.generate_content.side_effect = generate_content_side_effect
    self.mock_client.supports_pdf = False

    c = classifier.MultiDocumentClassifier(self.mock_client)

    # Set up context variables to collect metadata
    parent_token_usages = []
    parent_latencies = []
    token_usage_token = token_usage_var.set(parent_token_usages)
    latencies_token = classifier.classification_latencies_var.set(
        parent_latencies
    )

    # Assert logs to verify chunk numbers are correct (kills mutant on chunk_num)
    with self.assertLogs(level="INFO") as log:
      try:
        result = c.classify(
            b"fake_pdf_content", split_into_chunks=True, chunk_size=2
        )
      finally:
        token_usage_var.reset(token_usage_token)
        classifier.classification_latencies_var.reset(latencies_token)

    # Verify logs (kills mutant on chunk_num)
    log_output = "\n".join(log.output)
    self.assertIn("Processing classification chunk 1/2", log_output)
    self.assertIn("Classification chunk 1/2 completed", log_output)
    self.assertIn("Classification chunk 2/2 FAILED with exception", log_output)

    # Verify that we got results from the successful chunk
    self.assertLen(result.segments, 1)
    self.assertEqual(
        result.segments[0].document_type,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
    )
    self.assertEqual(result.segments[0].start_page, 1)
    self.assertEqual(result.segments[0].end_page, 2)

    # Verify that metadata was collected correctly
    self.assertLen(parent_token_usages, 1)
    self.assertEqual(parent_token_usages[0].prompt_tokens, 10)
    self.assertEqual(parent_token_usages[0].completion_tokens, 5)
    self.assertEqual(parent_token_usages[0].total_tokens, 15)

    # Verify that latency is recorded
    self.assertLen(parent_latencies, 1)
    self.assertGreater(parent_latencies[0], 0.0)

  def test_chunk_images_to_parts_scenarios(self):
    c = classifier.MultiDocumentClassifier(self.mock_client)

    # Case 1: total_pages == chunk_size
    images = [b"img1", b"img2", b"img3", b"img4", b"img5"]
    chunks = c._chunk_images_to_parts(
        images=images, chunk_size=5, overlap=1, mime_type="image/png"
    )
    self.assertLen(chunks, 1)

    # Case 2: total_pages < chunk_size
    images = [b"img1", b"img2", b"img3"]
    chunks = c._chunk_images_to_parts(
        images=images, chunk_size=5, overlap=1, mime_type="image/png"
    )
    self.assertLen(chunks, 1)

    # Case 3: total_pages > chunk_size
    images = [
        b"img1",
        b"img2",
        b"img3",
        b"img4",
        b"img5",
        b"img6",
    ]
    chunks = c._chunk_images_to_parts(
        images=images, chunk_size=5, overlap=1, mime_type="image/png"
    )
    self.assertLen(chunks, 2)

    # Case 4: total_pages = 20, chunk_size = 10, overlap = 1
    images = [b"img"] * 20
    chunks = c._chunk_images_to_parts(
        images=images, chunk_size=10, overlap=1, mime_type="image/png"
    )
    self.assertLen(chunks, 3)

    # Case 5: total_pages = 19, chunk_size = 10, overlap = 1
    images = [b"img"] * 19
    chunks = c._chunk_images_to_parts(
        images=images, chunk_size=10, overlap=1, mime_type="image/png"
    )
    self.assertLen(chunks, 2)


if __name__ == "__main__":
  absltest.main()

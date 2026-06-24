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

import logging
from unittest import mock

from absl.testing import absltest

from src.document_to_fhir.common.model_client import LLMUsage
from src.document_to_fhir.common.model_client import token_usage_var
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.core.orchestrator import composite_document_standardizer


class CompositeDocumentStandardizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_classifier = mock.Mock()
    self.mock_classifier.process_handwritten_medical_pages.side_effect = (
        lambda x: x
    )
    self.mock_standardizer = mock.Mock()
    self.standardizers = {
        document_types.MedicalDocumentType.LABORATORY_REPORT: (
            self.mock_standardizer
        )
    }

  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_policy_allow_only_supported_skips(
      self, mock_convert
  ):
    mock_convert.return_value = [b"page1"]

    # Mock classifier to return an unsupported document type
    mock_segment = mock.Mock()
    mock_segment.document_type = document_types.MedicalDocumentType.NON_MEDICAL
    mock_segment.start_page = 1
    mock_segment.end_page = 1
    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment]
    self.mock_classifier.classify.return_value = mock_doc
    standardizer = composite_document_standardizer.CompositeDocumentStandardizer(
        classifier=self.mock_classifier,
        standardizers=self.standardizers,
        document_standardization_policy=(
            document_types.DocumentStandardizationPolicy.ALLOW_ONLY_SUPPORTED
        ),
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.assertEqual(result.n_documents, 1)
    self.assertEqual(
        result.standardized_medical_documents[0].document_type,
        document_types.MedicalDocumentType.NON_MEDICAL,
    )

  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_policy_accept_all_continues(
      self, mock_convert, mock_print_fhir
  ):
    mock_convert.return_value = [b"page1", b"page2", b"page3"]
    mock_print_fhir.return_value = "{}"

    # Mock classifier to return supported and unsupported types
    mock_segment_lab = mock.Mock()
    mock_segment_lab.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment_lab.start_page = 1
    mock_segment_lab.end_page = 1

    mock_segment_presc = mock.Mock()
    mock_segment_presc.document_type = (
        document_types.MedicalDocumentType.PRESCRIPTION
    )
    mock_segment_presc.start_page = 2
    mock_segment_presc.end_page = 2

    mock_segment_other = mock.Mock()
    mock_segment_other.document_type = (
        document_types.MedicalDocumentType.NON_MEDICAL
    )
    mock_segment_other.start_page = 3
    mock_segment_other.end_page = 3

    mock_doc = mock.Mock()
    mock_doc.segments = [
        mock_segment_lab,
        mock_segment_presc,
        mock_segment_other,
    ]
    self.mock_classifier.classify.return_value = mock_doc

    # Mock standardizer for LAB_REPORT
    self.mock_standardizer.standardize.return_value = (
        mock.Mock(),
        mock.Mock(),
        {},
    )

    standardizer = composite_document_standardizer.CompositeDocumentStandardizer(
        classifier=self.mock_classifier,
        standardizers=self.standardizers,
        document_standardization_policy=document_types.DocumentStandardizationPolicy.ACCEPT_ALL,
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.mock_classifier.classify.assert_called_once()
    # Since we have at least one segment to process (LABORATORY_REPORT), it
    # should NOT return early.
    mock_convert.assert_called_once()
    # We only have standardizer for LAB_REPORT
    self.mock_standardizer.standardize.assert_called_once()
    self.assertEqual(result.n_documents, 3)
    self.assertEqual(
        result.standardized_medical_documents[0].document_type,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
    )
    self.assertEqual(
        result.standardized_medical_documents[1].document_type,
        document_types.MedicalDocumentType.PRESCRIPTION,
    )
    self.assertEqual(
        result.standardized_medical_documents[2].document_type,
        document_types.MedicalDocumentType.NON_MEDICAL,
    )

  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_policy_allow_unsupported_non_medical_filters_other(
      self, mock_convert, mock_print_fhir
  ):
    mock_convert.return_value = [b"page1", b"page2"]
    mock_print_fhir.return_value = "{}"

    # Mock classifier to return a supported and an unsupported non-medical type
    mock_segment_supported = mock.Mock()
    mock_segment_supported.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment_supported.start_page = 1
    mock_segment_supported.end_page = 1

    mock_segment_other = mock.Mock()
    mock_segment_other.document_type = (
        document_types.MedicalDocumentType.NON_MEDICAL
    )
    mock_segment_other.start_page = 2
    mock_segment_other.end_page = 2

    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment_supported, mock_segment_other]
    self.mock_classifier.classify.return_value = mock_doc

    # Mock standardizer to return something
    self.mock_standardizer.standardize.return_value = (
        mock.Mock(),
        mock.Mock(),
        {},
    )

    standardizer = composite_document_standardizer.CompositeDocumentStandardizer(
        classifier=self.mock_classifier,
        standardizers=self.standardizers,
        document_standardization_policy=document_types.DocumentStandardizationPolicy.ALLOW_UNSUPPORTED_NON_MEDICAL,
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.mock_classifier.classify.assert_called_once()
    mock_convert.assert_called_once()
    self.mock_standardizer.standardize.assert_called_once()
    self.assertEqual(result.n_documents, 2)
    self.assertEqual(
        result.standardized_medical_documents[0].document_type,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
    )

    self.assertEqual(
        result.standardized_medical_documents[1].document_type,
        document_types.MedicalDocumentType.NON_MEDICAL,
    )

  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_policy_allow_unsupported_non_medical_discards_medical(
      self, mock_convert
  ):

    mock_convert.return_value = [b"page1", b"page2"]

    # Mock classifier to return a supported and an unsupported medical type
    mock_segment_supported = mock.Mock()
    mock_segment_supported.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment_supported.start_page = 1
    mock_segment_supported.end_page = 1

    mock_segment_prescription = mock.Mock()
    mock_segment_prescription.document_type = (
        document_types.MedicalDocumentType.PRESCRIPTION
    )
    mock_segment_prescription.start_page = 2
    mock_segment_prescription.end_page = 2

    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment_supported, mock_segment_prescription]
    self.mock_classifier.classify.return_value = mock_doc

    standardizer = composite_document_standardizer.CompositeDocumentStandardizer(
        classifier=self.mock_classifier,
        standardizers=self.standardizers,
        document_standardization_policy=document_types.DocumentStandardizationPolicy.ALLOW_UNSUPPORTED_NON_MEDICAL,
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.mock_classifier.classify.assert_called_once()
    # Pre-processing is run upfront before classification, so
    # mock_convert is called.
    mock_convert.assert_called_once()
    self.mock_standardizer.standardize.assert_not_called()
    self.assertEqual(result.n_documents, 2)
    self.assertEqual(
        result.standardized_medical_documents[0].document_type,
        document_types.MedicalDocumentType.LABORATORY_REPORT,
    )
    self.assertEqual(
        result.standardized_medical_documents[1].document_type,
        document_types.MedicalDocumentType.PRESCRIPTION,
    )

  @mock.patch(
      "src.document_to_fhir.core.fhir.abdm.abdm_bundle_enricher.add_document_reference_to_lab_report"
  )
  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_attaches_document_reference(
      self, mock_convert, mock_print_fhir, mock_add_doc_ref
  ):
    mock_convert.return_value = [b"page1"]
    mock_print_fhir.return_value = '{"entry": []}'

    updated_bundle = {
        "entry": [{"resource": {"resourceType": "DocumentReference"}}]
    }
    mock_add_doc_ref.return_value = (updated_bundle, True)

    # Mock classifier
    mock_segment = mock.Mock()
    mock_segment.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment.start_page = 1
    mock_segment.end_page = 1
    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment]
    self.mock_classifier.classify.return_value = mock_doc

    # Mock standardizer
    self.mock_standardizer.standardize.return_value = (
        mock.Mock(),
        mock.Mock(),
        {},
    )

    standardizer = (
        composite_document_standardizer.CompositeDocumentStandardizer(
            classifier=self.mock_classifier,
            standardizers=self.standardizers,
            attach_document_to_bundle=True,
        )
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.assertEqual(result.n_documents, 1)
    doc = result.standardized_medical_documents[0]
    self.assertEqual(doc.fhir_bundle, updated_bundle)

  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_returns_metadata(
      self, mock_convert, mock_print_fhir
  ):
    mock_convert.return_value = [b"page1", b"page2"]
    mock_print_fhir.return_value = "{}"

    # Mock classifier
    mock_segment = mock.Mock()
    mock_segment.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment.start_page = 1
    mock_segment.end_page = 2
    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment]

    def mock_classify(*unused_args, **unused_kwargs):
      # Simulate classification token usage
      usage_list = token_usage_var.get()
      if usage_list is not None:
        usage_list.append(
            LLMUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
      return mock_doc

    self.mock_classifier.classify.side_effect = mock_classify

    # Mock standardizer
    def mock_standardize(*unused_args, **unused_kwargs):
      # Simulate standardization token usage
      usage_list = token_usage_var.get()
      if usage_list is not None:
        usage_list.append(
            LLMUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30)
        )
      return (
          mock.Mock(),
          mock.Mock(),
          {
              "extraction": 10.0,
              "terminology_mapping": 5.0,
              "fhir_generation": 2.0,
          },
      )

    self.mock_standardizer.standardize.side_effect = mock_standardize

    standardizer = (
        composite_document_standardizer.CompositeDocumentStandardizer(
            classifier=self.mock_classifier,
            standardizers=self.standardizers,
            return_metadata=True,
        )
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.assertIsNotNone(result.metadata)
    assert result.metadata is not None
    self.assertIn("total", result.metadata.latency_ms)
    self.assertIn("classification", result.metadata.latency_ms)
    self.assertIn("standardization", result.metadata.latency_ms)
    self.assertLen(result.metadata.latency_ms["standardization"]["segments"], 1)

    # Verify tokens
    self.assertLen(result.metadata.token_usage["classification"], 1)
    self.assertEqual(
        result.metadata.token_usage["classification"][0]["total_tokens"], 15
    )
    self.assertLen(result.metadata.token_usage["standardization"], 1)
    self.assertEqual(
        result.metadata.token_usage["standardization"][0]["document_type"],
        document_types.MedicalDocumentType.LABORATORY_REPORT.value,
    )
    self.assertLen(
        result.metadata.token_usage["standardization"][0]["calls"], 1
    )
    self.assertEqual(
        result.metadata.token_usage["standardization"][0]["calls"][0][
            "total_tokens"
        ],
        30,
    )
    self.assertNotIn("total", result.metadata.token_usage)

  def test_standardize_unsupported_mime_type_raises_value_error(self):
    standardizer = (
        composite_document_standardizer.CompositeDocumentStandardizer(
            classifier=self.mock_classifier,
            standardizers=self.standardizers,
        )
    )
    with self.assertRaises(ValueError):
      standardizer.standardize(b"fake_content", mime_type="text/plain")

  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  @mock.patch(
      "src.document_to_fhir.common.pdf_util.convert_pdf_pages_to_png_images"
  )
  def test_standardize_image_input(self, mock_convert, mock_print_fhir):
    mock_print_fhir.return_value = "{}"
    # Mock classifier
    mock_segment = mock.Mock()
    mock_segment.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment.start_page = 1
    mock_segment.end_page = 1
    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment]
    self.mock_classifier.classify.return_value = mock_doc

    # Mock standardizer
    self.mock_standardizer.standardize.return_value = (
        mock.Mock(),
        mock.Mock(),
        {},
    )

    standardizer = (
        composite_document_standardizer.CompositeDocumentStandardizer(
            classifier=self.mock_classifier,
            standardizers=self.standardizers,
        )
    )

    result = standardizer.standardize(
        b"fake_image_content", mime_type="image/jpeg"
    )

    # pdf_util.convert_pdf_pages_to_png_images should not be called.
    mock_convert.assert_not_called()
    self.mock_classifier.classify.assert_called_once_with(
        [b"fake_image_content"], temperature=0, mime_type="image/jpeg"
    )
    self.assertEqual(result.n_documents, 1)

  @mock.patch(
      "src.document_to_fhir.core.orchestrator.composite_document_standardizer.json_format.print_fhir_to_json_string"
  )
  def test_standardize_with_log_metrics_false(self, mock_print_fhir):
    mock_print_fhir.return_value = "{}"
    # Mock classifier
    mock_segment = mock.Mock()
    mock_segment.document_type = (
        document_types.MedicalDocumentType.LABORATORY_REPORT
    )
    mock_segment.start_page = 1
    mock_segment.end_page = 1
    mock_doc = mock.Mock()
    mock_doc.segments = [mock_segment]
    self.mock_classifier.classify.return_value = mock_doc

    # Mock standardizer
    self.mock_standardizer.standardize.return_value = (
        mock.Mock(),
        mock.Mock(),
        {},
    )

    standardizer = (
        composite_document_standardizer.CompositeDocumentStandardizer(
            classifier=self.mock_classifier,
            standardizers=self.standardizers,
            log_metrics=False,
        )
    )

    # We assert that no log messages at level INFO or above are printed
    # by our pipeline metrics logger when log_metrics=False.
    with self.assertLogs(level="INFO") as log_watcher:
      logging.info("dummy_log_before")
      standardizer.standardize(b"fake_image_content", mime_type="image/jpeg")
      logging.info("dummy_log_after")

    # The pipeline metrics log should NOT be in log_watcher.output
    has_pipeline_metrics_log = any(
        "MDDAS Pipeline Metrics" in log for log in log_watcher.output
    )
    self.assertFalse(
        has_pipeline_metrics_log,
        "Pipeline metrics were logged even though log_metrics=False:"
        f" {log_watcher.output}",
    )


if __name__ == "__main__":
  absltest.main()

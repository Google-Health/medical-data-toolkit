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

from unittest import mock

from absl.testing import absltest

from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.core.orchestrator import composite_document_standardizer


class CompositeDocumentStandardizerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_classifier = mock.Mock()
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
    self.mock_standardizer.standardize.return_value = (mock.Mock(), mock.Mock())

    standardizer = composite_document_standardizer.CompositeDocumentStandardizer(
        classifier=self.mock_classifier,
        standardizers=self.standardizers,
        document_standardization_policy=document_types.DocumentStandardizationPolicy.ACCEPT_ALL,
    )

    result = standardizer.standardize(b"fake_pdf_content")

    self.mock_classifier.classify.assert_called_once()
    # Since we have at least one segment to process (LABORATORY_REPORT), it should NOT
    # return early.
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
    self.mock_standardizer.standardize.return_value = (mock.Mock(), mock.Mock())

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
    # In the current implementation, if segments_to_process is empty,
    # it returns early BEFORE conversion.
    mock_convert.assert_not_called()
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
    self.mock_standardizer.standardize.return_value = (mock.Mock(), mock.Mock())

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


if __name__ == "__main__":
  absltest.main()

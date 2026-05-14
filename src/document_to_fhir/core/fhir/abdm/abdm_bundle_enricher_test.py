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
"""Tests for abdm_bundle_enricher."""

import json
from unittest import mock

from absl.testing import absltest

from google.fhir.r4 import json_format
from src.document_to_fhir.core.fhir.abdm import abdm_bundle_enricher


class AbdmBundleEnricherTest(absltest.TestCase):

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  @mock.patch.object(json_format, "print_fhir_to_json_string", autospec=True)
  def test_add_document_reference_to_lab_report_success(
      self, mock_print_fhir, mock_json_to_proto
  ):
    bundle_json = {
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-uuid",
                "resource": {"resourceType": "Patient"},
            },
            {
                "fullUrl": "urn:uuid:composition-uuid",
                "resource": {
                    "resourceType": "Composition",
                    "section": [{"entry": []}],
                },
            },
        ]
    }
    updated_bundle_json = {
        "entry": [
            {
                "fullUrl": "urn:uuid:patient-uuid",
                "resource": {"resourceType": "Patient"},
            },
            {
                "fullUrl": "urn:uuid:composition-uuid",
                "resource": {
                    "resourceType": "Composition",
                    "section": [
                        {"entry": [{"reference": "urn:uuid:doc-ref-uuid"}]}
                    ],
                },
            },
            {
                "fullUrl": "urn:uuid:doc-ref-uuid",
                "resource": {
                    "resourceType": "DocumentReference",
                    "status": "current",
                    "subject": {"reference": "urn:uuid:patient-uuid"},
                    "content": [
                        {
                            "attachment": {
                                "contentType": "application/pdf",
                                "data": "fake_encoded_data",
                            }
                        }
                    ],
                },
            },
        ]
    }
    mock_print_fhir.return_value = json.dumps(updated_bundle_json)

    # Mock Proto for bundle
    mock_bundle_proto = mock.Mock()
    mock_json_to_proto.return_value = mock_bundle_proto

    mock_composition_entry = mock.Mock()
    mock_composition_entry.resource.HasField.side_effect = (
        lambda field: field == "composition"
    )
    mock_composition_entry.resource.composition.meta.profile = [
        mock.Mock(
            value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportRecord"
        )
    ]
    mock_section = mock.Mock()
    mock_section.entry = mock.Mock()
    mock_composition_entry.resource.composition.section = [mock_section]

    mock_patient_entry = mock.Mock()
    mock_patient_entry.resource.HasField.side_effect = (
        lambda field: field == "patient"
    )
    mock_patient_entry.full_url.value = "urn:uuid:patient-uuid"

    mock_entry_list = [mock_composition_entry, mock_patient_entry]
    mock_bundle_proto.entry = mock.MagicMock()
    mock_bundle_proto.entry.__iter__.return_value = iter(mock_entry_list)
    mock_bundle_proto.entry.__getitem__.side_effect = (
        lambda idx: mock_entry_list[idx]
    )
    mock_bundle_proto.entry.add.return_value = mock.Mock()

    result, modified = (
        abdm_bundle_enricher.add_document_reference_to_lab_report(
            bundle_json, "fake_encoded_data", "application/pdf"
        )
    )

    self.assertEqual(result, updated_bundle_json)
    self.assertTrue(modified)

    # Verify that the DocumentReference was created with the encoded data
    mock_doc_ref_copy = (
        mock_bundle_proto.entry.add.return_value.resource.document_reference.CopyFrom
    )
    mock_doc_ref_copy.assert_called_once()
    doc_ref_proto = mock_doc_ref_copy.call_args[0][0]
    self.assertEqual(
        doc_ref_proto.content[0].attachment.data.value, b"fake_encoded_data"
    )

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  def test_add_document_reference_to_lab_report_skips_non_lab_report(
      self, mock_json_to_proto
  ):
    bundle_json = {"entry": []}

    # Mock Proto for bundle
    mock_bundle_proto = mock.Mock()
    mock_json_to_proto.return_value = mock_bundle_proto
    mock_bundle_proto.entry = []

    result, modified = (
        abdm_bundle_enricher.add_document_reference_to_lab_report(
            bundle_json, "fake_encoded_data", "application/pdf"
        )
    )

    self.assertEqual(result, bundle_json)
    self.assertFalse(modified)

  def test_add_document_reference_to_lab_report_empty_bundle(self):
    result, modified = (
        abdm_bundle_enricher.add_document_reference_to_lab_report(
            None, "fake_encoded_data", "application/pdf"
        )
    )
    self.assertIsNone(result)
    self.assertFalse(modified)

    result, modified = (
        abdm_bundle_enricher.add_document_reference_to_lab_report(
            {}, "fake_encoded_data", "application/pdf"
        )
    )
    self.assertEqual(result, {})
    self.assertFalse(modified)


if __name__ == "__main__":
  absltest.main()

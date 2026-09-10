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

  def _create_mock_bundle_proto(self, composition_profile):
    mock_bundle_proto = mock.Mock()

    mock_composition_entry = mock.Mock()
    mock_composition_entry.resource.HasField.side_effect = (
        lambda field: field == "composition"
    )
    mock_composition_entry.resource.composition.meta.profile = [
        mock.Mock(value=composition_profile)
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

    return mock_bundle_proto

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  @mock.patch.object(json_format, "print_fhir_to_json_string", autospec=True)
  def test_add_document_reference_to_bundle_diagnostic_report_success(
      self, mock_print_fhir, mock_json_to_proto
  ):
    bundle_json = {"entry": []}  # Content doesn't matter for mock
    updated_bundle_json = {"entry": [{"fake": "updated"}]}
    mock_print_fhir.return_value = json.dumps(updated_bundle_json)

    mock_bundle_proto = self._create_mock_bundle_proto(
        composition_profile="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportRecord",
    )
    mock_json_to_proto.return_value = mock_bundle_proto

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        bundle_json, "fake_encoded_data", "application/pdf"
    )

    self.assertEqual(result, updated_bundle_json)
    self.assertTrue(modified)

    # Verify DocumentReference details
    mock_doc_ref_copy = (
        mock_bundle_proto.entry.add.return_value.resource.document_reference.CopyFrom
    )
    mock_doc_ref_copy.assert_called_once()
    doc_ref_proto = mock_doc_ref_copy.call_args[0][0]
    self.assertEqual(
        doc_ref_proto.content[0].attachment.data.value, b"fake_encoded_data"
    )
    self.assertEqual(
        doc_ref_proto.type.coding[0].code.value, "721981007"
    )  # Diagnostic studies report
    self.assertEqual(
        doc_ref_proto.type.coding[0].display.value, "Diagnostic studies report"
    )

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  @mock.patch.object(json_format, "print_fhir_to_json_string", autospec=True)
  def test_add_document_reference_to_bundle_prescription_success(
      self, mock_print_fhir, mock_json_to_proto
  ):
    bundle_json = {"entry": []}
    updated_bundle_json = {"entry": [{"fake": "updated"}]}
    mock_print_fhir.return_value = json.dumps(updated_bundle_json)

    mock_bundle_proto = self._create_mock_bundle_proto(
        composition_profile="https://nrces.in/ndhm/fhir/r4/StructureDefinition/PrescriptionRecord",
    )
    mock_json_to_proto.return_value = mock_bundle_proto

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        bundle_json, "fake_encoded_data", "application/pdf"
    )

    self.assertEqual(result, updated_bundle_json)
    self.assertTrue(modified)

    # Verify DocumentReference details
    mock_doc_ref_copy = (
        mock_bundle_proto.entry.add.return_value.resource.document_reference.CopyFrom
    )
    mock_doc_ref_copy.assert_called_once()
    doc_ref_proto = mock_doc_ref_copy.call_args[0][0]
    self.assertEqual(
        doc_ref_proto.content[0].attachment.data.value, b"fake_encoded_data"
    )
    self.assertEqual(
        doc_ref_proto.type.coding[0].code.value, "440545006"
    )  # Prescription record
    self.assertEqual(
        doc_ref_proto.type.coding[0].display.value, "Prescription record"
    )

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  @mock.patch.object(json_format, "print_fhir_to_json_string", autospec=True)
  def test_add_document_reference_to_bundle_immunization_success(
      self, mock_print_fhir, mock_json_to_proto
  ):
    bundle_json = {"entry": []}
    updated_bundle_json = {"entry": [{"fake": "updated"}]}
    mock_print_fhir.return_value = json.dumps(updated_bundle_json)

    mock_bundle_proto = self._create_mock_bundle_proto(
        composition_profile=(
            "https://nrces.in/ndhm/fhir/r4/StructureDefinition/ImmunizationRecord"
        ),
    )
    mock_json_to_proto.return_value = mock_bundle_proto

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        bundle_json, "fake_encoded_data", "application/pdf"
    )

    self.assertEqual(result, updated_bundle_json)
    self.assertTrue(modified)

    mock_doc_ref_copy = (
        mock_bundle_proto.entry.add.return_value.resource.document_reference.CopyFrom
    )
    mock_doc_ref_copy.assert_called_once()
    doc_ref_proto = mock_doc_ref_copy.call_args[0][0]
    self.assertEqual(
        doc_ref_proto.content[0].attachment.data.value, b"fake_encoded_data"
    )
    self.assertEqual(
        doc_ref_proto.type.coding[0].code.value, "41000179103"
    )  # Immunization record
    self.assertEqual(
        doc_ref_proto.type.coding[0].display.value, "Immunization record"
    )

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  def test_add_document_reference_to_bundle_skips_unsupported_profile(
      self, mock_json_to_proto
  ):
    bundle_json = {"entry": []}
    mock_bundle_proto = self._create_mock_bundle_proto(
        composition_profile="https://some-other-profile"
    )
    mock_json_to_proto.return_value = mock_bundle_proto

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        bundle_json, "fake_encoded_data", "application/pdf"
    )

    self.assertEqual(result, bundle_json)
    self.assertFalse(modified)

  @mock.patch.object(json_format, "json_fhir_string_to_proto", autospec=True)
  def test_add_document_reference_to_bundle_missing_patient(
      self, mock_json_to_proto
  ):
    bundle_json = {"entry": []}
    mock_bundle_proto = mock.Mock()
    mock_composition_entry = mock.Mock()
    mock_composition_entry.resource.HasField.side_effect = (
        lambda field: field == "composition"
    )
    mock_composition_entry.resource.composition.meta.profile = [
        mock.Mock(
            value=(
                "https://nrces.in/ndhm/fhir/r4/StructureDefinition/"
                "DiagnosticReportRecord"
            )
        )
    ]
    mock_section = mock.Mock()
    mock_section.entry = mock.Mock()
    mock_composition_entry.resource.composition.section = [mock_section]

    mock_entry_list = [mock_composition_entry]

    mock_bundle_proto.entry = mock.MagicMock()
    mock_bundle_proto.entry.__iter__.return_value = iter(mock_entry_list)
    mock_bundle_proto.entry.__getitem__.side_effect = (
        lambda idx: mock_entry_list[idx]
    )

    mock_json_to_proto.return_value = mock_bundle_proto

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        bundle_json, "fake_encoded_data", "application/pdf"
    )

    self.assertEqual(result, bundle_json)
    self.assertFalse(modified)

  def test_add_document_reference_to_bundle_empty_bundle(self):
    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        None, "fake_encoded_data", "application/pdf"
    )
    self.assertIsNone(result)
    self.assertFalse(modified)

    result, modified = abdm_bundle_enricher.add_document_reference_to_bundle(
        {}, "fake_encoded_data", "application/pdf"
    )
    self.assertEqual(result, {})
    self.assertFalse(modified)


if __name__ == "__main__":
  absltest.main()

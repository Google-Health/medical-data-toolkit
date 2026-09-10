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
"""Unit tests for RSNA Playbook Radiology LOINC query engine."""

from absl.testing import absltest
from absl.testing import parameterized

from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc.radiology import config
from src.document_to_fhir.core.medical_coding.loinc.radiology import mappings
from src.document_to_fhir.core.medical_coding.loinc.radiology import query as rsna_mapper

# Test dataset fixture simulating precomputed RSNA Playbook records
TEST_RECORDS = [
    {
        config.FIELD_LOINC_NUM: "24627-2",
        config.FIELD_LONG_COMMON_NAME: "XR Chest PA and Lateral",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "PA and Lateral",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "xr chest pa and lateral",
    },
    {
        config.FIELD_LOINC_NUM: "30745-4",
        config.FIELD_LONG_COMMON_NAME: "XR Chest PA",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "PA",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "xr chest pa",
    },
    {
        config.FIELD_LOINC_NUM: "30746-2",
        config.FIELD_LONG_COMMON_NAME: "XR Chest AP",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "AP",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "xr chest ap",
    },
    {
        config.FIELD_LOINC_NUM: "30932-8",
        config.FIELD_LONG_COMMON_NAME: "CT Chest with contrast",
        config.FIELD_MODALITY_TYPE: "CT",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "W contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "ct chest with contrast",
    },
    {
        config.FIELD_LOINC_NUM: "30934-4",
        config.FIELD_LONG_COMMON_NAME: "CT Chest without contrast",
        config.FIELD_MODALITY_TYPE: "CT",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "ct chest without contrast",
    },
    {
        config.FIELD_LOINC_NUM: "24725-4",
        config.FIELD_LONG_COMMON_NAME: "CT Head without contrast",
        config.FIELD_MODALITY_TYPE: "CT",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Head",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "ct head without contrast",
    },
    {
        config.FIELD_LOINC_NUM: "103898-3",
        config.FIELD_LONG_COMMON_NAME: "CTA Head and Neck with contrast",
        config.FIELD_MODALITY_TYPE: "CT",
        config.FIELD_MODALITY_SUBTYPE: "Angio",
        config.FIELD_REGION_IMAGED: "Head",
        config.FIELD_CONTRAST_USED: "W contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: (
            "cta head and neck with contrast ct angio head"
        ),
    },
    {
        config.FIELD_LOINC_NUM: "24629-8",
        config.FIELD_LONG_COMMON_NAME: "MRI Brain without contrast",
        config.FIELD_MODALITY_TYPE: "MR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Brain",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "mri brain without contrast mr brain",
    },
    {
        config.FIELD_LOINC_NUM: "24882-3",
        config.FIELD_LONG_COMMON_NAME: "XR Knee Left AP and Lateral",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Knee",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "PA and Lateral",
        config.FIELD_LATERALITY: "Left",
        config.FIELD_SEARCH_TEXT: "xr knee left ap and lateral",
    },
    {
        config.FIELD_LOINC_NUM: "103912-2",
        config.FIELD_LONG_COMMON_NAME: "CT Cervical spine without contrast",
        config.FIELD_MODALITY_TYPE: "CT",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Spine Cervical",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: (
            "ct cervical spine without contrast ct spine cervical wo contrast"
        ),
    },
    {
        config.FIELD_LOINC_NUM: "24958-1",
        config.FIELD_LONG_COMMON_NAME: "XR Wrist 2 Views",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Extremity upper Wrist",
        config.FIELD_CONTRAST_USED: "WO contrast",
        config.FIELD_VIEW_TYPE: "PA and Lateral",
        config.FIELD_LATERALITY: "",
        config.FIELD_SEARCH_TEXT: "xr wrist 2 views xr extremity upper wrist",
    },
]


class RsnaPlaybookQueryEngineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.query_engine = rsna_mapper.RsnaPlaybookQueryEngine(TEST_RECORDS)

  @parameterized.named_parameters(
      dict(
          testcase_name="exact_tuple_match_ct_chest",
          report_fields={
              "exam_name": "CT Chest with contrast",
              "modality_type": "CT",
              "region_imaged": "Chest",
              "contrast_used": "W contrast",
              "findings": "Lungs are clear without focal consolidation.",
              "conclusion": "Normal CT examination of the chest.",
          },
          expected_loinc="30932-8",
          expected_common_name="CT Chest with contrast",
      ),
      dict(
          testcase_name="xr_view_type_matching_pa_and_lateral",
          report_fields={
              "exam_name": "Chest X-Ray PA and Lateral View",
              "modality_type": "XR",
              "region_imaged": "Chest",
              "view_type": "PA and Lateral",
          },
          expected_loinc="24627-2",
          expected_common_name="XR Chest PA and Lateral",
      ),
      dict(
          testcase_name="xr_view_type_matching_pa_only",
          report_fields={
              "exam_name": "Chest X-Ray PA View",
              "modality_type": "XR",
              "region_imaged": "Chest",
              "view_type": "PA",
          },
          expected_loinc="30745-4",
          expected_common_name="XR Chest PA",
      ),
      dict(
          testcase_name="laterality_match_xr_knee",
          report_fields={
              "exam_name": "XR Left Knee AP and Lateral",
              "modality_type": "XR",
              "region_imaged": "Knee",
              "contrast_used": "WO contrast",
              "view_type": "PA and Lateral",
              "laterality": "Left",
          },
          expected_loinc="24882-3",
          expected_common_name="XR Knee Left AP and Lateral",
      ),
      dict(
          testcase_name="preserves_existing_loinc_codes",
          report_fields={
              "exam_name": "CT Head without contrast",
              "modality_type": "CT",
              "region_imaged": "Head",
              "procedure_loinc_code": "CUSTOM-123",
              "findings": "Findings text",
              "conclusion": "Impression text",
          },
          expected_loinc="CUSTOM-123",
          expected_common_name=None,
      ),
      dict(
          testcase_name="unstandardized_modality_matching",
          report_fields={
              "exam_name": "Magnetic Resonance Img of Brain",
              "modality_type": "Magnetic Resonance Img",
              "region_imaged": "Brain",
              "contrast_used": "without contrast",
          },
          expected_loinc="24629-8",
          expected_common_name="MRI Brain without contrast",
      ),
      dict(
          testcase_name="modality_type_only_maps_to_standard_code",
          report_fields={
              "exam_name": "CT Head without contrast",
              "modality_type": "CT",
              "modality_subtype": None,
              "region_imaged": "Head",
              "contrast_used": "WO contrast",
          },
          expected_loinc="24725-4",
          expected_common_name="CT Head without contrast",
      ),
      dict(
          testcase_name="modality_type_and_subtype_maps_to_specialized_code",
          report_fields={
              "exam_name": "CTA Head and Neck with contrast",
              "modality_type": "CT",
              "modality_subtype": "Angio",
              "region_imaged": "Head",
              "contrast_used": "W contrast",
          },
          expected_loinc="103898-3",
          expected_common_name="CTA Head and Neck with contrast",
      ),
      dict(
          testcase_name="combined_region_and_focus_matching_cervical_spine",
          report_fields={
              "exam_name": "CT Cervical Spine without contrast",
              "modality_type": "CT",
              "region_imaged": "Cervical Spine",
              "contrast_used": "without contrast",
          },
          expected_loinc="103912-2",
          expected_common_name="CT Cervical spine without contrast",
      ),
      dict(
          testcase_name="combined_region_and_focus_matching_wrist",
          report_fields={
              "exam_name": "XR Wrist 2 Views",
              "modality_type": "XR",
              "region_imaged": "Wrist",
              "view_type": "2 Views",
          },
          expected_loinc="24958-1",
          expected_common_name="XR Wrist 2 Views",
      ),
      dict(
          testcase_name=(
              "similarity_score_matches_when_contrast_and_view_omitted"
          ),
          report_fields={
              "exam_name": "CT Chest examination with contrast",
              "modality_type": "CT",
              "region_imaged": "Chest",
          },
          expected_loinc="30932-8",
          expected_common_name="CT Chest with contrast",
      ),
      dict(
          testcase_name="similarity_score_filters_candidates_by_modality_type",
          report_fields={
              "exam_name": "XR Chest PA and Lateral view",
              "modality_type": "CT",
              "region_imaged": "Chest",
          },
          expected_loinc="30932-8",
          expected_common_name="CT Chest with contrast",
      ),
      dict(
          testcase_name=(
              "similarity_score_filters_candidates_by_modality_subtype"
          ),
          report_fields={
              "exam_name": "CT Head without contrast scan",
              "modality_type": "CT",
              "modality_subtype": "Angio",
              "region_imaged": "Head",
          },
          expected_loinc="103898-3",
          expected_common_name="CTA Head and Neck with contrast",
      ),
      dict(
          testcase_name="similarity_score_filters_candidates_by_region",
          report_fields={
              "exam_name": "XR Chest PA and Lateral",
              "modality_type": "XR",
              "region_imaged": "Knee",
          },
          expected_loinc="24882-3",
          expected_common_name="XR Knee Left AP and Lateral",
      ),
  )
  def test_map_report(
      self,
      report_fields: dict[str, str | None],
      expected_loinc: str,
      expected_common_name: str | None = None,
  ):
    report = medical_documents.RadiologyReport(
        patient=resources.Patient(name="Test Patient"),
        **report_fields,
    )
    self.query_engine.map_report(report)

    self.assertEqual(report.procedure_loinc_code, expected_loinc)
    if expected_common_name is not None:
      self.assertEqual(
          report.procedure_loinc_common_name, expected_common_name
      )

  @parameterized.parameters(
      ("CTA", "CT"),
      ("CT", "CT"),
      ("Computed Tomography", "CT"),
      ("cat scan", "CT"),
      ("MRA", "MR"),
      ("MRV", "MR"),
      ("fMRI", "MR"),
      ("Magnetic Resonance Imaging", "MR"),
      ("Chest X-Ray", "XR"),
      ("plain film", "XR"),
      ("Ultrasound", "US"),
      ("USG", "US"),
      ("DBT", "MG"),
      ("Mammogram", "MG"),
      ("", ""),
      (None, ""),
  )
  def test_normalize_attribute_modality_type(self, raw_input, expected):
    self.assertEqual(
        rsna_mapper.normalize_attribute(raw_input, mappings.MODALITY_PATTERNS),
        expected,
    )

  @parameterized.parameters(
      ("Angio", "Angio"),
      ("Angiography", "Angio"),
      ("Venography", "Angio"),
      ("CTA", "Angio"),
      ("MRA", "Angio"),
      ("Doppler", "Doppler"),
      ("Color Doppler", "Doppler"),
      ("Duplex", "Doppler"),
      ("Tomosynthesis", "tomosynthesis"),
      ("DBT", "tomosynthesis"),
      ("3D Mammogram", "tomosynthesis"),
      ("SPECT", "SPECT"),
      ("Single Photon", "SPECT"),
      ("Portable", "Portable"),
      ("Bedside", "Portable"),
      ("Functional", "Functional"),
      ("fMRI", "Functional"),
      ("Spectroscopy", "Spectroscopy"),
      ("MRS", "Spectroscopy"),
      ("Elastography", "Elastography"),
      ("", ""),
      (None, ""),
  )
  def test_normalize_attribute_modality_subtype(self, raw_input, expected):
    self.assertEqual(
        rsna_mapper.normalize_attribute(
            raw_input, mappings.MODALITY_SUBTYPE_PATTERNS
        ),
        expected,
    )

  def test_match_by_similarity_score_candidate_filtering(self):
    # Query with modality "CT" and region "Head" - even if exam name has heavy
    # "XR Knee" tokens
    query = rsna_mapper._NormalizedRsnaQuery(
        modality="ct",
        subtype="",
        region="head",
        contrast="",
        view="",
        laterality="",
    )
    loinc_num, _ = self.query_engine._match_by_similarity_score(
        query, exam_name="XR Knee Left AP Lateral head without"
    )
    # Must match CT Head record (24725-4) over 103898-3 because candidate
    # filtering excludes XR Knee despite 5 token matches
    self.assertEqual(loinc_num, "24725-4")

  def test_match_by_similarity_score_no_matching_candidate_returns_none(self):
    # Query with non-existent modality
    query = rsna_mapper._NormalizedRsnaQuery(
        modality="unknown_modality",
        subtype="",
        region="head",
        contrast="",
        view="",
        laterality="",
    )
    loinc_num, loinc_name = self.query_engine._match_by_similarity_score(
        query, exam_name="CT Head without contrast"
    )
    self.assertIsNone(loinc_num)
    self.assertIsNone(loinc_name)


if __name__ == "__main__":
  absltest.main()

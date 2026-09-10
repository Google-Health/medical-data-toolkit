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
"""Unit tests for RSNA Playbook builder."""

from absl.testing import absltest
from absl.testing import parameterized

from src.document_to_fhir.core.medical_coding.loinc.radiology import builder
from src.document_to_fhir.core.medical_coding.loinc.radiology import config


def _make_playbook_row(
    loinc_num: str,
    part_type: str,
    part_name: str,
    long_name: str = "Test Procedure",
    rpid: str = "",
) -> dict[str, str]:
  """Constructs an official LoincRsnaRadiologyPlaybook.csv row for testing."""
  return {
      config.COL_LOINC_NUMBER: loinc_num,
      config.COL_LONG_COMMON_NAME: long_name,
      config.COL_PART_TYPE_NAME: part_type,
      config.COL_PART_NAME: part_name,
      config.COL_RPID: rpid,
  }


# Canonical sample rows simulating raw LoincRsnaRadiologyPlaybook.csv input
OFFICIAL_SAMPLE_ROWS = [
    _make_playbook_row(
        loinc_num="24627-2",
        part_type=config.PART_TYPE_MODALITY_TYPE,
        part_name="XR",
        long_name="XR Chest PA and Lateral",
        rpid="RPID12",
    ),
    _make_playbook_row(
        loinc_num="24627-2",
        part_type=config.PART_TYPE_REGION_IMAGED,
        part_name="Chest",
        long_name="XR Chest PA and Lateral",
        rpid="RPID12",
    ),
    _make_playbook_row(
        loinc_num="24627-2",
        part_type=config.PART_TYPE_VIEW_TYPE,
        part_name="PA and Lateral",
        long_name="XR Chest PA and Lateral",
        rpid="RPID12",
    ),
    _make_playbook_row(
        loinc_num="30932-8",
        part_type=config.PART_TYPE_MODALITY_TYPE,
        part_name="CT",
        long_name="CT Chest with contrast",
        rpid="RPID45",
    ),
    _make_playbook_row(
        loinc_num="30932-8",
        part_type=config.PART_TYPE_REGION_IMAGED,
        part_name="Chest",
        long_name="CT Chest with contrast",
        rpid="RPID45",
    ),
    _make_playbook_row(
        loinc_num="30932-8",
        part_type=config.PART_TYPE_TIMING,
        part_name="W",
        long_name="CT Chest with contrast",
        rpid="RPID45",
    ),
    _make_playbook_row(
        loinc_num="30932-8",
        part_type=config.PART_TYPE_SUBSTANCE_GIVEN,
        part_name="contrast",
        long_name="CT Chest with contrast",
        rpid="RPID45",
    ),
]


class RsnaPlaybookBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.builder = builder.RsnaPlaybookBuilder()

  def test_process_official_playbook_table(self):
    rank_map = {"24627-2": 5, "30932-8": 20}
    processed = self.builder.process_official_playbook_table(
        OFFICIAL_SAMPLE_ROWS, common_test_rank_map=rank_map, max_rank=50
    )

    self.assertLen(processed, 2)
    xr_entry = next(
        r for r in processed if r[config.FIELD_LOINC_NUM] == "24627-2"
    )
    self.assertEqual(xr_entry[config.FIELD_MODALITY_TYPE], "XR")
    self.assertEqual(xr_entry[config.FIELD_MODALITY_SUBTYPE], "")
    self.assertEqual(xr_entry[config.FIELD_REGION_IMAGED], "Chest")
    self.assertEqual(xr_entry[config.FIELD_VIEW_TYPE], "PA and Lateral")
    self.assertEqual(xr_entry[config.FIELD_RPID], "RPID12")
    self.assertEqual(xr_entry[config.FIELD_COMMON_TEST_RANK], "5")

    ct_entry = next(
        r for r in processed if r[config.FIELD_LOINC_NUM] == "30932-8"
    )
    self.assertEqual(ct_entry[config.FIELD_MODALITY_TYPE], "CT")
    self.assertEqual(ct_entry[config.FIELD_MODALITY_SUBTYPE], "")
    self.assertEqual(ct_entry[config.FIELD_REGION_IMAGED], "Chest")
    self.assertEqual(ct_entry[config.FIELD_CONTRAST_USED], "W contrast")
    self.assertEqual(ct_entry[config.FIELD_RPID], "RPID45")
    self.assertEqual(ct_entry[config.FIELD_COMMON_TEST_RANK], "20")

  def test_process_official_playbook_table_with_max_rank_filter(self):
    rows = [
        _make_playbook_row("24627-2", config.PART_TYPE_MODALITY_TYPE, "XR"),
        _make_playbook_row("99999-9", config.PART_TYPE_MODALITY_TYPE, "XR"),
    ]
    rank_map = {"24627-2": 10, "99999-9": 5000}
    processed = self.builder.process_official_playbook_table(
        rows, common_test_rank_map=rank_map, max_rank=100
    )
    self.assertLen(processed, 1)
    self.assertEqual(processed[0][config.FIELD_LOINC_NUM], "24627-2")

  @parameterized.named_parameters(
      dict(
          testcase_name="extracts_modality_subtype_angio",
          loinc_number="103898-3",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_MODALITY_SUBTYPE, "Angio"),
              (config.PART_TYPE_REGION_IMAGED, "Head"),
              (config.PART_TYPE_SUBSTANCE_GIVEN, "Contrast"),
          ],
          expected_indexed_record_fields={
              config.FIELD_MODALITY_TYPE: "CT",
              config.FIELD_MODALITY_SUBTYPE: "Angio",
              config.FIELD_REGION_IMAGED: "Head",
              config.FIELD_CONTRAST_USED: "Contrast",
          },
      ),
      dict(
          testcase_name="handles_standard_modality_without_subtype",
          loinc_number="103903-1",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_REGION_IMAGED, "Head"),
              (config.PART_TYPE_TIMING, "WO"),
          ],
          expected_indexed_record_fields={
              config.FIELD_MODALITY_TYPE: "CT",
              config.FIELD_MODALITY_SUBTYPE: "",
              config.FIELD_REGION_IMAGED: "Head",
              config.FIELD_CONTRAST_USED: "WO",
          },
      ),
      dict(
          testcase_name="blends_macro_region_spine_with_focus_cervical",
          loinc_number="103912-2",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_REGION_IMAGED, "Spine"),
              (config.PART_TYPE_IMAGING_FOCUS, "Cervical"),
          ],
          expected_indexed_record_fields={
              config.FIELD_REGION_IMAGED: "Spine Cervical",
          },
      ),
      dict(
          testcase_name="blends_macro_region_upper_extremity_with_focus_wrist",
          loinc_number="24958-1",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "XR"),
              (config.PART_TYPE_REGION_IMAGED, "Extremity upper"),
              (config.PART_TYPE_IMAGING_FOCUS, "Wrist"),
          ],
          expected_indexed_record_fields={
              config.FIELD_REGION_IMAGED: "Extremity upper Wrist",
          },
      ),
      dict(
          testcase_name="deduplicates_redundant_region_and_focus_head",
          loinc_number="103903-1",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_REGION_IMAGED, "Head"),
              (config.PART_TYPE_IMAGING_FOCUS, "Head"),
          ],
          expected_indexed_record_fields={
              config.FIELD_REGION_IMAGED: "Head",
          },
      ),
      dict(
          testcase_name="blends_timing_with_and_substance_contrast",
          loinc_number="30932-8",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_TIMING, "W"),
              (config.PART_TYPE_SUBSTANCE_GIVEN, "contrast"),
          ],
          expected_indexed_record_fields={
              config.FIELD_CONTRAST_USED: "W contrast",
          },
      ),
      dict(
          testcase_name="blends_timing_without_and_with_and_substance_contrast",
          loinc_number="103901-5",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "CT"),
              (config.PART_TYPE_TIMING, "WO & W"),
              (config.PART_TYPE_SUBSTANCE_GIVEN, "contrast"),
          ],
          expected_indexed_record_fields={
              config.FIELD_CONTRAST_USED: "WO & W contrast",
          },
      ),
      dict(
          testcase_name="blends_view_projection_lateral_and_aggregation_1_view",
          loinc_number="100655-0",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "XR"),
              (config.PART_TYPE_VIEW_TYPE, "Lateral"),
              (config.PART_TYPE_VIEW_AGGREGATION, "1 View"),
          ],
          expected_indexed_record_fields={
              config.FIELD_VIEW_TYPE: "Lateral 1 View",
          },
      ),
      dict(
          testcase_name="blends_all_part_combinations_simultaneously",
          loinc_number="99999-1",
          input_playbook_part_tuples=[
              (config.PART_TYPE_MODALITY_TYPE, "XR"),
              (config.PART_TYPE_REGION_IMAGED, "Extremity upper"),
              (config.PART_TYPE_IMAGING_FOCUS, "Wrist"),
              (config.PART_TYPE_TIMING, "W"),
              (config.PART_TYPE_SUBSTANCE_GIVEN, "contrast"),
              (config.PART_TYPE_VIEW_TYPE, "PA and Lateral"),
              (config.PART_TYPE_VIEW_AGGREGATION, "2 Views"),
              (config.PART_TYPE_LATERALITY, "Right"),
          ],
          expected_indexed_record_fields={
              config.FIELD_MODALITY_TYPE: "XR",
              config.FIELD_REGION_IMAGED: "Extremity upper Wrist",
              config.FIELD_CONTRAST_USED: "W contrast",
              config.FIELD_VIEW_TYPE: "PA and Lateral 2 Views",
              config.FIELD_LATERALITY: "Right",
          },
      ),
  )
  def test_process_official_playbook_part_blending(
      self,
      loinc_number: str,
      input_playbook_part_tuples: list[tuple[str, str]],
      expected_indexed_record_fields: dict[str, str],
  ):
    rows = [
        _make_playbook_row(
            loinc_num=loinc_number, part_type=part_type, part_name=part_name
        )
        for part_type, part_name in input_playbook_part_tuples
    ]
    processed = self.builder.process_official_playbook_table(rows)
    self.assertLen(processed, 1)
    entry = processed[0]
    self.assertEqual(entry[config.FIELD_LOINC_NUM], loinc_number)
    for field_key, expected_val in expected_indexed_record_fields.items():
      self.assertEqual(
          entry[field_key],
          expected_val,
          f"Field '{field_key}' mismatch in testcase",
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="distinct_anatomic_regions",
          first_part="Spine",
          second_part="Cervical",
          expected_combined_part="Spine Cervical",
      ),
      dict(
          testcase_name="timing_and_substance",
          first_part="W",
          second_part="contrast",
          expected_combined_part="W contrast",
      ),
      dict(
          testcase_name="view_projection_and_aggregation",
          first_part="PA and Lateral",
          second_part="2 Views",
          expected_combined_part="PA and Lateral 2 Views",
      ),
      dict(
          testcase_name="first_part_none",
          first_part=None,
          second_part="contrast",
          expected_combined_part="contrast",
      ),
      dict(
          testcase_name="second_part_none",
          first_part="Spine",
          second_part=None,
          expected_combined_part="Spine",
      ),
      dict(
          testcase_name="first_part_empty_string",
          first_part="",
          second_part="AP",
          expected_combined_part="AP",
      ),
      dict(
          testcase_name="second_part_empty_string",
          first_part="Lateral",
          second_part="",
          expected_combined_part="Lateral",
      ),
      dict(
          testcase_name="both_parts_none",
          first_part=None,
          second_part=None,
          expected_combined_part="",
      ),
      dict(
          testcase_name="both_parts_empty_strings",
          first_part="",
          second_part="",
          expected_combined_part="",
      ),
      dict(
          testcase_name="identical_parts_case_insensitive",
          first_part="Head",
          second_part="head",
          expected_combined_part="Head",
      ),
      dict(
          testcase_name="first_part_contains_second_part",
          first_part="2 Views",
          second_part="Views",
          expected_combined_part="2 Views",
      ),
      dict(
          testcase_name="second_part_contains_first_part",
          first_part="Views",
          second_part="2 Views",
          expected_combined_part="2 Views",
      ),
  )
  def test_combine_parts(
      self,
      first_part: str | None,
      second_part: str | None,
      expected_combined_part: str,
  ):
    self.assertEqual(
        builder._combine_parts(first_part, second_part),
        expected_combined_part,
    )

  def test_save_to_csv(self):
    records = [{
        config.FIELD_LOINC_NUM: "24627-2",
        config.FIELD_LONG_COMMON_NAME: "XR Chest PA and Lateral",
        config.FIELD_MODALITY_TYPE: "XR",
        config.FIELD_MODALITY_SUBTYPE: "",
        config.FIELD_REGION_IMAGED: "Chest",
        config.FIELD_CONTRAST_USED: "",
        config.FIELD_VIEW_TYPE: "PA and Lateral",
        config.FIELD_LATERALITY: "",
        config.FIELD_COMMON_TEST_RANK: "1",
        config.FIELD_RPID: "RPID12",
        config.FIELD_SEARCH_TEXT: "xr chest pa and lateral",
    }]
    temp_dir = self.create_tempdir()
    output_file = f"{temp_dir.full_path}/test_output.csv"
    self.builder.save_to_csv(records, output_file)
    with open(output_file, "r") as f:
      content = f.read()
    self.assertIn("24627-2", content)
    self.assertIn("XR Chest PA and Lateral", content)


if __name__ == "__main__":
  absltest.main()

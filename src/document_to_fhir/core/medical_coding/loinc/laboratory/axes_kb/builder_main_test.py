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
"""Tests for builder_main.py."""

import pathlib
from unittest import mock

from absl import app
from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pandas

from src.document_to_fhir.common import model_client
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb import builder_main
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import builder as core_analyte_builder
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import builder as property_builder
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system import builder as system_builder


class BuilderMainTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.output_dir = self.create_tempdir().full_path
    self.enter_context(
        flagsaver.flagsaver(
            loinc_csv_path="/path/to/LoincTableCore.csv",
            output_csv_folder=self.output_dir,
            client_type="gemini",
            model_name="gemini-2.5-flash",
            max_rank=2000,
            limit=0,
            workers=8,
            batch_size=4,
            max_attempts=4,
            resume=True,
            filename_suffix="",
        )
    )

  @parameterized.named_parameters(
      (
          "expand_all",
          ["all"],
          ["core_analyte", "property", "system"],
      ),
      (
          "deduplicate_and_normalize",
          [" Property ", "SYSTEM", "property"],
          ["property", "system"],
      ),
  )
  def test_resolve_axes_success(
      self, requested: list[str], expected: list[str]
  ):
    self.assertEqual(builder_main._resolve_axes(requested), expected)

  @parameterized.named_parameters(
      ("typo_axis", ["core_analyt"]),
      ("empty_axes", ["", "   "]),
  )
  def test_resolve_axes_invalid_raises_usage_error(
      self, invalid_axes: list[str]
  ):
    with self.assertRaises(app.UsageError):
      builder_main._resolve_axes(invalid_axes)

  @parameterized.named_parameters(
      (
          "core_analyte_default",
          "core_analyte",
          0,
          "",
          "gemini-2.5-flash",
          "analyte_records__LoincTableCore__top_2000__gemini__gemini-2.5-flash__20260918.csv",
      ),
      (
          "core_analyte_with_limit_and_suffix",
          "core_analyte",
          50,
          "v1",
          "vertex_ai/gemini:pro",
          "analyte_records__LoincTableCore__top_2000__limit_50__gemini__vertex_ai_gemini_pro__20260918__v1.csv",
      ),
      (
          "property_default",
          "property",
          0,
          "",
          "gemini-2.5-flash",
          "loinc_axis_property_kb_top_2000__gemini__gemini-2.5-flash__20260918.csv",
      ),
      (
          "system_with_limit_and_suffix",
          "system",
          100,
          "sample",
          "gemini-2.5-flash",
          "loinc_axis_system_kb_top_2000__limit_100__gemini__gemini-2.5-flash__20260918__sample.csv",
      ),
  )
  def test_get_output_filename(
      self,
      axis: str,
      limit: int,
      suffix: str,
      model_name: str,
      expected_filename: str,
  ):
    with flagsaver.flagsaver(
        limit=limit,
        model_name=model_name,
        filename_suffix=suffix,
    ):
      actual = builder_main._get_output_filename(axis, "20260918")
      self.assertEqual(actual, expected_filename)

  @parameterized.named_parameters(
      ("core_analyte", "core_analyte"),
      ("property", "property"),
      ("system", "system"),
  )
  @mock.patch.object(
      system_builder, "SystemKBBuilder", autospec=True, spec_set=True
  )
  @mock.patch.object(
      property_builder, "PropertyKBBuilder", autospec=True, spec_set=True
  )
  @mock.patch.object(
      core_analyte_builder, "AnalyteKBBuilder", autospec=True, spec_set=True
  )
  def test_build_axis_dispatches_to_expected_builder(
      self,
      axis: str,
      mock_analyte_builder_cls,
      mock_property_builder_cls,
      mock_system_builder_cls,
  ):
    mock_client = mock.create_autospec(
        model_client.LLMClient, instance=True, spec_set=True
    )
    df_loinc = pandas.DataFrame({"LOINC_NUM": ["2345-7"]})

    builder_main._build_axis(axis, df_loinc, mock_client, "20260918")

    if axis == "core_analyte":
      mock_analyte_builder_cls.assert_called_once_with(
          df_loinc=df_loinc,
          client_inst=mock_client,
          workers=8,
          batch_size=4,
          max_attempts=4,
      )
      mock_analyte_builder_cls.return_value.build_kb.assert_called_once()
    elif axis == "property":
      mock_property_builder_cls.assert_called_once_with(
          df_loinc=df_loinc,
          client=mock_client,
          workers=8,
          max_attempts=4,
      )
      mock_property_builder_cls.return_value.build_kb.assert_called_once()
    elif axis == "system":
      mock_system_builder_cls.assert_called_once_with(
          df_loinc=df_loinc,
          client=mock_client,
          workers=8,
          max_attempts=4,
      )
      mock_system_builder_cls.return_value.build_kb.assert_called_once()

  @parameterized.named_parameters(
      (
          "too_many_positional_args",
          ["builder_main.py", "extra_arg"],
          ["core_analyte"],
      ),
      (
          "unknown_axis_before_io",
          ["builder_main.py"],
          ["core_analyt"],
      ),
  )
  @mock.patch.object(utils, "read_loinc_kb", autospec=True, spec_set=True)
  def test_main_invalid_invocation_raises_usage_error(
      self, argv: list[str], axes: list[str], mock_read_loinc_kb
  ):
    with flagsaver.flagsaver(axes=axes):
      with self.assertRaises(app.UsageError):
        builder_main.main(argv)

    mock_read_loinc_kb.assert_not_called()

  @mock.patch.object(builder_main, "_build_axis", autospec=True, spec_set=True)
  @mock.patch.object(
      model_client, "create_llm_client", autospec=True, spec_set=True
  )
  @mock.patch.object(utils, "read_loinc_kb", autospec=True, spec_set=True)
  def test_main_loads_additional_loinc_codes_and_builds_axes(
      self, mock_read_loinc_kb, unused_mock_create_client, mock_build_axis
  ):
    mock_read_loinc_kb.return_value = pandas.DataFrame(
        {"LOINC_NUM": ["2345-7"]}
    )
    codes_file = pathlib.Path(self.output_dir) / "additional_codes.txt"
    codes_file.write_text(
        " 2345-7 \n\n  718-7\n   \n4544-3\n", encoding="utf-8"
    )

    with flagsaver.flagsaver(
        axes=["property", "system"],
        additional_loinc_codes_file=str(codes_file),
    ):
      builder_main.main(["builder_main.py"])

    mock_read_loinc_kb.assert_called_once_with(
        file_path="/path/to/LoincTableCore.csv",
        max_rank=2000,
        limit=0,
        additional_loinc_codes=["2345-7", "718-7", "4544-3"],
    )
    self.assertEqual(mock_build_axis.call_count, 2)
    passed_date_str = mock_build_axis.call_args[0][3]
    self.assertRegex(passed_date_str, r"^\d{8}$")


if __name__ == "__main__":
  flags.FLAGS.set_default("loinc_csv_path", "/tmp/loinc.csv")
  flags.FLAGS.set_default("output_csv_folder", "/tmp/out")
  flags.FLAGS.set_default("client_type", "gemini")
  flags.FLAGS.set_default("model_name", "gemini-2.5-flash")
  absltest.main()

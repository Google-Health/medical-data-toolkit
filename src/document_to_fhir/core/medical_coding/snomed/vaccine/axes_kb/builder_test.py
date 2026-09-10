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
"""Unit tests for the sequential 3-stage Vaccine Axes KB Builder."""

import dataclasses
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine.axes_kb import builder
from src.document_to_fhir.core.medical_coding.snomed.vaccine.axes_kb import prompts


@dataclasses.dataclass(frozen=True)
class _DummyResponse:
  text: str


class VaccineAxesKBBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.initial_df = pd.DataFrame({
        "raw_snomed_code": ["1001", "1002"],
        "raw_display_name": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
        ],
    })
    self.mock_client = mock.create_autospec(
        model_client.LLMClient, instance=True, spec_set=True
    )
    self.default_column_names = builder.VaccinesKBColumnNames(
        snomed_code_col_name="snomed_code",
        preferred_term_col_name="preferred_display_term",
        parent_snomed_code_col_name="parent_snomed_code",
        parent_preferred_display_term_col_name="parent_preferred_display_term",
        patient_type_col_name="patient_type",
        clinical_variations_col_name="clinical_variations",
        antigens_col_name="antigens",
    )

  def _create_builder(
      self,
      df: pd.DataFrame | None = None,
      client: model_client.LLMClient | None = None,
      column_names: builder.VaccinesKBColumnNames | None = None,
      **kwargs,
  ) -> builder.VaccinesKBBuilder:
    target_df = self.initial_df.copy() if df is None else df
    target_client = self.mock_client if client is None else client
    target_column_names = (
        self.default_column_names if column_names is None else column_names
    )
    kwargs.setdefault("n_snomed_codes", len(target_df))
    return builder.VaccinesKBBuilder(
        df=target_df,
        client=target_client,
        input_snomed_code_col_name="raw_snomed_code",
        input_preferred_term_col_name="raw_display_name",
        column_names=target_column_names,
        **kwargs,
    )

  # ---------------------------------------------------------------------------
  # Response Extraction & Parsing Tests
  # ---------------------------------------------------------------------------

  def test_extract_csv_from_response_plain(self):
    raw_csv = "col1,col2\nval1,val2"
    extracted = builder.extract_csv_from_response(raw_csv)
    self.assertEqual(extracted, "col1,col2\nval1,val2")

  def test_extract_csv_from_response_code_block(self):
    wrapped = (
        "Here is the result:\n```csv\ncol1,col2\nval1,val2\n```\nHope it helps!"
    )
    extracted = builder.extract_csv_from_response(wrapped)
    self.assertEqual(extracted, "col1,col2\nval1,val2")

  def test_extract_csv_from_response_multiple_code_blocks(self):
    wrapped = (
        "Short preview:\n```csv\nsmall\n```\nFull"
        " table:\n```csv\ncol1,col2\nval1,val2\n```"
    )
    extracted = builder.extract_csv_from_response(wrapped)
    self.assertEqual(extracted, "col1,col2\nval1,val2")

  def test_extract_csv_from_response_with_thoughts(self):
    wrapped = (
        "<|channel>thought\nThinking about data...\n<channel|>\n"
        "```csv\ncol1,col2\nval1,val2\n```"
    )
    extracted = builder.extract_csv_from_response(wrapped)
    self.assertEqual(extracted, "col1,col2\nval1,val2")

  def test_extract_csv_from_response_empty_raises(self):
    with self.assertRaises(ValueError):
      builder.extract_csv_from_response("   ")

  def test_parse_csv_to_dataframe(self):
    csv_text = "snomed_code,preferred_display_term\n1001,BCG"
    df = builder.parse_csv_to_dataframe(csv_text)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertEqual(
        list(df.columns), ["snomed_code", "preferred_display_term"]
    )
    self.assertEqual(df.iloc[0]["snomed_code"], 1001)
    self.assertEqual(df.iloc[0]["preferred_display_term"], "BCG")

  def test_parse_llm_response_to_dataframe_json(self):
    json_response = (
        '[{"snomed_code": "1001", "preferred_display_term": "BCG Vaccine"}]'
    )
    df = builder.parse_llm_response_to_dataframe(json_response)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]["snomed_code"], "1001")
    self.assertEqual(df.iloc[0]["preferred_display_term"], "BCG Vaccine")

  def test_parse_llm_response_to_dataframe_csv_code_block(self):
    csv_response = (
        "```csv\nsnomed_code,preferred_display_term\n1001,BCG Vaccine\n```"
    )
    df = builder.parse_llm_response_to_dataframe(csv_response)
    self.assertIsInstance(df, pd.DataFrame)
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]["snomed_code"], 1001)

  def test_parse_llm_response_to_dataframe_invalid_json_falls_back_to_csv(self):
    response = "[invalid json\n```csv\nsnomed_code,term\n1001,BCG\n```"
    df = builder.parse_llm_response_to_dataframe(response)
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]["term"], "BCG")

  # ---------------------------------------------------------------------------
  # Builder Initialization Tests
  # ---------------------------------------------------------------------------

  def test_init_success(self):
    kb_builder = self._create_builder()
    self.assertIn("snomed_code", kb_builder.df.columns)
    self.assertIn("preferred_display_term", kb_builder.df.columns)
    self.assertNotIn("raw_snomed_code", kb_builder.df.columns)
    self.assertNotIn("raw_display_name", kb_builder.df.columns)
    self.assertEqual(kb_builder.snomed_code_col_name, "snomed_code")
    self.assertEqual(
        kb_builder.preferred_term_col_name, "preferred_display_term"
    )
    self.assertEqual(
        kb_builder.parent_snomed_code_col_name, "parent_snomed_code"
    )
    self.assertEqual(
        kb_builder.parent_preferred_display_term_col_name,
        "parent_preferred_display_term",
    )
    self.assertEqual(kb_builder.patient_type_col_name, "patient_type")
    self.assertEqual(
        kb_builder.clinical_variations_col_name, "clinical_variations"
    )
    self.assertEqual(kb_builder.antigens_col_name, "antigens")
    self.assertEqual(kb_builder._n_expected_snomed_codes, 2)
    self.assertEqual(kb_builder.temperature, 0.0)
    self.assertEqual(kb_builder.max_attempts, 3)

  def test_init_custom_column_names(self):
    custom_df = pd.DataFrame({"c_in": ["101"], "d_in": ["Test"]})

    column_names = builder.VaccinesKBColumnNames(
        snomed_code_col_name="c_out",
        preferred_term_col_name="d_out",
        parent_snomed_code_col_name="p_code",
        parent_preferred_display_term_col_name="p_display",
        patient_type_col_name="p_type",
        clinical_variations_col_name="c_vars",
        antigens_col_name="c_antigens",
    )
    kb_builder = builder.VaccinesKBBuilder(
        df=custom_df,
        client=self.mock_client,
        input_snomed_code_col_name="c_in",
        input_preferred_term_col_name="d_in",
        column_names=column_names,
        temperature=0.5,
        max_attempts=5,
    )
    self.assertIn("c_out", kb_builder.df.columns)
    self.assertIn("d_out", kb_builder.df.columns)
    self.assertEqual(kb_builder.snomed_code_col_name, "c_out")
    self.assertEqual(kb_builder.preferred_term_col_name, "d_out")
    self.assertEqual(kb_builder.parent_snomed_code_col_name, "p_code")
    self.assertEqual(
        kb_builder.parent_preferred_display_term_col_name, "p_display"
    )
    self.assertEqual(kb_builder.patient_type_col_name, "p_type")
    self.assertEqual(kb_builder.clinical_variations_col_name, "c_vars")
    self.assertEqual(kb_builder.antigens_col_name, "c_antigens")
    self.assertEqual(
        kb_builder._n_expected_snomed_codes,
        builder._N_SNOMED_NDHM_VACCINE_CODES,
    )
    self.assertEqual(kb_builder.temperature, 0.5)
    self.assertEqual(kb_builder.max_attempts, 5)

  def test_init_none_client_raises(self):
    with self.assertRaisesRegex(ValueError, "LLM `client` cannot be None."):
      builder.VaccinesKBBuilder(
          df=self.initial_df,
          client=None,
          input_snomed_code_col_name="raw_snomed_code",
          input_preferred_term_col_name="raw_display_name",
          column_names=self.default_column_names,
      )

  def test_init_none_df_raises(self):
    with self.assertRaisesRegex(ValueError, "Input `df` cannot be None."):
      builder.VaccinesKBBuilder(
          df=None,
          client=self.mock_client,
          input_snomed_code_col_name="raw_snomed_code",
          input_preferred_term_col_name="raw_display_name",
          column_names=self.default_column_names,
      )

  # ---------------------------------------------------------------------------
  # Prompt Formatting Tests
  # ---------------------------------------------------------------------------

  def test_format_prompt_replaces_all_placeholders(self):
    kb_builder = self._create_builder()
    template = (
        "Code: {snomed_code_col_name}\n"
        "Parent: {parent_snomed_code_col_name}\n"
        "Term: {preferred_term_col_name}\n"
        "Variations: {clinical_variations_col_name}\n"
        "Antigens: {antigens_col_name}\n"
        "Expected: {expected_snomed_codes_count}\n"
        "Data:\n{input_csv}"
    )
    formatted = kb_builder._format_prompt(template, kb_builder.df)
    self.assertIn("Code: snomed_code", formatted)
    self.assertIn("Parent: parent_snomed_code", formatted)
    self.assertIn("Term: preferred_display_term", formatted)
    self.assertIn("Variations: clinical_variations", formatted)
    self.assertIn("Antigens: antigens", formatted)
    self.assertIn("Expected: 2", formatted)
    self.assertIn("Adult Hepatitis B vaccine", formatted)
    self.assertIn("Pediatric Rotavirus vaccine", formatted)
    self.assertNotIn("{input_csv}", formatted)
    self.assertNotIn("{expected_snomed_codes_count}", formatted)

  def test_prompts_configured(self):
    self.assertIn(
        "{input_csv}", prompts.PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY
    )
    self.assertIn(
        "{parent_snomed_code_col_name}",
        prompts.PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY,
    )
    self.assertIn(
        "{expected_snomed_codes_count}",
        prompts.PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY,
    )
    self.assertIn(
        "{input_csv}", prompts.PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS
    )
    self.assertIn(
        "{clinical_variations_col_name}",
        prompts.PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS,
    )
    self.assertIn(
        "{snomed_code_col_name}",
        prompts.PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS,
    )
    self.assertIn(
        "{expected_snomed_codes_count}",
        prompts.PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS,
    )
    self.assertIn("{input_csv}", prompts.PROMPT_SNOMED_VACCINES_ANTIGENS)
    self.assertIn(
        "{antigens_col_name}", prompts.PROMPT_SNOMED_VACCINES_ANTIGENS
    )
    self.assertIn(
        "{expected_snomed_codes_count}",
        prompts.PROMPT_SNOMED_VACCINES_ANTIGENS,
    )

  # ---------------------------------------------------------------------------
  # LLM Generation Test
  # ---------------------------------------------------------------------------

  def test_generate_llm_response(self):
    kb_builder = self._create_builder(temperature=0.0)
    mock_response = _DummyResponse(text="Model output text")
    self.mock_client.generate_content.return_value = mock_response

    output = kb_builder._generate_llm_response("Sample prompt")

    self.assertEqual(output, "Model output text")
    self.mock_client.generate_content.assert_called_once_with(
        contents=["Sample prompt"],
        config={"temperature": 0.0},
    )

  # ---------------------------------------------------------------------------
  # Patient Type Determination Tests
  # ---------------------------------------------------------------------------

  @parameterized.named_parameters(
      (
          "adult_standard",
          "Adult Hepatitis B vaccine",
          resources.PatientType.ADULT,
      ),
      (
          "adult_lowercase",
          "tetanus toxoid for adult use",
          resources.PatientType.ADULT,
      ),
      (
          "paediatric_uk_spelling",
          "Paediatric Rotavirus vaccine",
          resources.PatientType.PAEDIATRIC,
      ),
      (
          "pediatric_us_spelling",
          "Pediatric Rotavirus vaccine",
          resources.PatientType.PAEDIATRIC,
      ),
      (
          "unknown_general_vaccine",
          "BCG Vaccine",
          resources.PatientType.UNKNOWN,
      ),
      (
          "unknown_brand_name",
          "Covaxin",
          resources.PatientType.UNKNOWN,
      ),
  )
  def test_determine_patient_type(self, preferred_name, expected_patient_type):
    kb_builder = self._create_builder()
    patient_type = kb_builder._determine_patient_type(preferred_name)
    self.assertEqual(patient_type, expected_patient_type)

  # ---------------------------------------------------------------------------
  # Preferred Display Name Lookup Tests
  # ---------------------------------------------------------------------------

  def test_fetch_preferred_display_name_single_code(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001", "1002"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
        ],
    })
    result = kb_builder._fetch_preferred_display_name_for_snomed_codes(
        lookup_df, "1001"
    )
    self.assertEqual(result, "Adult Hepatitis B vaccine")

  def test_fetch_preferred_display_name_multiple_codes(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001", "1002", "1003"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
            "BCG Vaccine",
        ],
    })
    result = kb_builder._fetch_preferred_display_name_for_snomed_codes(
        lookup_df, "1001, 1002"
    )
    self.assertEqual(
        result, "Adult Hepatitis B vaccine|Pediatric Rotavirus vaccine"
    )

  def test_fetch_preferred_display_name_deduplicates(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001"],
        "preferred_display_term": ["Adult Hepatitis B vaccine"],
    })
    result = kb_builder._fetch_preferred_display_name_for_snomed_codes(
        lookup_df, "1001, 1001"
    )
    self.assertEqual(result, "Adult Hepatitis B vaccine")

  def test_fetch_preferred_display_name_na_or_not_found(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001"],
        "preferred_display_term": ["Adult Hepatitis B vaccine"],
    })
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, "N/A"
        ),
        "N/A",
    )
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, "9999"
        ),
        "N/A",
    )

  def test_fetch_preferred_display_name_numeric_and_float_codes(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001", "1002"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
        ],
    })
    # Integer code
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, 1001
        ),
        "Adult Hepatitis B vaccine",
    )
    # Numpy int64
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, np.int64(1001)
        ),
        "Adult Hepatitis B vaccine",
    )
    # Float code (e.g. from pandas inferring float for integer columns
    # with NaNs)
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, 1001.0
        ),
        "Adult Hepatitis B vaccine",
    )
    # Float code with fractional part
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, 1001.5
        ),
        "N/A",
    )
    # String representation with trailing .0
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, "1001.0, 1002.0"
        ),
        "Adult Hepatitis B vaccine|Pediatric Rotavirus vaccine",
    )

  def test_fetch_preferred_display_name_missing_values(self):
    kb_builder = self._create_builder()
    lookup_df = pd.DataFrame({
        "snomed_code": ["1001"],
        "preferred_display_term": ["Adult Hepatitis B vaccine"],
    })
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, np.nan
        ),
        "N/A",
    )
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, None
        ),
        "N/A",
    )
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, pd.NA
        ),
        "N/A",
    )
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, ""
        ),
        "N/A",
    )
    self.assertEqual(
        kb_builder._fetch_preferred_display_name_for_snomed_codes(
            lookup_df, "   "
        ),
        "N/A",
    )

  # ---------------------------------------------------------------------------
  # Post-Processing Tests
  # ---------------------------------------------------------------------------

  def test_post_process(self):
    kb_builder = self._create_builder()
    stage3_df = pd.DataFrame({
        "snomed_code": ["1001", "1002", "1003"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
            "BCG Vaccine",
        ],
        "parent_snomed_code": ["1002", "N/A", "1001, 1002"],
        "clinical_variations": ["HepB", "Rota", "BCG"],
        "antigens": ["Hepatitis B", "Rotavirus", "Mycobacterium bovis"],
    })

    processed_df = kb_builder._post_process(stage3_df)

    self.assertEqual(
        processed_df.iloc[0]["parent_preferred_display_term"],
        "Pediatric Rotavirus vaccine",
    )
    self.assertEqual(
        processed_df.iloc[0]["patient_type"], resources.PatientType.ADULT
    )

    self.assertEqual(
        processed_df.iloc[1]["parent_preferred_display_term"], "N/A"
    )
    self.assertEqual(
        processed_df.iloc[1]["patient_type"], resources.PatientType.PAEDIATRIC
    )

    self.assertEqual(
        processed_df.iloc[2]["parent_preferred_display_term"],
        "Adult Hepatitis B vaccine|Pediatric Rotavirus vaccine",
    )
    self.assertEqual(
        processed_df.iloc[2]["patient_type"], resources.PatientType.UNKNOWN
    )

  def test_post_process_with_numeric_and_nan_parent_snomed_codes(self):
    kb_builder = self._create_builder()
    stage3_df = pd.DataFrame({
        "snomed_code": ["1001", "1002", "1003"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
            "BCG Vaccine",
        ],
        "parent_snomed_code": [1002, np.nan, 1001.0],
        "clinical_variations": ["HepB", "Rota", "BCG"],
        "antigens": ["Hepatitis B", "Rotavirus", "Mycobacterium bovis"],
    })

    processed_df = kb_builder._post_process(stage3_df)

    self.assertEqual(
        processed_df.iloc[0]["parent_preferred_display_term"],
        "Pediatric Rotavirus vaccine",
    )
    self.assertEqual(
        processed_df.iloc[1]["parent_preferred_display_term"], "N/A"
    )
    self.assertEqual(
        processed_df.iloc[2]["parent_preferred_display_term"],
        "Adult Hepatitis B vaccine",
    )

  # ---------------------------------------------------------------------------
  # End-to-End build_kb Pipeline Tests
  # ---------------------------------------------------------------------------

  def test_build_kb_3_sequential_stages(self):
    kb_builder = self._create_builder(temperature=0.0)

    stage1_csv = (
        "```csv\n"
        "snomed_code,preferred_display_term,parent_snomed_code\n"
        '1001,Adult Hepatitis B vaccine,"1002"\n'
        '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
        "```"
    )
    stage2_csv = (
        "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
        ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
        ' vaccine,"1001, 1002",Rota\n```'
    )
    stage3_csv = (
        "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
        ' Hepatitis B vaccine,"1002",HepB,Hepatitis B surface'
        ' antigen\n1002,Pediatric Rotavirus vaccine,"1001, 1002",Rota,Rotavirus'
        " antigen\n```"
    )
    self.mock_client.generate_content.side_effect = [
        _DummyResponse(text=stage1_csv),
        _DummyResponse(text=stage2_csv),
        _DummyResponse(text=stage3_csv),
    ]

    final_df = kb_builder.build_kb()

    self.assertEqual(self.mock_client.generate_content.call_count, 3)
    self.assertLen(final_df, 2)
    self.assertIn("parent_preferred_display_term", final_df.columns)
    self.assertIn("patient_type", final_df.columns)
    self.assertEqual(
        final_df.iloc[0]["parent_preferred_display_term"],
        "Pediatric Rotavirus vaccine",
    )
    self.assertEqual(
        final_df.iloc[0]["patient_type"], resources.PatientType.ADULT
    )
    self.assertEqual(
        final_df.iloc[0]["antigens"], "Hepatitis B surface antigen"
    )
    self.assertEqual(
        final_df.iloc[1]["parent_preferred_display_term"],
        "Adult Hepatitis B vaccine|Pediatric Rotavirus vaccine",
    )
    self.assertEqual(
        final_df.iloc[1]["patient_type"], resources.PatientType.PAEDIATRIC
    )

    # Verify sequential chaining of prompts
    calls = self.mock_client.generate_content.call_args_list
    stage1_prompt = calls[0].kwargs["contents"][0]
    stage2_prompt = calls[1].kwargs["contents"][0]
    stage3_prompt = calls[2].kwargs["contents"][0]

    # Stage 1 received base DataFrame
    self.assertIn("Adult Hepatitis B vaccine", stage1_prompt)
    # Stage 2 received Stage 1 output (contains parent_snomed_code)
    self.assertIn("parent_snomed_code", stage2_prompt)
    # Stage 3 received Stage 2 output (contains clinical_variations)
    self.assertIn("clinical_variations", stage3_prompt)

  def test_build_kb_save_to_tsv(self):
    temp_dir = self.create_tempdir()
    output_tsv = os.path.join(temp_dir.full_path, "vaccines_kb.tsv")
    kb_builder = self._create_builder()

    stage_responses = [
        _DummyResponse(
            text=(
                "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
                ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric'
                ' Rotavirus vaccine,"1001, 1002",Rota,Rotavirus\n```'
            )
        )
        for _ in range(3)
    ]
    self.mock_client.generate_content.side_effect = stage_responses

    final_df = kb_builder.build_kb(save_tsv_path=output_tsv)

    self.assertTrue(os.path.exists(output_tsv))
    saved_df = pd.read_csv(output_tsv, sep="\t")
    pd.testing.assert_frame_equal(final_df, saved_df)

  def test_build_kb_retry_on_failure_then_succeeds(self):
    kb_builder = self._create_builder(max_attempts=2)
    bad_response = _DummyResponse(text="Error: Could not generate CSV")
    good_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    good_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
            ' vaccine,"1001, 1002",Rota\n```'
        )
    )
    good_stage3 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric'
            ' Rotavirus vaccine,"1001, 1002",Rota,Rotavirus\n```'
        )
    )
    # Stage 1 fails on first attempt, succeeds on second attempt
    self.mock_client.generate_content.side_effect = [
        bad_response,
        good_stage1,
        good_stage2,
        good_stage3,
    ]

    final_df = kb_builder.build_kb()

    self.assertLen(final_df, 2)
    self.assertEqual(self.mock_client.generate_content.call_count, 4)

  def test_build_kb_retry_on_row_count_mismatch(self):
    kb_builder = self._create_builder(max_attempts=2)
    # Mismatched row count (1 row instead of 2)
    mismatched_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            "```"
        )
    )
    matched_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    good_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
            ' vaccine,"1001, 1002",Rota\n```'
        )
    )
    good_stage3 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric'
            ' Rotavirus vaccine,"1001, 1002",Rota,Rotavirus\n```'
        )
    )
    self.mock_client.generate_content.side_effect = [
        mismatched_stage1,
        matched_stage1,
        good_stage2,
        good_stage3,
    ]

    final_df = kb_builder.build_kb()

    self.assertLen(final_df, 2)
    self.assertEqual(self.mock_client.generate_content.call_count, 4)

  def test_build_kb_retry_on_stage2_row_count_mismatch(self):
    kb_builder = self._create_builder(max_attempts=2)
    good_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    mismatched_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n```'
        )
    )
    good_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
            ' vaccine,"1001, 1002",Rota\n```'
        )
    )
    good_stage3 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric'
            ' Rotavirus vaccine,"1001, 1002",Rota,Rotavirus\n```'
        )
    )
    self.mock_client.generate_content.side_effect = [
        good_stage1,
        mismatched_stage2,
        good_stage2,
        good_stage3,
    ]

    final_df = kb_builder.build_kb()

    self.assertLen(final_df, 2)
    self.assertEqual(self.mock_client.generate_content.call_count, 4)

  def test_build_kb_retry_on_stage3_row_count_mismatch(self):
    kb_builder = self._create_builder(max_attempts=2)
    good_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    good_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
            ' vaccine,"1001, 1002",Rota\n```'
        )
    )
    mismatched_stage3 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n```'
        )
    )
    good_stage3 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric'
            ' Rotavirus vaccine,"1001, 1002",Rota,Rotavirus\n```'
        )
    )
    self.mock_client.generate_content.side_effect = [
        good_stage1,
        good_stage2,
        mismatched_stage3,
        good_stage3,
    ]

    final_df = kb_builder.build_kb()

    self.assertLen(final_df, 2)
    self.assertEqual(self.mock_client.generate_content.call_count, 4)

  def test_build_kb_stage1_failure_raises_value_error(self):
    kb_builder = self._create_builder(max_attempts=2)
    self.mock_client.generate_content.side_effect = RuntimeError("API error")

    with self.assertRaisesRegex(
        ValueError,
        "Failed to generate SNOMED parent-child relationships after"
        " 2 attempts.",
    ):
      kb_builder.build_kb()
    self.assertEqual(self.mock_client.generate_content.call_count, 2)

  def test_build_kb_stage2_failure_raises_value_error(self):
    kb_builder = self._create_builder(max_attempts=2)
    good_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    self.mock_client.generate_content.side_effect = [
        good_stage1,
        RuntimeError("Stage 2 API error"),
        RuntimeError("Stage 2 API error retry"),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "Failed to generate SNOMED clinical variations after 2 attempts.",
    ):
      kb_builder.build_kb()
    self.assertEqual(self.mock_client.generate_content.call_count, 3)

  def test_build_kb_stage3_failure_raises_value_error(self):
    kb_builder = self._create_builder(max_attempts=2)
    good_stage1 = _DummyResponse(
        text=(
            "```csv\n"
            "snomed_code,preferred_display_term,parent_snomed_code\n"
            '1001,Adult Hepatitis B vaccine,"1002"\n'
            '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
            "```"
        )
    )
    good_stage2 = _DummyResponse(
        text=(
            "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
            ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
            ' vaccine,"1001, 1002",Rota\n```'
        )
    )
    self.mock_client.generate_content.side_effect = [
        good_stage1,
        good_stage2,
        RuntimeError("Stage 3 API error"),
        RuntimeError("Stage 3 API error retry"),
    ]

    with self.assertRaisesRegex(
        ValueError,
        "Failed to generate SNOMED vaccine antigens after 2 attempts.",
    ):
      kb_builder.build_kb()
    self.assertEqual(self.mock_client.generate_content.call_count, 4)

  def test_infer_parent_child_relationships(self):
    kb_builder = self._create_builder()
    stage1_csv = (
        "```csv\n"
        "snomed_code,preferred_display_term,parent_snomed_code\n"
        '1001,Adult Hepatitis B vaccine,"1002"\n'
        '1002,Pediatric Rotavirus vaccine,"1001, 1002"\n'
        "```"
    )
    self.mock_client.generate_content.return_value = _DummyResponse(
        text=stage1_csv
    )
    result_df = kb_builder.infer_parent_child_relationships()
    self.assertLen(result_df, 2)
    self.assertIn("parent_snomed_code", result_df.columns)
    self.mock_client.generate_content.assert_called_once()

  def test_generate_clinical_variations(self):
    kb_builder = self._create_builder()
    input_df = pd.DataFrame({
        "snomed_code": ["1001", "1002"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
        ],
        "parent_snomed_code": ["1002", "1001, 1002"],
    })
    stage2_csv = (
        "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations\n1001,Adult"
        ' Hepatitis B vaccine,"1002",HepB\n1002,Pediatric Rotavirus'
        ' vaccine,"1001, 1002",Rota\n```'
    )
    self.mock_client.generate_content.return_value = _DummyResponse(
        text=stage2_csv
    )
    result_df = kb_builder.generate_clinical_variations(input_df)
    self.assertLen(result_df, 2)
    self.assertIn("clinical_variations", result_df.columns)
    self.mock_client.generate_content.assert_called_once()

  def test_generate_antigens(self):
    kb_builder = self._create_builder()
    input_df = pd.DataFrame({
        "snomed_code": ["1001", "1002"],
        "preferred_display_term": [
            "Adult Hepatitis B vaccine",
            "Pediatric Rotavirus vaccine",
        ],
        "parent_snomed_code": ["1002", "1001, 1002"],
        "clinical_variations": ["HepB", "Rota"],
    })
    stage3_csv = (
        "```csv\nsnomed_code,preferred_display_term,parent_snomed_code,clinical_variations,antigens\n1001,Adult"
        ' Hepatitis B vaccine,"1002",HepB,Hepatitis B\n1002,Pediatric Rotavirus'
        ' vaccine,"1001, 1002",Rota,Rotavirus\n```'
    )
    self.mock_client.generate_content.return_value = _DummyResponse(
        text=stage3_csv
    )
    result_df = kb_builder.generate_antigens(input_df)
    self.assertLen(result_df, 2)
    self.assertIn("antigens", result_df.columns)
    self.mock_client.generate_content.assert_called_once()


if __name__ == "__main__":
  absltest.main()

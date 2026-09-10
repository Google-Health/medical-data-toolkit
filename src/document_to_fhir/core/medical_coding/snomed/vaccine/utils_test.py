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
"""Unit tests for vaccine utils."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.vaccine import utils


class RemoveVaccineKeywordsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("vaccine_singular", "COVID-19 vaccine", "COVID-19"),
      ("vaccine_plural", "Polio vaccines", "Polio"),
      ("vax_singular", "Flu vax", "Flu"),
      ("vax_plural", "Flu vaxes", "Flu"),
      ("antigen_singular", "Rabies antigen", "Rabies"),
      ("antigen_plural", "Hepatitis B antigens", "Hepatitis B"),
      ("virus_keyword", "Measles virus", "Measles"),
      ("bacteria_keyword", "Tetanus bacteria", "Tetanus"),
      ("paediatric_keyword", "paediatric Hepatitis B", "Hepatitis B"),
      ("pediatric_keyword", "pediatric Polio", "Polio"),
      ("adult_keyword", "Tetanus adult", "Tetanus"),
      ("ped_keyword", "ped DTP", "DTP"),
  )
  def test_removes_single_keywords(self, input_text, expected):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)

  @parameterized.named_parameters(
      ("all_caps", "COVID-19 VACCINE", "COVID-19"),
      ("mixed_case", "PEDIATRIC hepatitis b Vax", "hepatitis b"),
      ("title_case", "Adult Vaccines", ""),
  )
  def test_case_insensitivity(self, input_text, expected):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)

  @parameterized.named_parameters(
      (
          "multiple_keywords",
          "pediatric adult hepatitis B vaccine",
          "hepatitis B",
      ),
      ("virus_and_antigen", "virus antigen combination", "combination"),
      (
          "keywords_with_commas",
          "measles vaccine , mumps vaccine , rubella vaccine",
          "measles, mumps, rubella",
      ),
  )
  def test_removes_multiple_keywords(self, input_text, expected):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)

  @parameterized.named_parameters(
      ("pedvaxhib_brand", "pedvaxhib", "pedvaxhib"),
      ("antigenic_word", "antigenic variation", "antigenic variation"),
      ("bacterial_word", "bacterial infection", "bacterial infection"),
      ("viral_word", "viral strain", "viral strain"),
      ("rotavirus_word", "rotavirus", "rotavirus"),
      ("pediatrics_word", "pediatrics clinic", "pediatrics clinic"),
      ("paediatrics_word", "paediatrics care", "paediatrics care"),
      ("adults_plural", "adults only", "adults only"),
  )
  def test_preserves_words_containing_keyword_substrings(
      self, input_text, expected
  ):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)

  @parameterized.named_parameters(
      ("multiple_spaces", "Measles   vaccine   booster", "Measles booster"),
      ("spacing_around_comma", "Measles vaccine ,Rubella", "Measles, Rubella"),
      ("leading_comma", ", Hepatitis B vaccine", "Hepatitis B"),
      ("trailing_comma", "Hepatitis B vaccine,", "Hepatitis B"),
      ("both_commas", "vaccine, Polio vaccine, vaccine", "Polio"),
  )
  def test_whitespace_and_comma_cleanup(self, input_text, expected):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)

  @parameterized.named_parameters(
      ("empty_string", "", ""),
      ("whitespace_only", "   ", ""),
      ("keyword_only", "vaccine", ""),
      ("multiple_keywords_only", "pediatric adult vaccine", ""),
      ("no_keywords", "Covaxin", "Covaxin"),
  )
  def test_edge_cases(self, input_text, expected):
    self.assertEqual(utils.remove_vaccine_keywords(input_text), expected)


class ParseAndNormalizeVaccinesTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("simple_name", "Covaxin", ["covaxin"]),
      ("name_with_keyword", "Covid-19 vaccine", ["covid19"]),
      ("hyphenated_name", "DTP-HepB", ["dtphepb"]),
      ("multiple_hyphens", "DTP-HepB-Hib", ["dtphepbhib"]),
  )
  def test_single_vaccine_normalization(self, input_text, expected):
    self.assertEqual(utils.parse_and_normalize_vaccines(input_text), expected)

  @parameterized.named_parameters(
      (
          "comma_separated",
          "Measles, Mumps, Rubella",
          ["measles", "mumps", "rubella"],
      ),
      (
          "comma_separated_with_keywords",
          "DTP vaccine, HepB vaccine, Polio vaccine",
          ["dtp", "hepb", "polio"],
      ),
      (
          "whitespace_around_commas",
          "Measles  ,  Mumps ,  Rubella",
          ["measles", "mumps", "rubella"],
      ),
  )
  def test_comma_separated_vaccines(self, input_text, expected):
    self.assertEqual(utils.parse_and_normalize_vaccines(input_text), expected)

  @parameterized.named_parameters(
      (
          "parentheses_subterms",
          "Measles (Rubella, Mumps)",
          ["measles", "rubella", "mumps"],
      ),
      (
          "parentheses_single_term",
          "Polio (Oral) vaccine",
          ["polio", "oral"],
      ),
      (
          "parentheses_with_hyphen",
          "Combined (DTP-HepB)",
          ["combined", "dtphepb"],
      ),
  )
  def test_parentheses_expansion(self, input_text, expected):
    self.assertEqual(utils.parse_and_normalize_vaccines(input_text), expected)

  @parameterized.named_parameters(
      ("empty_string", "", [""]),
      ("keyword_only", "vaccine", [""]),
      ("multiple_keywords_only", "pediatric adult vaccine", [""]),
  )
  def test_edge_cases(self, input_text, expected):
    self.assertEqual(utils.parse_and_normalize_vaccines(input_text), expected)


class GetTopParentCodeTest(parameterized.TestCase):

  def test_single_ancestor_chain_traversal(self):
    df = pd.DataFrame({
        "snomed_code": ["100", "200", "300"],
        "parent_snomed_code": ["200", "300", None],
    })
    result = utils.get_top_parent_code(df, excluded_codes=[])
    self.assertEqual(result, "300")

  def test_stops_before_excluded_parent(self):
    df = pd.DataFrame({
        "snomed_code": ["100", "200", "300"],
        "parent_snomed_code": ["200", "300", "400"],
    })
    result = utils.get_top_parent_code(df, excluded_codes=["300"])
    self.assertEqual(result, "200")

  def test_stops_when_immediate_parent_is_excluded(self):
    df = pd.DataFrame({
        "snomed_code": ["100", "200"],
        "parent_snomed_code": ["200", "300"],
    })
    result = utils.get_top_parent_code(df, excluded_codes=["200"])
    self.assertEqual(result, "100")

  def test_picks_first_non_excluded_concept_as_start(self):
    df = pd.DataFrame({
        "snomed_code": ["100", "200", "300"],
        "parent_snomed_code": ["None", "300", None],
    })
    # '100' is excluded, so traversal starts at '200' and reaches '300'.
    result = utils.get_top_parent_code(df, excluded_codes=["100"])
    self.assertEqual(result, "300")

  def test_returns_none_when_all_concepts_are_excluded(self):
    df = pd.DataFrame({
        "snomed_code": ["100", "200"],
        "parent_snomed_code": ["200", None],
    })
    result = utils.get_top_parent_code(df, excluded_codes=["100", "200"])
    self.assertIsNone(result)

  def test_returns_none_for_empty_dataframe(self):
    df = pd.DataFrame({
        "snomed_code": [],
        "parent_snomed_code": [],
    })
    result = utils.get_top_parent_code(df, excluded_codes=[])
    self.assertIsNone(result)

  @parameterized.named_parameters(
      ("none_value", None),
      ("nan_string", "nan"),
      ("none_string", "None"),
      ("empty_string", ""),
  )
  def test_stops_at_various_missing_parent_representations(
      self, missing_parent
  ):
    df = pd.DataFrame({
        "snomed_code": ["100"],
        "parent_snomed_code": [missing_parent],
    })
    result = utils.get_top_parent_code(df, excluded_codes=[])
    self.assertEqual(result, "100")

  def test_parent_not_in_dataframe_concept_column(self):
    # '200' is the parent of '100', but '200' is not in the concept column.
    df = pd.DataFrame({
        "snomed_code": ["100"],
        "parent_snomed_code": ["200"],
    })
    result = utils.get_top_parent_code(df, excluded_codes=[])
    self.assertEqual(result, "200")

  def test_handles_cycle_gracefully(self):
    # Cycle: 100 -> 200 -> 100
    df = pd.DataFrame({
        "snomed_code": ["100", "200"],
        "parent_snomed_code": ["200", "100"],
    })
    result = utils.get_top_parent_code(df, excluded_codes=[])
    # Traversal visits 100 -> 200 -> sees 100 already in visited, breaks.
    self.assertEqual(result, "100")

  def test_custom_column_names(self):
    df = pd.DataFrame({
        "my_concept": ["A", "B", "C"],
        "my_parent": ["B", "C", None],
    })
    result = utils.get_top_parent_code(
        df,
        excluded_codes=[],
        concept_col="my_concept",
        parent_col="my_parent",
    )
    self.assertEqual(result, "C")

  def test_numeric_string_identifiers_with_integer_exclusions(self):
    df = pd.DataFrame({
        "snomed_code": ["10001", "10002", "10003"],
        "parent_snomed_code": ["10002", "10003", None],
    })
    # Passing an int in excluded_codes should match string concept codes.
    result = utils.get_top_parent_code(df, excluded_codes=[10003])
    self.assertEqual(result, "10002")


if __name__ == "__main__":
  absltest.main()

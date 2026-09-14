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
import os

from absl.testing import absltest

from src.document_to_fhir.core.medical_coding.loinc.common import utils


class FileUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()

  def test_open_file_and_file_exists(self):
    test_path = os.path.join(self.temp_dir.full_path, "sample.txt")
    self.assertFalse(utils.file_exists(test_path))

    with utils.open_file(test_path, "w") as f:
      f.write("hello world\n")

    self.assertTrue(utils.file_exists(test_path))
    with utils.open_file(test_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "hello world\n")

  def test_read_text_file_and_read_prompt(self):
    prompt_path = os.path.join(self.temp_dir.full_path, "prompt.txt")
    with utils.open_file(prompt_path, "w") as f:
      f.write("  prompt instructions \n")

    self.assertEqual(utils.read_text_file(prompt_path), "prompt instructions")
    self.assertEqual(utils.read_prompt(prompt_path), "prompt instructions")

  def test_read_csv(self):
    csv_path = os.path.join(self.temp_dir.full_path, "data.csv")
    with utils.open_file(csv_path, "w") as f:
      f.write("colA,colB\nval1,val2\n")

    df = utils.read_csv(csv_path)
    self.assertEqual(list(df.columns), ["colA", "colB"])
    self.assertLen(df, 1)
    self.assertEqual(df.iloc[0]["colA"], "val1")


class StringUtilsTest(absltest.TestCase):

  def test_extract_json_list_valid(self):
    text = 'Here is the result: ["prop1", "prop2"]'
    res = utils.StringUtils.extract_json_list_from_llm_response(text)
    self.assertEqual(res, ["prop1", "prop2"])

  def test_extract_json_list_empty(self):
    res = utils.StringUtils.extract_json_list_from_llm_response("")
    self.assertEqual(res, [])

  def test_extract_json_list_no_match(self):
    text = "No list here"
    res = utils.StringUtils.extract_json_list_from_llm_response(text)
    self.assertEqual(res, [])

  def test_extract_json_list_invalid_json(self):
    text = '[ "unclosed ]'
    res = utils.StringUtils.extract_json_list_from_llm_response(text)
    self.assertEqual(res, [])

  def test_levenshtein_distance(self):
    self.assertEqual(
        utils.StringUtils.levenshtein_distance("kitten", "sitting"), 3
    )
    self.assertEqual(utils.StringUtils.levenshtein_distance("flaw", "lawn"), 2)
    self.assertEqual(utils.StringUtils.levenshtein_distance("", "test"), 4)
    self.assertEqual(utils.StringUtils.levenshtein_distance("test", ""), 4)
    self.assertEqual(utils.StringUtils.levenshtein_distance("same", "same"), 0)

  def test_normalized_levenshtein_distance(self):
    self.assertAlmostEqual(
        utils.StringUtils.normalized_levenshtein_distance("kitten", "sitting"),
        3 / 7,
    )
    self.assertAlmostEqual(
        utils.StringUtils.normalized_levenshtein_distance(
            "haemoglobin", "hemoglobin"
        ),
        1 / 11,
    )
    self.assertEqual(
        utils.StringUtils.normalized_levenshtein_distance("", ""), 0.0
    )
    self.assertEqual(
        utils.StringUtils.normalized_levenshtein_distance("", "test"), 1.0
    )
    self.assertEqual(
        utils.StringUtils.normalized_levenshtein_distance("test", ""), 1.0
    )
    self.assertEqual(
        utils.StringUtils.normalized_levenshtein_distance("same", "same"), 0.0
    )

  def test_tokenize(self):
    self.assertEqual(
        utils.StringUtils.tokenize("Bilirubin, Direct (mg/dL)!"),
        ["bilirubin", "direct", "mg", "dl"],
    )
    self.assertEqual(utils.StringUtils.tokenize(""), [])

  def test_compute_fuzzy_token_overlap(self):
    # Exact match of all query tokens in target
    overlap = utils.StringUtils.compute_fuzzy_token_overlap(
        "bilirubin direct", "bilirubin direct ser/plas"
    )
    self.assertEqual(overlap, 1.0)

    # Partial match
    overlap = utils.StringUtils.compute_fuzzy_token_overlap(
        "bilirubin direct", "bilirubin total ser/plas"
    )
    self.assertAlmostEqual(overlap, 0.5)

    # Fuzzy match: British vs US spelling (haemoglobin -> hemoglobin)
    overlap = utils.StringUtils.compute_fuzzy_token_overlap(
        "haemoglobin", "hemoglobin bld"
    )
    self.assertEqual(overlap, 1.0)

    # Short token exact matching (< 4 chars)
    overlap = utils.StringUtils.compute_fuzzy_token_overlap("rbc count", "rbc")
    self.assertAlmostEqual(overlap, 0.5)

    # Dissimilar 3-character acronyms must not fuzzy match
    overlap = utils.StringUtils.compute_fuzzy_token_overlap("mcv", "mch")
    self.assertEqual(overlap, 0.0)

    # Empty and punctuation inputs
    self.assertEqual(
        utils.StringUtils.compute_fuzzy_token_overlap("", "text"), 0.0
    )
    self.assertEqual(
        utils.StringUtils.compute_fuzzy_token_overlap("text", ""), 0.0
    )
    self.assertEqual(
        utils.StringUtils.compute_fuzzy_token_overlap("---", "text"), 0.0
    )
    self.assertEqual(
        utils.StringUtils.compute_fuzzy_token_overlap("text", "---"), 0.0
    )

  def test_get_similarity_ratio(self):
    # Full overlap returns 0.0 distance
    self.assertAlmostEqual(
        utils.StringUtils.get_similarity_ratio(
            "bilirubin direct", "bilirubin direct ser/plas"
        ),
        0.0,
    )
    # Disjoint tokens return 1.0 distance
    self.assertAlmostEqual(
        utils.StringUtils.get_similarity_ratio("glucose", "albumin"),
        1.0,
    )


if __name__ == "__main__":
  absltest.main()

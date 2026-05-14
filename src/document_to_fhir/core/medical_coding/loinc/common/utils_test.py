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
from absl.testing import absltest

from src.document_to_fhir.core.medical_coding.loinc.common import utils


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

  def test_extract_json_list_not_list(self):
    text = '{"key": "value"}'
    res = utils.StringUtils.extract_json_list_from_llm_response(text)
    self.assertEqual(res, [])


if __name__ == "__main__":
  absltest.main()

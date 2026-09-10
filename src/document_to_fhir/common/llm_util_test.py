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
"""Tests for functions in llm_util.py."""

from absl.testing import absltest
from absl.testing import parameterized
from src.document_to_fhir.common import llm_util


class LlmUtilTest(parameterized.TestCase):
  """Tests for llm_util.py functions."""

  @parameterized.named_parameters(
      (
          'simple',
          'Here is the JSON: {"key": "value"} hope it helps.',
          '{"key": "value"}',
      ),
      (
          'markdown',
          '```json\n{"key": "value"}\n```',
          '{"key": "value"}',
      ),
      (
          'markdown_no_lang',
          '```\n{"key": "value"}\n```',
          '{"key": "value"}',
      ),
      (
          'stray_brackets',
          'Some { stray } brackets. ```json\n{"key": "value"}\n```',
          '{"key": "value"}',
      ),
  )
  def test_extract_json_from_llm_response_success(self, raw_text, expected):
    """Tests successful extraction of a JSON object."""
    self.assertEqual(
        llm_util.extract_json_from_llm_response(raw_text), expected
    )

  @parameterized.named_parameters(
      ('no_brackets', 'No JSON here.'),
      ('invalid_brackets', '} "key": "value" {'),
  )
  def test_extract_json_from_llm_response_failures(self, raw_text):
    """Tests failures in extracting a JSON object."""
    with self.assertRaisesRegex(
        ValueError, 'Valid JSON brackets {} were not found'
    ):
      llm_util.extract_json_from_llm_response(raw_text)

  @parameterized.named_parameters(
      (
          'list_success',
          'Here is the list: [1, 2, 3] hope it helps.',
          '[1, 2, 3]',
      ),
      (
          'list_markdown',
          '```json\n[1, 2, 3]\n```',
          '[1, 2, 3]',
      ),
      (
          'list_markdown_no_lang',
          '```\n[1, 2, 3]\n```',
          '[1, 2, 3]',
      ),
      (
          'list_stray_brackets',
          'Some [ stray ] brackets. ```json\n[1, 2, 3]\n```',
          '[1, 2, 3]',
      ),
      (
          'object_fallback',
          'Here is the object: {"key": "value"} hope it helps.',
          '{"key": "value"}',
      ),
      (
          'list_before_object',
          'List: [1, 2] and Object: {"a": 1}',
          '[1, 2]',
      ),
      (
          'object_before_list',
          'Object: {"a": 1} and List: [1, 2]',
          '{"a": 1}',
      ),
      (
          'nested_list_in_object',
          '{"key": [1, 2, 3]}',
          '{"key": [1, 2, 3]}',
      ),
      (
          'nested_object_in_list',
          '[{"key": "value"}]',
          '[{"key": "value"}]',
      ),
  )
  def test_extract_json_list_from_llm_response_success(
      self, raw_text, expected
  ):
    """Tests successful extraction of a JSON list."""
    self.assertEqual(
        llm_util.extract_json_list_from_llm_response(raw_text), expected
    )

  @parameterized.named_parameters(
      (
          'list_missing_closing',
          'List: [1, 2',
          'Valid JSON closing bracket ] was not found',
      ),
      (
          'object_missing_closing',
          'Object: {"a": 1',
          'Valid JSON brackets {} were not found',
      ),
  )
  def test_extract_json_list_from_llm_response_failures(
      self, raw_text, expected_error
  ):
    """Tests failures in extracting a JSON list."""
    with self.assertRaisesRegex(ValueError, expected_error):
      llm_util.extract_json_list_from_llm_response(raw_text)


if __name__ == '__main__':
  absltest.main()

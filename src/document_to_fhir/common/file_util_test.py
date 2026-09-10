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

from absl import flags
from absl.testing import absltest

from src.document_to_fhir.common import file_util

flags.FLAGS.mark_as_parsed()


class FileUtilTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir()

  def test_open_file_and_file_exists(self):
    test_path = os.path.join(self.temp_dir.full_path, "sample.txt")
    self.assertFalse(file_util.file_exists(test_path))

    with file_util.open_file(test_path, "w") as f:
      f.write("hello world\n")

    self.assertTrue(file_util.file_exists(test_path))
    with file_util.open_file(test_path, "r") as f:
      content = f.read()
    self.assertEqual(content, "hello world\n")

  def test_read_text_file_and_read_prompt(self):
    prompt_path = os.path.join(self.temp_dir.full_path, "prompt.txt")
    with file_util.open_file(prompt_path, "w") as f:
      f.write("  prompt instructions \n")

    self.assertEqual(
        file_util.read_text_file(prompt_path), "prompt instructions"
    )
    self.assertEqual(file_util.read_prompt(prompt_path), "prompt instructions")

  def test_file_exists_empty_string(self):
    self.assertFalse(file_util.file_exists(""))

  def test_file_exists_none(self):
    self.assertFalse(file_util.file_exists(None))


if __name__ == "__main__":
  absltest.main()

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
"""Central file utility functions."""

from typing import Any


def file_exists(filepath: str | None) -> bool:
  """Checks if a file exists."""
  import os
  return bool(filepath and os.path.exists(filepath))


def open_file(
    filepath: str, mode: str = "r", encoding: str | None = None
) -> Any:
  """Opens a file using gfile internally or built-in open externally."""
  return open(filepath, mode, encoding=encoding)


def read_text_file(filepath: str, encoding: str = "utf-8") -> str:
  """Reads an entire text file and returns its content stripped."""
  with open(filepath, "r", encoding=encoding) as f:
    return f.read().strip()


def read_prompt(prompt_path: str) -> str:
  """Reads a prompt template from the given path."""
  return read_text_file(prompt_path)

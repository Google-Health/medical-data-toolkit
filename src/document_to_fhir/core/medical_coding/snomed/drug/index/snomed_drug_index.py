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
"""Base index for SNOMED Drug Codes."""

from typing import Any

import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug import utils


def _file_exists(csv_path: str) -> bool:
  """Returns True if the file exists, False otherwise."""
  return utils.file_exists(csv_path)


class SnomedDrugIndex:
  """Base class for SNOMED Drug Indexes."""

  def __init__(self, df: pd.DataFrame, groups: dict[str, Any]):
    self.master_df = df
    self.groups = groups

  @classmethod
  def from_index_file(cls, path: str, **kwargs) -> "SnomedDrugIndex":
    """Loads and builds a SnomedDrugIndex from an index file.

    Args:
      path: The path to the input index file (csv/txt).
      **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
      An instance of SnomedDrugIndex populated with data from the index file.

    Raises:
      FileNotFoundError: If the specified path does not exist.
    """
    if not _file_exists(path):
      raise FileNotFoundError(f"File not found: {path}")

    # Default parameters for read_csv that are common.
    read_csv_args = {
        "sep": "\t",
        "index_col": False,
        "dtype": str,
        "on_bad_lines": "skip",
    }
    read_csv_args.update(kwargs)

    df = pd.read_csv(path, **read_csv_args)
    df, groups = cls._preprocess(df)
    return cls(df, groups)

  @classmethod
  def _preprocess(
      cls, df: pd.DataFrame
  ) -> tuple[pd.DataFrame, dict[str, Any]]:
    raise NotImplementedError("Subclasses must implement _preprocess")

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
"""Index for SNOMED Vaccine Codes."""

import os
from typing import Any

import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine import utils


def _file_exists(csv_path: str) -> bool:
  """Returns True if the file exists, False otherwise."""
  return os.path.exists(csv_path)


def _parse_patient_type(
    val: Any
) -> resources.PatientType:
  """Parses a patient type value into a PatientType enum."""
  if isinstance(val, resources.PatientType):
    return val
  if pd.notna(val):
    val_str = str(val).strip().upper()
    if val_str in resources.PatientType.__members__:
      return resources.PatientType(val_str)
  return resources.PatientType.UNKNOWN


def _extract_norm_variations(row: pd.Series) -> str:
  """Extracts and normalizes clinical variations and antigens from a row."""
  variations = row.get("clinical_variations")
  antigens = row.get("antigens")

  var_str = str(variations) if pd.notna(variations) else ""
  antigens_str = (
      ", ".join(s.strip() for s in antigens.split("+"))
      if pd.notna(antigens)
      else ""
  )
  terms = utils.parse_and_normalize_vaccines(var_str)
  if antigens_str:
    terms.extend(utils.parse_and_normalize_vaccines(antigens_str))
  return ", ".join(term for term in terms if term)


class SnomedVaccineIndex:
  """Index for SNOMED Vaccine Codes."""

  def __init__(
      self,
      df: pd.DataFrame,
  ):
    """Initializes the vaccine index with the preprocessed DataFrame."""
    self.master_df = df

  @classmethod
  def from_index_file(cls, path: str, **kwargs: Any) -> "SnomedVaccineIndex":
    """Loads and builds a SnomedVaccineIndex from an index file.

    Args:
      path: The path to the input index file (csv/tsv/txt).
      **kwargs: Additional arguments to pass to pandas.read_csv.

    Returns:
      An instance of SnomedVaccineIndex populated with data from the index file.

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
    }
    read_csv_args.update(kwargs)

    df = pd.read_csv(path, **read_csv_args)
    df = cls._preprocess(df)
    return cls(df)

  @classmethod
  def _preprocess(cls, df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw vaccine DataFrame with normalized variations and types.

    Args:
      df: Raw input DataFrame from the index file.

    Returns:
      Preprocessed DataFrame with normalized columns.
    """
    df = df.copy()
    if "clinical_variations" in df.columns:
      df["norm_clinical_variations"] = df.apply(
          _extract_norm_variations, axis=1
      )
    else:
      df["norm_clinical_variations"] = ""

    if "patient_type" in df.columns:
      df["patient_type"] = df["patient_type"].apply(_parse_patient_type)
    else:
      df["patient_type"] = resources.PatientType.UNKNOWN

    return df

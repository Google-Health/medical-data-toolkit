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
"""Index for SNOMED-CT India Extension Branded Drug Codes."""

import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.index import snomed_drug_index


class BrandIndex(snomed_drug_index.SnomedDrugIndex):
  """Index for SNOMED-CT India Extension Branded Drug Codes."""

  @classmethod
  def from_index_file(cls, path: str) -> "BrandIndex":
    """Loads and builds a BrandIndex from an index file.

    Args:
      path: The path to the input index file (csv/txt).

    Returns:
      An instance of BrandIndex populated with data from the index file.

    Raises:
      FileNotFoundError: If the specified path does not exist.
    """
    return super().from_index_file(path, on_bad_lines="skip")

  @classmethod
  def _preprocess(
      cls, df: pd.DataFrame
  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    df["norm_product_id"] = df["Product Identifier"].apply(utils.normalize_id)
    groups = {str(k): g for k, g in df.groupby("norm_product_id")}
    return df, groups

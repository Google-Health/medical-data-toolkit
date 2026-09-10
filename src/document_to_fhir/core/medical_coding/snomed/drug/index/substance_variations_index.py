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
"""Index for SNOMED-CT India Extension Drug Substance Variations."""

import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.index import snomed_drug_index


class SubstanceVariationsIndex(snomed_drug_index.SnomedDrugIndex):
  """Index for SNOMED-CT India Extension Drug Substance Variations."""

  @classmethod
  def from_index_file(cls, path: str) -> "SubstanceVariationsIndex":
    """Loads and builds a SubstanceVariationsIndex from an index file.

    Args:
      path: The path to the input index file (csv/txt).

    Returns:
      An instance of SubstanceVariationsIndex populated with data from the
      index file.

    Raises:
      FileNotFoundError: If the specified path does not exist.
    """
    return super().from_index_file(path, sep="\t", index_col=None)

  @classmethod
  def _preprocess(
      cls, df: pd.DataFrame
  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Preprocesses the DataFrame. To be implemented."""
    df["norm_name"] = df["Substance Name"].apply(
        utils.normalize_snomed_entity_name
    )
    df = df.set_index("norm_name")
    df.variations = df.variations.fillna("")
    df.variations = df.variations.apply(lambda x: x.split(" | "))
    groups = {}

    return df, groups

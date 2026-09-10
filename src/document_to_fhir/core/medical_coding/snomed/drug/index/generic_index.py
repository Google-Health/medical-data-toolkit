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
"""Index for SNOMED-CT India Extension Generic Drug Codes."""

import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug.index import snomed_drug_index


class GenericIndex(snomed_drug_index.SnomedDrugIndex):
  """Index for SNOMED-CT India Extension Generic Drug Codes."""

  @classmethod
  def _make_substance_key(cls, sub_id_str: str):
    if pd.isna(sub_id_str):
      return ""
    return "+".join(sorted(str(sub_id_str).split("+")))

  @classmethod
  def _preprocess(
      cls, df: pd.DataFrame
  ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    # Pre-group GenericMaster by a standardized, sorted substance key.
    # This optimizes substance combination matching to O(1) complexity:
    # 1. 'make_substance_key' splits, sorts, and rejoins substance IDs with '+'
    #    (e.g., '456+123' -> '123+456').
    #    This guarantees that the key is order-independent, matching
    #    combinations regardless of how they are listed.
    # 2. We group 'GenericMaster' by this key into a dictionary. Inside the
    #    mapping loop, retrieving matching generic candidates becomes a simple,
    #    instant dictionary lookup, avoiding a slow scan of the entire sheet.
    df["substance_key"] = df["Substance Identifier"].apply(
        cls._make_substance_key
    )
    groups = {str(k): g for k, g in df.groupby("substance_key")}
    return df, groups

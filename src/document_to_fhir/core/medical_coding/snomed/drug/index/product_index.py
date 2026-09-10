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
"""Index for SNOMED-CT India Extension Drug Product Codes."""

import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.index import snomed_drug_index


class ProductIndex(snomed_drug_index.SnomedDrugIndex):
  """Index for SNOMED-CT India Extension Product Codes."""

  @classmethod
  def _preprocess(
      cls, df: pd.DataFrame
  ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Preprocesses the DataFrame. To be implemented."""
    df["normalized_product_name"] = df["Product Name"].apply(
        utils.normalize_snomed_entity_name
    )
    df["normalized_product_identifier"] = df["Identifier"].apply(
        utils.normalize_id
    )
    groups = df.set_index("normalized_product_name")[
        "normalized_product_identifier"
    ].to_dict()

    return df, groups


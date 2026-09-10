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
"""Snomed Drug Query Engine that coordinates different matching strategies."""

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.drug import config
from src.document_to_fhir.core.medical_coding.snomed.drug.index import brand_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import form_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import generic_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import product_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import route_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_variations_index
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import brand_query
from src.document_to_fhir.core.medical_coding.snomed.drug.query import generic_query


BRAND_MATCHING_CUTOFF = 0.95
GENERIC_MATCHING_CUTOFF = 0.95


class SnomedDrugQueryEngine(base.BaseDrugQueryEngine):
  """Coordinates SNOMED drug query mapping using brand and generic strategies."""

  def __init__(
      self,
      brand_idx: brand_index.BrandIndex,
      generic_idx: generic_index.GenericIndex,
      substance_idx: substance_index.SubstanceIndex,
      product_idx: product_index.ProductIndex,
      substance_variations_idx: substance_variations_index.SubstanceVariationsIndex,
      form_idx: form_index.FormIndex,
      route_idx: route_index.RouteIndex,
  ):
    self.brand_index = brand_idx
    self.generic_index = generic_idx
    self.substance_index = substance_idx
    self.product_index = product_idx
    self.substance_variations_index = substance_variations_idx
    self.form_index = form_idx
    self.route_index = route_idx

    self.brand_query = brand_query.SnomedBrandDrugQueryEngine(
        self, brand_idx, product_idx, form_idx
    )
    self.generic_query = generic_query.SnomedGenericDrugQueryEngine(
        self,
        generic_idx,
        substance_idx,
        substance_variations_idx,
        form_idx,
        route_idx,
    )

  def query(
      self,
      medication: resources.Medication,
      support_alternate_substances: bool = True,
      brand_matching_cutoff: float = BRAND_MATCHING_CUTOFF,
      generic_matching_cutoff: float = GENERIC_MATCHING_CUTOFF,
      verbose: bool = False,
  ) -> base.DrugResult:
    """Returns a default SNOMED drug code."""

    predicted_code = None
    display_name = None

    # Brand matching is attempted first, if enabled.
    if config.ENABLE_BRAND:
      predicted_code, display_name = self.brand_query.query(
          medication,
          verbose=verbose,
          brand_matching_cutoff=brand_matching_cutoff,
      )

    # Generic matching is attempted if brand matching failed and is enabled.
    if not predicted_code and config.ENABLE_GENERIC:
      predicted_code, display_name = self.generic_query.query(
          medication,
          verbose=verbose,
          generic_matching_cutoff=generic_matching_cutoff,
          support_alternate_substances=support_alternate_substances,
      )

    code_system = "SNOMED-CT" if predicted_code else None
    return base.DrugResult(
        code=predicted_code,
        display_name=display_name,
        code_system=code_system,
    )

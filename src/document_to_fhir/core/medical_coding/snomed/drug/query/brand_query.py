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
"""Mapper for Brand SNOMED drug codes."""

import difflib
import re

from absl import logging
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.drug import config
from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.index.brand_index import BrandIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.form_index import FormIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.product_index import ProductIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import snomed_drug_query


class SnomedBrandDrugQueryEngine(snomed_drug_query.SnomedDrugQuery):
  """Mapper for Brand SNOMED drug codes."""

  def __init__(
      self,
      engine: base.BaseDrugQueryEngine,
      brand_index: BrandIndex,
      product_index: ProductIndex,
      form_index: FormIndex,  # pylint: disable=unused-argument
  ):
    super().__init__(engine)
    self.brand_groups = brand_index.groups
    self.product_exact_dict = product_index.groups
    self.fuzzy_cache = {}

  def query(
      self,
      medication: resources.Medication,
      brand_matching_cutoff: float,
      **kwargs,
  ) -> tuple[str | None, str | None]:
    """Maps a medication to a SNOMED brand drug code.

    This engine attempts to identify the product ID from the brand name (or
    active substances if brand name is missing) and then find candidates in the
    Brand Index using this product ID. The candidates are optionally filtered
    based on strength, dose form and route of administration as defined in the
    config.

    Args:
      medication: The medication resources containing brand name, active
        substances, strength, etc.
      brand_matching_cutoff: The threshold ratio for fuzzy matching brand names.
      **kwargs: Additional arguments.

    Returns:
      A tuple containing the predicted SNOMED code and display name, or
      (None, None) if no match is found.
    """
    brand_name = medication.brand_name.strip() if medication.brand_name else ""

    if not brand_name:
      # Some active substances are present in brand names, e.g. Paracetamol
      brand_name = (
          medication.active_substances.strip()
          if medication.active_substances
          else ""
      )

    if not brand_name:
      return None, None

    # Normalize the brand name for matching
    norm_brand = utils.normalize_snomed_entity_name(brand_name)
    product_id = self._lookup_product_id(
        brand_name, norm_brand, brand_matching_cutoff
    )

    # If a Product ID is found, search for the product ID in the Brand Index
    if product_id:
      logging.info(
          "Identified as Brand '%s' (Product ID: %s). Running Brand Mapping...",
          brand_name,
          product_id,
      )
      predicted_code, display_name = self._predict_brand_snomed_code(
          medication, product_id
      )

      if predicted_code:
        return str(predicted_code), display_name

    return None, None

  def _lookup_product_id(
      self,
      brand_name: str,
      norm_brand: str,
      brand_matching_cutoff: float,
  ) -> str | None:
    """Finds product ID using exact lookup first, then fuzzy cache/lookup."""
    # If the brand name contains '|' or '/' or '+', this is likely a generic
    # name (containing a combination of active substances). So brand mapping can
    # be skipped.
    if "|" in brand_name or "/" in brand_name or "+" in brand_name:
      return None

    # Search for the exact Brand Name in the Product Index (excluding special
    # characters like space, hyphen etc.). Note that the normalized brand name
    # is expected to match the normalized product name in the Product Index.
    product_id = self.product_exact_dict.get(norm_brand)
    if product_id:
      return product_id

    # Search for the Brand Name in the fuzzy cache if no exact match is present.
    if norm_brand in self.fuzzy_cache:
      cached_id, cached_cutoff = self.fuzzy_cache[norm_brand]
      if cached_id is not None and brand_matching_cutoff <= cached_cutoff:
        return cached_id
      elif cached_id is None and brand_matching_cutoff >= cached_cutoff:
        return None

    # In case, of no exact match, fuzzy matching is performed. High cutoff is
    # preferred to prevent errors in brand name.
    matches = difflib.get_close_matches(
        norm_brand,
        list(self.product_exact_dict.keys()),
        n=1,
        cutoff=brand_matching_cutoff,
    )
    if matches:
      best_match = matches[0]
      word1 = norm_brand.split()[0]
      word2 = best_match.split()[0]
      first_word_ratio = difflib.SequenceMatcher(None, word1, word2).ratio()

      if first_word_ratio >= brand_matching_cutoff:
        product_id = self.product_exact_dict[best_match]
        self.fuzzy_cache[norm_brand] = (product_id, brand_matching_cutoff)
        return product_id

    self.fuzzy_cache[norm_brand] = (None, brand_matching_cutoff)
    return None

  def _match_strengths(
      self, brand_candidates: pd.DataFrame, strengths: list[str]
  ) -> pd.DataFrame:
    """Filters brand candidates that match all the given strengths."""
    matched_candidates = []

    # Identify candidates that match the given combination of strengths.
    # Optionally, perform unit conversions for the strengths. This feature
    # allows for flexibility in matching units, e.g. 0.5mg and 500mcg. SNOMED-CT
    # data does not follow a strict convention for strength units so enabling
    # this feature improves accuracy.
    for _, candidate_row in brand_candidates.iterrows():
      candidate_brand_name = candidate_row["Brand Name"]
      matched_all = True

      for strength in strengths:
        if not strength:
          continue

        if config.BRAND_MATCHING_ENABLE_UNIT_CONVERSION:
          equivs = utils.generate_equivalent_strengths(strength)
        else:
          equivs = [strength]

        matched_any_equiv = False
        for eq in equivs:
          str_escaped = re.escape(eq)
          str_pattern = re.sub(r"(?:\\\s)+", r"\\s*", str_escaped)
          str_pattern = re.sub(
              r"(?<=\d)(?=[a-zA-Z])", lambda m: r"\s*", str_pattern
          )
          str_pattern = str_pattern.replace("/", r"\s*/\s*").replace(
              r"\/", r"\s*/\s*"
          )

          lookahead_slash = r"(?!\s*/)" if "/" not in eq else ""
          pattern = f"(?<![\\d\\.]){str_pattern}{lookahead_slash}(?![\\d\\.])"
          match = re.search(pattern, candidate_brand_name, re.IGNORECASE)

          if match:
            matched_any_equiv = True
            break

        if not matched_any_equiv:
          matched_all = False
          break

      if matched_all:
        logging.info(
            "Brand Candidate '%s' matched all strengths: %s",
            candidate_brand_name,
            strengths,
        )
        matched_candidates.append(candidate_row)

    return pd.DataFrame(matched_candidates)

  def _predict_brand_snomed_code(
      self,
      medication: resources.Medication,
      product_id: str,
  ) -> tuple[str | None, str | None]:
    """Maps a derived brand row to a SNOMED Brand Master ID using O(1) optimizations."""
    strengths = self.parse_field(
        medication.strength if medication.strength else ""
    )
    # Identify all candidates in the Brand Index that contain the Product ID
    brand_candidates = self.brand_groups.get(str(product_id), pd.DataFrame())

    # No candidates found in the Brand Index that match the Product ID.
    if brand_candidates.empty:
      logging.info(
          "No SNOMED Brand Candidate found for %s [Product ID: %s, strength:"
          " %s]",
          medication.brand_name,
          product_id,
          strengths,
      )
      return None, None

    # Identify all candidates that match the strength(s) of the medication.
    if config.BRAND_MATCHING_STRENGTH:
      brand_candidates = self._match_strengths(brand_candidates, strengths)

    if brand_candidates.empty:
      logging.info(
          "No SNOMED Brand Candidate found for %s [Product ID: %s, strength:"
          " %s]",
          medication.brand_name,
          product_id,
          strengths,
      )
      return None, None

    # Identify all candidates that match the dose form of the medication, e.g.
    # "tablet", "capsule", "syrup", "injection" etc.
    if config.BRAND_MATCHING_DOSE_FORM:
      dose_form = medication.medicine_form if medication.medicine_form else ""
      brand_candidates = self.match_dose_form(
          brand_candidates, dose_form, "Brand Name"
      )

    if brand_candidates.empty:
      logging.info(
          "No SNOMED Brand Candidate found for %s [Product ID: %s, strength:"
          " %s]",
          medication.brand_name,
          product_id,
          strengths,
      )
      return None, None

    # Identify the candidates that have matching routes of administration e.g.
    # oral, intravenous, intramuscular etc. This information is not always
    # available in the source document.
    if config.BRAND_MATCHING_ROUTE:
      route = medication.route if medication.route else ""
      brand_candidates = self.match_route(brand_candidates, route, "Brand Name")

    if brand_candidates.empty:
      logging.info(
          "No SNOMED Brand Candidate found for %s [Product ID: %s, strength:"
          " %s]",
          medication.brand_name,
          product_id,
          strengths,
      )
      return None, None

    if len(brand_candidates) == 1:
      predicted_code = brand_candidates.iloc[0]["Identifier"]
      display_name = brand_candidates.iloc[0]["Brand Name"]
      if predicted_code:
        return str(predicted_code), display_name

    # If multiple candidates are found, avoid returning a SNOMED code since
    # there is no way to further narrow down the search window using the
    # available attributes.
    logging.info(
        "Too many SNOMED Brand Candidates found for %s [Product ID: %s,"
        " strength: %s]",
        medication.brand_name,
        product_id,
        strengths,
    )
    return None, None

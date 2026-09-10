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
"""Snomed Vaccine Query Engine that coordinates vaccine matching strategies."""

from collections.abc import Sequence
import re

from absl import logging
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine import utils
from src.document_to_fhir.core.medical_coding.snomed.vaccine.index import snomed_vaccine_index
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import base

# The SNOMED codes below are excluded from consideration as they are too broad.
_EXCLUDED_CODES = (
    "836368004",  # Bacteria antigen-containing vaccine product
    "836369007",  # Virus antigen-containing vaccine product
    "863950005",  # Bacteria and virus antigens-containing vaccine product
)

_REQUIRED_COLUMNS = (
    "norm_clinical_variations",
    "preferred_display_term",
    "snomed_code",
    "patient_type",
)


def _build_term_pattern(term: str) -> str:
  """Builds a regex pattern to match a term as a token in comma-separated text.

  The regex ensures matching whole strings or tokens separated by commas,
  avoiding partial matches within combination vaccines (e.g. "Hib" will
  not match "DTaP-Hib").

  Args:
    term: The vaccine token to match.

  Returns:
    A regex pattern string.
  """
  return rf"(?:^|,)\s*{re.escape(term.strip())}\s*(?:,|$)"


class SnomedVaccineQueryEngine(base.BaseVaccineQueryEngine):
  """Coordinates SNOMED vaccine query mapping strategies."""

  def __init__(
      self,
      vaccine_index: snomed_vaccine_index.SnomedVaccineIndex,
  ):
    """Initializes the query engine with a SnomedVaccineIndex."""
    self.vaccine_index = vaccine_index

  def _build_term_pattern(self, term: str) -> str:
    """Builds a regex pattern to match a term. Delegates to module function."""
    return _build_term_pattern(term)

  def _is_index_valid(self) -> bool:
    """Validates that the vaccine index is ready for queries."""
    return (
        self.vaccine_index is not None
        and self.vaccine_index.master_df is not None
        and not self.vaccine_index.master_df.empty
        and all(
            col in self.vaccine_index.master_df.columns
            for col in _REQUIRED_COLUMNS
        )
    )

  def _find_matches_for_terms(self, terms: Sequence[str]) -> pd.DataFrame:
    """Searches for all terms in variations and sorts matches by match count.

    Args:
      terms: Candidate vaccine terms to search for.

    Returns:
      Matching DataFrame rows sorted by number of terms matched.
    """
    cleaned_terms = [term.strip() for term in terms if term.strip()]
    if not cleaned_terms:
      return pd.DataFrame()

    master_df = self.vaccine_index.master_df
    match_counts = {}

    for term in dict.fromkeys(cleaned_terms):
      pattern = _build_term_pattern(term)
      term_match = master_df["norm_clinical_variations"].str.contains(
          pattern, case=False, na=False, regex=True
      ).astype(int)

      matching_mask = term_match > 0
      if not matching_mask.any():
        continue

      # Sort matches based on dataframe rows matching the most number of terms.
      matches = master_df[matching_mask].copy()

      for _, row in matches.iterrows():
        code = row["snomed_code"]
        if code not in match_counts:
          match_counts[code] = {"_match_count": 0}
        match_counts[code]["_match_count"] += 1

    match_counts_series = master_df["snomed_code"].apply(
        lambda x: match_counts.get(x, {}).get("_match_count", 0)
    )
    max_count = match_counts_series.max()
    if pd.notna(max_count) and max_count > 0:
      return master_df[match_counts_series == max_count]
    return pd.DataFrame()

  def _resolve_candidate_matches(
      self,
      candidate_matches: pd.DataFrame,
      log_prefix: str | None = None,
      vaccine_name: str | None = None,
  ) -> base.VaccineResult | None:
    """Resolves a single VaccineResult from candidate matching rows.

    Args:
      candidate_matches: Filtered DataFrame rows of potential vaccine matches.
      log_prefix: Optional logging prefix for diagnostic messages.
      vaccine_name: Optional original vaccine name from immunization.

    Returns:
      A VaccineResult if a match could be resolved, None otherwise.
    """
    if len(candidate_matches) == 1:
      predicted_code = candidate_matches["snomed_code"].iloc[0]
      display_name = candidate_matches["preferred_display_term"].iloc[0]
      if log_prefix and vaccine_name:
        logging.info(
            "%s: SNOMED code found for vaccine: %s, %s, %s",
            log_prefix,
            vaccine_name,
            predicted_code,
            display_name,
        )
      return base.VaccineResult(
          code=predicted_code,
          display_name=display_name,
          code_system="SNOMED-CT",
      )

    if len(candidate_matches) > 1:
      matched_parent_codes = utils.get_top_parent_code(
          candidate_matches, _EXCLUDED_CODES, parent_col="parent_snomed_code"
      )
      if matched_parent_codes:
        n_matched_codes = len(matched_parent_codes.split(", "))

        # If multiple parents are found, return its child code. This is done
        # because there are cases where a code may have multiple parents. Such
        # child codes are usually combination vaccines (E.g. Hep A + Hep B)
        # whose parents will be the codes for Hep A only antigen and  Hep B only
        # antigen. In such a scenario, since the child code is one of the
        # matches, it is highly likely that the user intended to refer to the
        # combination vaccine and not the individual ones.
        if n_matched_codes > 1:
          matching_row = self.vaccine_index.master_df[
              self.vaccine_index.master_df["parent_snomed_code"]
              == matched_parent_codes
          ]
          if not matching_row.empty:
            return base.VaccineResult(
                code=matching_row["snomed_code"].iloc[0],
                display_name=matching_row["preferred_display_term"].iloc[0],
                code_system="SNOMED-CT",
            )
        else:
          matching_row = self.vaccine_index.master_df[
              self.vaccine_index.master_df["snomed_code"]
              == matched_parent_codes
          ]
          if not matching_row.empty:
            return base.VaccineResult(
                code=matched_parent_codes,
                display_name=matching_row["preferred_display_term"].iloc[0],
                code_system="SNOMED-CT",
            )
    return None

  def _resolve_matches_by_patient_type(
      self,
      matches: pd.DataFrame,
      patient_type: resources.PatientType,
      vaccine_name: str,
  ) -> base.VaccineResult:
    """Resolves matches considering patient type, falling back to general.

    Args:
      matches: DataFrame of matched vaccine candidates.
      patient_type: PatientType enum (ADULT, PAEDIATRIC, UNKNOWN).
      vaccine_name: Original vaccine name.

    Returns:
      The resolved VaccineResult.
    """

    def _matches_patient_type(target: resources.PatientType) -> pd.Series:
      col = matches["patient_type"]
      target_val = target.value if hasattr(target, "value") else str(target)
      mask = (col == target_val) | (col == target_val.lower())
      try:
        mask = mask | (col == target)
      except (TypeError, ValueError):
        # Comparison may fail when Series contains mixed or non-enum types.
        pass
      return mask.fillna(False)

    if patient_type == resources.PatientType.ADULT:
      adult_matches = matches[
          _matches_patient_type(resources.PatientType.ADULT)
      ]
      result = self._resolve_candidate_matches(
          adult_matches,
          log_prefix="Patient is an adult",
          vaccine_name=vaccine_name,
      )
      if result is not None:
        return result

    elif patient_type == resources.PatientType.PAEDIATRIC:
      paediatric_matches = matches[
          _matches_patient_type(resources.PatientType.PAEDIATRIC)
      ]
      result = self._resolve_candidate_matches(
          paediatric_matches,
          log_prefix="Patient is paediatric",
          vaccine_name=vaccine_name,
      )
      if result is not None:
        return result

    general_matches = matches[
        _matches_patient_type(resources.PatientType.UNKNOWN)
    ]
    result = self._resolve_candidate_matches(
        general_matches,
        log_prefix="Patient type is unknown",
        vaccine_name=vaccine_name,
    )
    if result is not None:
      return result

    return base.VaccineResult()

  def query(self, immunization: resources.Immunization) -> base.VaccineResult:
    """Queries SNOMED vaccine codes for the given immunization record.

    Normalizes the vaccine name, finds matching terms, and resolves
    candidates by patient type hierarchy (Adult/Paediatric -> Unknown -> Top
    Parent).

    Args:
      immunization: The immunization record to map.

    Returns:
      VaccineResult containing matching SNOMED code and preferred display name.
    """
    if not self._is_index_valid():
      return base.VaccineResult()

    vaccine_name = immunization.vaccine_name
    if not vaccine_name:
      return base.VaccineResult()

    patient_type = immunization.patient_type

    norm_vaccine_names = utils.parse_and_normalize_vaccines(vaccine_name)
    if not norm_vaccine_names:
      return base.VaccineResult()

    matches = self._find_matches_for_terms(norm_vaccine_names)
    if matches.empty:
      return base.VaccineResult()

    return self._resolve_matches_by_patient_type(
        matches=matches,
        patient_type=patient_type,
        vaccine_name=vaccine_name,
    )

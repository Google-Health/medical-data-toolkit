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
"""Mapper for Generic SNOMED drug codes."""

import difflib
import itertools
import re

from absl import logging
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.drug import config
from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.index.form_index import FormIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.generic_index import GenericIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.route_index import RouteIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_index import SubstanceIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_variations_index import SubstanceVariationsIndex
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import snomed_drug_query


class SnomedGenericDrugQueryEngine(snomed_drug_query.SnomedDrugQuery):
  """Mapper for Generic SNOMED drug codes."""

  def __init__(
      self,
      engine: base.BaseDrugQueryEngine,
      generic_index: GenericIndex,
      substance_index: SubstanceIndex,
      substance_variations_index: SubstanceVariationsIndex,
      form_index: FormIndex,  # pylint: disable=unused-argument
      route_index: RouteIndex,
  ):
    super().__init__(engine)
    self.generic_substance_groups = generic_index.groups
    self.substance_exact_dict = substance_index.groups
    self.substance_variations_df = substance_variations_index.master_df
    self.route_df = route_index.master_df

  def _get_substance_combinations(
      self, active_substances: str
  ) -> list[list[str]]:
    """Generates Cartesian product combinations of substances and their variations."""
    if "|" in active_substances:
      separator = "|"
    elif "+" in active_substances:
      separator = "+"
    else:
      separator = ","
    substances = [
        s.strip() for s in active_substances.split(separator) if s.strip()
    ]

    options_per_position = []
    for sub in substances:
      try:
        variations = self.substance_variations_df.loc[sub.lower()].variations
      except KeyError:
        variations = []

      # dict.fromkeys preserves order while eliminating duplicates if
      # 'sub' is already in variations. Preserving order is important because
      # the goal is to keep the original substance name as the first option.
      unique_options = list(dict.fromkeys([sub.lower()] + variations))
      options_per_position.append(unique_options)

    # Generate all Cartesian product combinations
    return [list(combo) for combo in itertools.product(*options_per_position)]

  def query(
      self,
      medication: resources.Medication,
      generic_matching_cutoff: float,
      support_alternate_substances: bool = True,
      **kwargs,
  ) -> tuple[str | None, str | None]:

    active_substances = (
        medication.active_substances if medication.active_substances else ""
    )

    # If alternate substances are supported, generate all combinations and query
    # the generic index for each combination in case the original
    # combination does not yield a match. For example, "metformin hydrochloride"
    # and "metformin" are often used interchangeably. SNOMED-CT
    # does not follow any strict convention on how the active substances
    # should be listed. In this example, the description in SNOMED-CT would be
    # "Metformin (as Metformin Hydrochloride)". By generating all such
    # combinations, there is a higher probability of finding a match.
    if (
        config.GENERIC_MATCHING_SUPPORT_ALTERNATE_SUBSTANCES
        and active_substances
    ):
      substance_combinations = self._get_substance_combinations(
          active_substances
      )

      for substances in substance_combinations:
        updated_medication = medication
        updated_medication.active_substances = "|".join(substances)

        predicted_code, display_name = self._predict_generic_snomed_code(
            updated_medication,
            updated_medication.active_substances,
            cutoff=generic_matching_cutoff,
        )
        if predicted_code:
          logging.info(
              "SNOMED Generic Candidate found for %s [Substances: %s, strength:"
              " %s]: SNOMED Code: %s, Display Name: %s",
              medication.active_substances,
              substances,
              medication.strength,
              predicted_code,
              display_name,
          )
          return str(predicted_code), display_name
        else:
          continue

      logging.info(
          "No SNOMED Generic Candidate found for %s [strength: %s]",
          medication.active_substances,
          medication.strength,
      )
      return None, None

    else:
      return self._predict_generic_snomed_code(
          medication,
          medication.active_substances,
          cutoff=generic_matching_cutoff,
      )

  def _lookup_substance_ids(
      self, substances: list[str], cutoff: float
  ) -> list[str] | None:
    """Looks up substance IDs using exact matching first, falling back to fuzzy matching."""
    substance_ids = []
    for sub_name in substances:
      if not sub_name:
        continue
      norm_query = utils.normalize_snomed_entity_name(sub_name)
      sub_id = self.substance_exact_dict.get(norm_query)

      if sub_id:
        substance_ids.append(str(sub_id))
      else:
        matches = difflib.get_close_matches(
            norm_query,
            list(self.substance_exact_dict.keys()),
            n=1,
            cutoff=cutoff,
        )
        if matches:
          matched_norm_name = matches[0]
          sub_id = self.substance_exact_dict[matched_norm_name]
          substance_ids.append(str(sub_id))

        # If even a single substance does not have a match, the entire matching
        # process is aborted. This is because it is critical to ensure that the
        # predicted code matches the original medication exactly.
        else:
          return None
    return substance_ids

  def _match_generic_strengths(
      self,
      filtered_generic: pd.DataFrame,
      substances: list[str],
      strengths: list[str],
  ) -> pd.DataFrame:
    """Filters generic candidates that match the substances and strengths."""

    matched_candidates = []
    for _, row in filtered_generic.iterrows():
      generic_name = row["Generic Name"]
      matched_all = True
      for sub, strength in zip(substances, strengths):
        if not sub or not strength:
          continue

        sub_escaped = r"\s*".join([re.escape(p) for p in sub.split(" ")])

        # Generate equivalent strengths if unit conversion is enabled. This will
        # generate alternatives for the strength value with different units. For
        # example, "1 mg" will generate alternatives such as "1000 mcg" and
        # "0.001 g". This improves the probability of a match since SNOMED-CT
        # does not follow any strict convention on how the strength should be
        # listed. All strengths need to match for a candidate to be considered.
        if config.GENERIC_MATCHING_ENABLE_UNIT_CONVERSION:
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

          pattern = (
              f"{sub_escaped}[^\\d]*(?<![\\d\\.]){str_pattern}(?![\\d\\.])"
          )

          if "/" not in eq:
            pattern += r"(?!\s*/)"

          match = re.search(pattern, generic_name, re.IGNORECASE)

          if match:
            matched_any_equiv = True
            break

        if not matched_any_equiv:
          matched_all = False
          break

      if matched_all:
        logging.info(
            "Generic Candidate '%s' matched all substances: %s and"
            " strengths: %s",
            generic_name,
            substances,
            strengths,
        )
        matched_candidates.append(row)

    return pd.DataFrame(matched_candidates)

  def _predict_generic_snomed_code(
      self,
      medication: resources.Medication,
      substances_str: str,
      cutoff=0.6,
  ) -> tuple[str | None, str | None]:
    """Maps a generic entry to a SNOMED Generic Master ID using O(1) optimizations."""
    substances = (
        [s.strip() for s in substances_str.split("|")] if substances_str else []
    )
    if not substances:
      # Some active substances are present in brand names, e.g. Paracetamol"
      if medication.brand_name:
        substances = [
            s.strip() for s in medication.brand_name.split("|") if s.strip()
        ]
      else:
        substances = None

    if not substances:
      return None, None

    strengths = self.parse_field(medication.strength)
    dose_form = medication.medicine_form if medication.medicine_form else ""
    route = medication.route if medication.route else ""

    # Identify exact substance IDs for all present active substances.
    substance_ids = self._lookup_substance_ids(substances, cutoff)
    if not substance_ids:
      return None, None

    # Identify all candidates that are a combination of ALL the identified
    # substance IDs.
    lookup_key = "+".join(sorted(substance_ids))
    filtered_generic = self.generic_substance_groups.get(
        lookup_key, pd.DataFrame()
    )

    if filtered_generic.empty:
      return None, None

    # Identify the candidates that have matching strengths for each identified
    # active substance.
    if config.GENERIC_MATCHING_STRENGTH:
      filtered_generic = self._match_generic_strengths(
          filtered_generic, substances, strengths
      )

    if filtered_generic.empty:
      return None, None

    if len(filtered_generic) == 1:
      predicted_code = filtered_generic.iloc[0]["Identifier"]
      if not pd.isna(predicted_code):
        return str(predicted_code), filtered_generic.iloc[0]["Generic Name"]
      else:
        return None, None

    # Identify the candidates that have matching dose forms e.g. tablet,
    # capsule, syrup, injection etc.
    if config.GENERIC_MATCHING_DOSE_FORM:
      filtered_generic = self.match_dose_form(
          filtered_generic, dose_form, "Generic Name"
      )

    if filtered_generic.empty:
      return None, None

    if len(filtered_generic) == 1:
      predicted_code = filtered_generic.iloc[0]["Identifier"]
      display_name = filtered_generic.iloc[0]["Generic Name"]
      if predicted_code:
        return str(predicted_code), display_name
      else:
        return None, None

    # Identify the candidates that have matching routes of administration e.g.
    # oral, intravenous, intramuscular etc. This information is not always
    # available in the source document.
    if config.GENERIC_MATCHING_ROUTE:
      filtered_generic = self.match_route(
          filtered_generic, route, "Generic Name"
      )

    if filtered_generic.empty:
      logging.info(
          "No SNOMED Generic Candidates found for %s [strength: %s]",
          medication.active_substances,
          medication.strength,
      )
      return None, None

    if len(filtered_generic) == 1:
      predicted_code = filtered_generic.iloc[0]["Identifier"]
      display_name = filtered_generic.iloc[0]["Generic Name"]
      if predicted_code:
        return str(predicted_code), display_name

    # If multiple candidates are found, avoid returning a SNOMED code since
    # there is no way to further narrow down the search window using the
    # available attributes.
    logging.info(
        "Too many SNOMED Generic Candidates found for %s [strength: %s]",
        medication.active_substances,
        medication.strength,
    )
    return None, None

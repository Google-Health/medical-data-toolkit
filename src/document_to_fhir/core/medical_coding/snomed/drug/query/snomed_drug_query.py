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
"""Base classes and protocols for SNOMED drug query engines."""

import difflib
import re
import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug import utils
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base


class SnomedDrugQuery(base.BaseDrugQuery):
  """Base class for SNOMED drug queries."""

  def parse_field(self, field_str: str | None) -> list[str]:
    """Parses a separated string field into a list of strings."""
    if not field_str:
      return []
    # Split by '+', ',', '|'
    parts = re.split(r"[+\|,\t\n]", field_str)
    return [p.strip() for p in parts if p.strip()]

  def match_dose_form(
      self, candidates_df: pd.DataFrame, dose_form: str | None, field_name: str
  ) -> pd.DataFrame:
    """Filters candidates by matching their dose forms."""
    if not dose_form:
      return candidates_df

    norm_form_query = utils.normalize_snomed_entity_name(dose_form)
    exact_matches = candidates_df[
        candidates_df[field_name].astype(str).str.contains(norm_form_query)
    ]

    if not exact_matches.empty:
      return exact_matches
    else:
      matches = difflib.get_close_matches(
          norm_form_query, candidates_df[field_name].tolist(), n=1, cutoff=0.6
      )
      if matches:
        matched_norm_form = matches[0]
        return candidates_df[candidates_df[field_name] == matched_norm_form]
      return candidates_df

  def match_route(
      self, candidates_df: pd.DataFrame, route: str | None, field_name: str
  ) -> pd.DataFrame:
    """Filters candidates by matching their routes."""
    if not route:
      return candidates_df

    norm_route_query = utils.normalize_snomed_entity_name(route)
    exact_matches = candidates_df[
        candidates_df[field_name].astype(str).str.contains(norm_route_query)
    ]
    if not exact_matches.empty:
      return exact_matches
    else:
      matches = difflib.get_close_matches(
          norm_route_query, candidates_df[field_name].tolist(), n=1, cutoff=0.6
      )
      if matches:
        matched_norm_route = matches[0]
        return candidates_df[candidates_df[field_name] == matched_norm_route]
      return candidates_df

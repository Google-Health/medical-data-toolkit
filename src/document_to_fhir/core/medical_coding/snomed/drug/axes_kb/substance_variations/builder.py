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
"""Builds a knowledge base of substance variations from SNOMED drug data.

This module contains functions and classes to read SNOMED drug substance and
generic data, cluster substance variations (including salt forms found in
combination drugs), and build a SubstanceVariationsIndex for efficient lookups.
"""

import re
from typing import Any

from absl import logging
import numpy as np
import pandas as pd
import tqdm.auto

from src.document_to_fhir.core.medical_coding.snomed.drug import utils

_SUBSTANCE_VARIANT_PATTERN = re.compile(
    r"\b(([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s*\(\s*as\s+([^)]+)\))"
)

class SubstanceVariationsKBBuilder:
  """Builds a knowledge base of substance variations by clustering SNOMED data.

  This class takes DataFrames of SNOMED substance codes and generic drugs, and
  clusters substance variations (such as salt forms) across non-combination and
  combination drugs. It can save the enriched variation clusters to a CSV file
  and build a SubstanceVariationsIndex for efficient lookups.
  """

  def __init__(
      self,
      substance_df: pd.DataFrame,
      generic_df: pd.DataFrame,
  ):
    self._substance_df = substance_df
    self._generic_df = generic_df
    self._prepare_data()

  def _prepare_data(self) -> None:
    """Prepares and normalizes input DataFrames."""
    if "Identifier" in self._substance_df.columns:
      self._substance_df.set_index("Identifier", inplace=True)

    if "norm_name" not in self._substance_df.columns:
      if "Substance Name" in self._substance_df.columns:
        self._substance_df["norm_name"] = (
            self._substance_df["Substance Name"]
            .fillna("")
            .astype(str)
            .str.lower()
        )
      else:
        raise ValueError("Column 'Substance Name' not found in substance_df")

    if "norm_name" not in self._generic_df.columns:
      if "Generic Name" in self._generic_df.columns:
        self._generic_df["norm_name"] = (
            self._generic_df["Generic Name"].fillna("").astype(str).str.lower()
        )
      else:
        raise ValueError("Column 'Generic Name' not found in generic_df")

  def _find_substance_variant_in_combination_drug(
      self, text: str, active_substance: str
  ) -> str | None:
    """Finds a substance variant (e.g., salt form) in a combination drug string.

    Matches patterns like "Amoxicillin (as amoxicillin sodium)" and returns the
    inner variant name ("amoxicillin sodium") if active_substance matches the
    outer substance name.

    Args:
      text: The generic drug name or combination text to search within.
      active_substance: The active substance name to match against.

    Returns:
      The inner substance variant string if found, otherwise None.
    """
    if not text or not active_substance:
      return None

    for match in _SUBSTANCE_VARIANT_PATTERN.finditer(text):
      outer_string = match.group(1)
      inner_string = match.group(3)

      if active_substance.lower() in outer_string.lower():
        return inner_string
    return None

  def _find_generic_matches_for_substance(
      self, active_substance: str, substance_id: Any
  ) -> pd.DataFrame:
    """Finds generic drugs that mention active_substance as a variant."""
    return self._generic_df[
        (
            self._generic_df["norm_name"].str.contains(
                f"{active_substance} (as ", regex=False
            )
        )
        & ~(self._generic_df["Substance Identifier"] == substance_id)
    ]

  def _extract_variants_from_combination_drugs(
      self, generic_matches: pd.DataFrame, active_substance: str
  ) -> list[str]:
    """Extracts variant names from combination drug generic names."""
    substance_names = []
    generic_names = generic_matches["norm_name"].unique().tolist()
    for name in generic_names:
      substance_name = self._find_substance_variant_in_combination_drug(
          name, active_substance
      )
      if substance_name:
        substance_names.append(substance_name)
    return np.unique(substance_names).tolist()

  def _find_alternate_substances(
      self, active_substance: str, substance_id: Any
  ) -> list[str] | None:
    """Finds alternate substance variants for a given active substance.

    Args:
      active_substance: The name of the substance to find variants for.
      substance_id: The identifier of the substance.

    Returns:
      A list of alternate substance variant names if possible generic matches
      are found, or None if no generic matches exist.
    """
    possible_matches = self._find_generic_matches_for_substance(
        active_substance, substance_id
    )
    if possible_matches.empty:
      return None

    alternate_ids = (
        possible_matches["Substance Identifier"].unique().astype(str).tolist()
    )

    # Identify non-combination drugs i.e. drugs with a single substance ID.
    non_combination_substance_ids = [
        id_val for id_val in alternate_ids if "+" not in id_val
    ]

    # If no non-combination drugs are found, use combination drugs.
    if not non_combination_substance_ids:
      return self._extract_variants_from_combination_drugs(
          possible_matches, active_substance
      )

    return (
        self._substance_df[
            self._substance_df.index.astype(str).isin(
                non_combination_substance_ids
            )
        ]["norm_name"]
        .unique()
        .tolist()
    )

  def _add_to_clusters(
      self,
      substance_clusters: dict[str, dict[str, list[str]]],
      active_substance: str,
      alternate_substances: list[str],
  ) -> None:
    """Adds substance variations bidirectionally to the clusters mapping."""
    substance_clusters.setdefault(active_substance, {"variations": []})
    substance_clusters[active_substance]["variations"].extend(
        alternate_substances
    )

    for matched_substance in alternate_substances:
      substance_clusters.setdefault(matched_substance, {"variations": []})
      substance_clusters[matched_substance]["variations"].append(
          active_substance
      )

  def _generate_substance_clusters(
      self,
  ) -> dict[str, dict[str, list[str]]]:
    """Iterates through all substances to generate variation clusters."""
    substances = self._substance_df["norm_name"].unique().tolist()
    substance_clusters: dict[str, dict[str, list[str]]] = {}

    for substance in tqdm.auto.tqdm(
        substances, desc="Generating Substance Variations"
    ):
      matching_substances = self._substance_df[
          self._substance_df["norm_name"] == substance
      ]
      if matching_substances.empty:
        continue
      substance_id = matching_substances.index[0]
      alternate_substances = self._find_alternate_substances(
          substance, substance_id
      )
      if alternate_substances is not None:
        self._add_to_clusters(
            substance_clusters, substance, alternate_substances
        )

    return substance_clusters

  def _create_clusters_dataframe(
      self, substance_clusters: dict[str, dict[str, list[str]]]
  ) -> pd.DataFrame:
    """Formats the substance clusters dictionary into a pandas DataFrame."""
    if not substance_clusters:
      logging.warning("No substance clusters generated.")
      df = pd.DataFrame(columns=["variations"])
      df.index.name = "Substance Name"
      return df

    df = pd.DataFrame.from_dict(substance_clusters, orient="index")
    df.index.name = "Substance Name"
    df["variations"] = df["variations"].apply(
        lambda x: " | ".join(dict.fromkeys(x)) if isinstance(x, list) else ""
    )
    return df

  def _save_clusters(
      self, substance_clusters_df: pd.DataFrame, save_path: str
  ) -> None:
    """Saves the substance clusters DataFrame to a file."""
    logging.info("Saving substance variations KB to: %s", save_path)
    sep = "," if save_path.endswith(".csv") else "\t"
    try:
      with utils.open_file(save_path, "w") as f:
        substance_clusters_df.reset_index().to_csv(f, sep=sep, index=False)
      logging.info("Saved substance variations KB to %s", save_path)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Failed to save substance variations KB to %s: %s",
          save_path,
          e,
      )

  def build_kb(
      self,
      save_txt_path: str = "",
  ) -> pd.DataFrame:
    """Builds the substance variations KB and returns a DataFrame.

    Args:
      save_txt_path: An optional path to save the enriched KB as a file. If
        empty, the records are not saved.

    Returns:
      A DataFrame populated with the substance variation clusters.
    """
    save_path = save_txt_path
    logging.info(
        "--- Starting Substance Variations KB Build (Substances: %d) ---",
        len(self._substance_df),
    )

    substance_clusters = self._generate_substance_clusters()
    substance_clusters_df = self._create_clusters_dataframe(substance_clusters)

    if save_path:
      self._save_clusters(substance_clusters_df, save_path)

    return substance_clusters_df

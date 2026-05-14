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
"""Logic for mapping specimen synonyms to canonical LOINC systems.

This module provides a class to translate raw specimen strings into
standardized LOINC system categories.
"""


import pandas as pd


class SpecimenToSystemMapper:
  """Maps specimen synonyms to canonical LOINC systems."""

  def __init__(self, specimen_to_systems: dict[str, set[str]]):
    """Initializes the mapper with the systems KB.

    Args:
        specimen_to_systems: Mapping from specimen synonyms to canonical LOINC
          systems (e.g., {"serum": {"Ser/Plas", "Ser"}}).
    """
    self.specimen_to_systems = specimen_to_systems

  @classmethod
  def from_csv(cls, csv_path: str) -> "SpecimenToSystemMapper":
    """Loads mapping from CSV file and builds SpecimenToSystemMapper."""
    specimen_to_systems = {}
    try:
      df = pd.read_csv(csv_path)

      if "canonical" in df.columns and "synonym" in df.columns:
        for _, row in df.iterrows():
          canonical = str(row["canonical"]).strip()
          synonym = str(row["synonym"]).strip().lower()

          # Initialize as a set if the key doesn't exist
          if synonym not in specimen_to_systems:
            specimen_to_systems[synonym] = set()

          # Sets automatically handle uniqueness, so no need to check if
          # canonical system exists
          specimen_to_systems[synonym].add(canonical)

      else:
        print(
            f"Warning: CSV {csv_path} missing 'canonical' or 'synonym' columns."
        )
    except FileNotFoundError:
      print(f"Warning: System mapping file not found at {csv_path}.")

    return cls(specimen_to_systems)

  def get_canonical_systems(self, raw_specimen: str) -> set[str]:
    """Translates the raw specimen to mapping LOINC systems.

    Args:
        raw_specimen: Free-text specimen from the lab report (e.g., "serum").

    Returns:
        Set of canonical LOINC systems.
    """
    if not raw_specimen:
      return set()

    raw_specimen_sig = self._get_signature(raw_specimen)
    return self.specimen_to_systems.get(raw_specimen_sig, set())

  def _get_signature(self, raw_specimen: str) -> str:
    """Normalizes the raw specimen string to use as a lookup key."""
    if not raw_specimen:
      return ""
    return raw_specimen.strip().lower()

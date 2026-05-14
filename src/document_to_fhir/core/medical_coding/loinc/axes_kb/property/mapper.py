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
"""Logic for mapping units to canonical LOINC properties.

This module provides a class to translate raw unit strings into
standardized LOINC property categories.
"""

import pandas as pd


class UnitToPropertyMapper:
  """Maps units to canonical LOINC properties."""

  def __init__(self, unit_to_properties: dict[str, set[str]]):
    """Initializes the mapper with the property KB.

    Args:
        unit_to_properties: Mapping from unit synonyms to canonical LOINC
          properties (e.g., {"mg/dl": {"MCnc"}}).
    """
    self.unit_to_properties = unit_to_properties

  @classmethod
  def from_csv(cls, csv_path: str) -> "UnitToPropertyMapper":
    """Loads mapping from CSV file and builds UnitToPropertyMapper."""
    unit_to_properties = {}
    try:
      df = pd.read_csv(csv_path)

      if "property" in df.columns and "synonym" in df.columns:
        for _, row in df.iterrows():
          prop = str(row["property"]).strip()
          unit = str(row["synonym"]).strip().lower()

          # Initialize as a set if the key doesn't exist
          if unit not in unit_to_properties:
            unit_to_properties[unit] = set()

          # Sets automatically handle uniqueness, so no need to check if
          # property exists
          unit_to_properties[unit].add(prop)

      else:
        print(
            f"Warning: CSV {csv_path} missing 'property' or 'synonym' columns."
        )
    except FileNotFoundError:
      print(f"Warning: Property mapping file not found at {csv_path}.")

    return cls(unit_to_properties)

  def get_canonical_properties(self, raw_unit: str) -> set[str]:
    """Translates the raw unit to mapping LOINC properties.

    Args:
        raw_unit: Free-text unit from the lab report.

    Returns:
        Set of inferred LOINC properties.
    """
    if not raw_unit:
      return set()

    raw_unit_sig = self._get_signature(raw_unit)
    return self.unit_to_properties.get(raw_unit_sig, set())

  def _get_signature(self, raw_unit: str) -> str:
    """Normalizes the raw unit string to use as a lookup key."""
    if not raw_unit:
      return ""
    return raw_unit.strip().lower()

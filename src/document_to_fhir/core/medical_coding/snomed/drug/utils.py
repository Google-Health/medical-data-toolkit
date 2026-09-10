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
"""Utility functions for Pre-processing SNOMED-CT India Extension Drug Code Data."""

import itertools
import re
from typing import Any

import pandas as pd

from src.document_to_fhir.common import file_util

file_exists = file_util.file_exists
open_file = file_util.open_file
read_text_file = file_util.read_text_file


def read_snomed_drug_kb(filepath: str, **kwargs: Any) -> pd.DataFrame | None:
  """Reads a SNOMED drug knowledge base file into a pandas DataFrame."""
  if not file_exists(filepath):
    return None
  read_args = {
      "sep": "\t",
      "index_col": False,
      "dtype": str,
  }
  read_args.update(kwargs)
  with open_file(filepath, "r") as f:
    return pd.read_csv(f, **read_args)


def normalize_snomed_entity_name(s):
  """Normalizes the input string by converting to lowercase, replacing specific punctuation with spaces, and collapsing spaces.

  Args:
    s: The input string to normalize.

  Returns:
    The normalized string, stripped of leading/trailing whitespace.
  """
  if pd.isna(s) or s is None:
    return ""
  s = str(s).lower()
  s = s.replace(
      "+", " + "
  )  # Pad '+' with spaces to handle spacing inconsistencies safely
  s = re.sub(
      r"[/,\&|()\[\]\-]", " ", s
  )  # Replace specific separators with spaces (removed + from here)
  s = re.sub(
      r"[^\w\s\+]", "", s
  )  # Strip other punctuation completely, but PRESERVE +
  s = re.sub(r"\s+", " ", s)  # Collapse multiple spaces into one
  return s.strip()


def normalize_id(val):
  """Normalizes an identifier to a clean integer string, stripping float decimals."""
  if pd.isna(val):
    return ""

  s = str(val).strip()
  if not s:
    return ""

  # Strips trailing .0 decimals that occur when integer IDs are read/written as
  # floats.
  if "." in s:
    parts = s.split(".")
    if (
        len(parts) == 2
        and parts[1].isdigit()
        and all(c == "0" for c in parts[1])
    ):
      return parts[0]

  return s


_UNIT_MAP = {
    "mcg": "mcg",
    "microgram": "mcg",
    "micrograms": "mcg",
    "µg": "mcg",
    "μg": "mcg",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "ml": "ml",
    "mL": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "l": "l",
    "L": "l",
    "liter": "l",
    "liters": "l",
}


def parse_part(part: str) -> tuple[float, str, bool] | None:
  """Parses a strength part into (value, unit, is_implicit)."""
  part = part.strip()
  if not part:
    return None
  # Try matching value + unit
  match = re.match(r"^([0-9\.]+)\s*([a-zA-Zμµ%]+)$", part)
  if match:
    return float(match.group(1)), match.group(2), False
  # Try matching just unit (implicit 1)
  match_unit = re.match(r"^([a-zA-Zμµ%]+)$", part)
  if match_unit:
    return 1.0, match_unit.group(1), True
  return None


def get_conversions(
    value: float, unit: str, is_implicit: bool
) -> list[tuple[float, str, bool]]:
  """Generates compatible unit conversions for a single value and unit."""
  norm_unit = _UNIT_MAP.get(unit.lower(), unit.lower())
  results = [(value, norm_unit, is_implicit)]

  if norm_unit == "mg":
    results.append((value / 1000.0, "g", False))
    results.append((value * 1000.0, "mcg", False))
  elif norm_unit == "g":
    results.append((value * 1000.0, "mg", False))
  elif norm_unit == "ml":
    results.append((value / 1000.0, "l", False))
  elif norm_unit == "l":
    results.append((value * 1000.0, "ml", False))

  return results


def generate_equivalent_strengths(strength_str: str) -> list[str]:
  """Generates equivalent strength strings by performing unit conversions."""
  if not strength_str:
    return []

  parts = strength_str.split("/")
  if len(parts) > 1:
    # Concentrations (with '/') are not converted
    return [strength_str]

  parsed_parts = []
  for part in parts:
    parsed = parse_part(part)
    if not parsed:
      return [strength_str]
    parsed_parts.append(parsed)

  conversions_per_part = []
  for val, unit, impl in parsed_parts:
    conversions_per_part.append(get_conversions(val, unit, impl))

  combinations = itertools.product(*conversions_per_part)

  results = []
  for combo in combinations:
    formatted_parts = []
    for val, unit, impl in combo:
      if impl and val == 1.0:
        formatted_parts.append(unit)
      else:
        formatted_parts.append(f"{val:g}{unit}")
    results.append("/".join(formatted_parts))

  return results

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
"""Logic for mapping result values to canonical LOINC scales."""


class ScaleMapper:
  """Maps result values to canonical LOINC scales."""

  def get_canonical_scales(self, raw_result: str) -> set[str]:
    """Translates the raw result to expected LOINC scale(s).

    Args:
        raw_result: Free-text result value from the lab report.

    Returns:
        Set of expected LOINC scales.
    """
    if not raw_result:
      return set()

    clean_result = str(raw_result).strip()

    # 1. Check for Quantitative (Qn) - Simple float parse to match old behavior
    try:
      float(clean_result)
      return {'Qn'}
    except ValueError:
      pass

    # 2. Check for Ordinal (Ord)
    ordinal_terms = {
        'positive',
        'negative',
        'detected',
        'not detected',
        'trace',
        'reactive',
        'non-reactive',
        '+',
        '++',
        '+++',
    }
    if clean_result.lower() in ordinal_terms:
      return {'Ord'}

    # 3. Fallback to Ordinal, Nominal or Narrative for non-numeric results
    return {'Ord', 'Nom', 'Nar'}

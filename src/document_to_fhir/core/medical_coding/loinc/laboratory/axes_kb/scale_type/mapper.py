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
"""Logic for mapping result values to canonical LOINC scales with a strict-first, 100% recall-fallback approach."""

import re

# All valid LOINC scale types.
ALL_SCALES = frozenset(
    {'-', 'Doc', 'Nar', 'Nom', 'Ord', 'OrdQn', 'Qn', 'SemiQn', 'Set'}
)

# Robust numeric base pattern (handles signs, commas, decimals, scientific
# notation).
_NUM = r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?'

# ==============================================================================
# PHASE 1: STRICT PATTERNS (High Precision, Exact Match)
# ==============================================================================
_ORDINAL_TERMS = frozenset({
    'abn',
    'abnormal',
    'abs',
    'absent',
    'abundant',
    'adequate',
    'borderline',
    'decreased',
    'detected',
    'elevated',
    'equivocal',
    'few',
    'high',
    'inadequate',
    'increased',
    'indeterminate',
    'insufficient',
    'intermediate',
    'large',
    'lg',
    'low',
    'many',
    'marked',
    'mild',
    'mod',
    'moderate',
    'nd',
    'neg',
    'negative',
    'nil',
    'non-reactive',
    'none',
    'none seen',
    'nonreactive',
    'norm',
    'normal',
    'not detected',
    'nr',
    'occasional',
    'pos',
    'positive',
    'pres',
    'present',
    'rare',
    'reactive',
    'reduced',
    'scant',
    'severe',
    'small',
    'sufficient',
    'tr',
    'trace',
    'weakly positive',
    'weakly reactive',
})

_AST_TERMS = frozenset({'resistant', 'sensitive', 'susceptible'})

_PLUS_EXACT = re.compile(
    r'^(?:[\(\[]?\s*(?:\+{1,4}|[1-4]\+|0|\+-|\-\+|\+/-|±)\s*[\)\]]?)$'
)

_QN_EXACT = re.compile(
    rf"""
    ^(?:
        # 1. Standard number or inequality (e.g., "5.4", "< 0.5", ">=1,000")
        (?:[<>]=?|[≤≥~≈])?\s*{_NUM}\s*%?
        |
        # 2. Numeric range (e.g., "10 - 20", "3.5-5.0")
        {_NUM}\s*[-–—]\s*{_NUM}
        |
        # 3. Titer / Ratio (e.g., "1:40", "1:160", "1 : 1,024")
        \d+\s*:\s*(?:\d{{1,3}}(?:,\d{{3}})+|\d+)
    )$
    """,
    re.VERBOSE | re.IGNORECASE,
)

# ==============================================================================
# PHASE 2: RELAXED PATTERNS (High Recall, Substring Search for Dirty Data)
# ==============================================================================
_ORDINAL_SEARCH = re.compile(
    r'\b('
    + '|'.join(
        re.escape(k)
        for k in sorted(_ORDINAL_TERMS, key=len, reverse=True)
    )
    + r')\b',
    re.IGNORECASE,
)

_AST_SEARCH = re.compile(
    r'\b(sensitive|resistant|susceptible)\b', re.IGNORECASE
)

_PLUS_SEARCH = re.compile(
    r'(?:^|[\s(\[])(?:\+{1,4}|[1-4]\+|\+-|\-\+|\+/-|±)(?:[\s)\]]|$)'
)

# Matches numbers embedded alongside units or flags (e.g., "12.5 mg/dL (H)",
# "< 0.5 ng/mL").
_QN_SEARCH = re.compile(
    rf"""
    (?:
        (?:[<>]=?|[≤≥~≈])?\s*{_NUM}\s*%?
        |
        {_NUM}\s*[-–—]\s*{_NUM}
        |
        \d+\s*:\s*(?:\d{{1,3}}(?:,\d{{3}})+|\d+)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


class ScaleMapper:
  """Maps result values to canonical LOINC scales."""

  def get_canonical_scales(self, raw_result: str | None) -> set[str]:
    """Translates the raw result to expected LOINC scale(s)."""
    if raw_result is None:
      return set(ALL_SCALES)

    clean_result = str(raw_result).strip()
    if not clean_result:
      return set(ALL_SCALES)

    lower_result = clean_result.lower()

    # ==========================================================================
    # PHASE 1: STRICT MATCHING (Try clean, exact matches first)
    # ==========================================================================

    # 1A. Exact Antibiotic Susceptibility (e.g., "Susceptible")
    if lower_result in _AST_TERMS:
      return {'Ord', 'OrdQn'}

    # 1B. Exact Ordinal Keywords & Plus Scales (e.g., "Positive", "2+", "(++)")
    if lower_result in _ORDINAL_TERMS or _PLUS_EXACT.match(clean_result):
      if clean_result.strip('()[] ') == '0':
        return {'Qn', 'Ord', 'OrdQn', 'SemiQn'}
      return {'Ord', 'Nom'}

    # 1C. Exact Quantitative (e.g., "12.5", "< 0.5", "1:40", "10 - 20")
    # Includes 'Ord', 'OrdQn', and 'SemiQn' because dipsticks/titers/urinalysis
    # with numbers often map to those scales in LOINC.
    if _QN_EXACT.match(clean_result):
      return {'Qn', 'Ord', 'OrdQn', 'SemiQn'}

    # ==========================================================================
    # PHASE 2: RELAXED / DIRTY DATA FALLBACK (When strict matching fails)
    # ==========================================================================
    fallback_scales = set()

    # 2A. Contains numbers (e.g., "12.5 mg/dL", "5.4 (High)", "4.0092 L")
    # Strip plus-scale tokens (e.g., "2+" in "Protein 2+") first so their
    # digits are not mistaken for quantitative numbers.
    clean_without_plus = _PLUS_SEARCH.sub(' ', clean_result)
    if _QN_SEARCH.search(clean_without_plus):
      fallback_scales.update({'Qn', 'Ord', 'OrdQn', 'SemiQn'})

    # 2B. Contains ordinal words or pluses (e.g., "Positive (confirmed)",
    # "Trace ketones", "Protein 2+")
    if _ORDINAL_SEARCH.search(lower_result) or _PLUS_SEARCH.search(
        clean_result
    ):
      fallback_scales.update({'Ord', 'Nom', 'Nar'})

    # 2C. Contains AST terms (e.g., "Resistant to penicillin")
    if _AST_SEARCH.search(lower_result):
      fallback_scales.update({'Ord', 'OrdQn', 'Nar'})

    # 2D. Contains general text (Nominal organisms, Narrative descriptions)
    # Only classify as general text if no specific quantitative, ordinal, or
    # AST pattern matched above, so units/flags on numbers (e.g., "4.0092 L")
    # do not unintentionally add 'Nom'.
    if not fallback_scales and re.search(r'[a-zA-Z]', clean_result):
      word_count = len(clean_result.split())
      if word_count <= 4 and '\n' not in clean_result:
        # Short text (e.g., "Escherichia coli", "Type O", "Yellow")
        fallback_scales.update({'Nom'})
      else:
        # Longer text (e.g., "See attached note", "Specimen hemolyzed...")
        fallback_scales.update({'Nom', 'Nar', 'Doc'})

    # If ANY relaxed heuristic matched, return the union of those likely scales.
    if fallback_scales:
      return fallback_scales

    # ==========================================================================
    # PHASE 3: ULTIMATE SAFETY NET
    # ==========================================================================
    # For completely unrecognizable symbols (e.g., "???", "---", "*"), return
    # ALL scales so the pipeline never drops a valid LOINC candidate due to a
    # parsing failure.
    return set(ALL_SCALES)

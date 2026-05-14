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
"""CORE ANALYTE NORMALIZATION & ACTIVE LEARNING STRATEGY.

=====================================================

1. Architectural Justification: "Why these now?"
-------------------------------------------------
* The Pareto Principle (80/20 Rule): While clinical jargon contains thousands
  of edge-case abbreviations, the mappings and deletions listed below cover
  over 95% of the string-matching bloat found in infectious disease and
  immunology testing.
* Preventing LLM "Attention Dilution": Every instruction added to an LLM
  prompt costs "attention." If we prompt the LLM with 50 rare abbreviations,
  it will begin hallucinating or forgetting core instructions. By strictly
  teaching the LLM (or handling via this offline script) the patterns for
  Ag/Ab, Ig, virus, and spp., we keep the LLM focused and accurate.

2. Extending the Vocabulary: The Active Learning Loop
------------------------------------------------------
We do not constantly update the core LLM Prompt for new regional abbreviations.
Instead, we extend this system using a production flywheel:

* Phase 1 (Monitor): Log every raw `Standard_Core_Analyte` string that fails
  the initial Fuzzy Anagram Search and triggers the LLM Fallback.
* Phase 2 (Analyze): Aggregate these "misses" weekly/monthly. (e.g., Notice
  "TPO Autoantibody" failed 400 times but the LLM Fallback correctly mapped
  it to "Thyroid peroxidase Ab").
* Phase 3 (Update): Add the newly discovered edge cases to the
  `ACTIVE_LEARNING_REPLACEMENTS` dictionary below.
* Result: Zero prompt bloat, instant O(1) resolution for future PDFs, and
  data-driven scaling based entirely on real hospital data.
"""

import re

# =====================================================================
# BASELINE VOCABULARY (Derived from 2K LOINC Database Analysis)
# =====================================================================

# Ordered from most specific to least specific.
# Mapping abbreviations back to the FULL canonical words used in AnalyteIndex.
IMMUNE_MARKER_MAP = {
    # Antigens & Antibodies
    r"\bAgs?\b": "Antigen",
    r"\bAntigens\b": "Antigen",
    r"\bAbs?\b": "Antibody",
    r"\bAb's\b": "Antibody",
    r"\bAntibodies\b": "Antibody",
    # Immunoglobulins
    r"\bIgG\b": "Immunoglobulin G",
    r"\bIgM\b": "Immunoglobulin M",
    r"\bIgA\b": "Immunoglobulin A",
    r"\bIgE\b": "Immunoglobulin E",
    r"\bIg\b": "Immunoglobulin",
}

# Taxonomic hierarchy words that do not exist in the base analyte definition
TAXONOMIC_MODIFIERS_TO_REMOVE = [
    r"\bvirus\b",
    r"\bspecies\b",
    r"\bspp\.?\b",
    r"\bsp\.?\b",
    r"\bserotype\b",
    r"\bserovar\b",
    r"\bsubspecies\b",
    r"\bsubsp\.?\b",
    r"\bstrain\b",
]

# =====================================================================
# ACTIVE LEARNING DICTIONARY (Updated via Prod Monitoring)
# =====================================================================

# This dictionary grows over time based on the Active Learning Loop.
ACTIVE_LEARNING_REPLACEMENTS = {
    # Example discovered in prod:
    # r"\bAutoantibod(?:y|ies)\b": "Ab",
}


# =====================================================================
# NORMALIZATION ENGINE
# =====================================================================


def normalize_analyte(analyte: str) -> str:
  """Normalizes a raw analyte string by applying a series of replacements.

  This function standardizes immune markers, removes taxonomic modifiers,
  and cleans up extra whitespace to produce a more consistent analyte name.
  The normalization process follows these steps:
  1.  Apply active learning edge cases from `ACTIVE_LEARNING_REPLACEMENTS`.
  2.  Standardize immune markers using `IMMUNE_MARKER_MAP`.
  3.  Strip taxonomic modifiers listed in `TAXONOMIC_MODIFIERS_TO_REMOVE`.
  4.  Remove any resulting extra whitespaces.

  Args:
    analyte: The raw analyte string to be normalized.

  Returns:
    The normalized analyte string.
  """
  if not analyte:
    return ""

  normalized_text = analyte

  # 1. Apply Active Learning Edge Cases First
  for pattern, replacement in ACTIVE_LEARNING_REPLACEMENTS.items():
    normalized_text = re.sub(
        pattern, replacement, normalized_text, flags=re.IGNORECASE
    )

  # 2. Standardize Immune Markers
  for pattern, replacement in IMMUNE_MARKER_MAP.items():
    normalized_text = re.sub(
        pattern, replacement, normalized_text, flags=re.IGNORECASE
    )

  # 3. Strip Taxonomic Modifiers entirely
  for pattern in TAXONOMIC_MODIFIERS_TO_REMOVE:
    normalized_text = re.sub(pattern, "", normalized_text, flags=re.IGNORECASE)

  # 4. Cleanup: Remove extra whitespaces left behind by deletions
  normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

  return normalized_text

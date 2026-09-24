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
"""Prompts for generating system synonyms."""

PROMPT_TEMPLATE = """You are a Medical Laboratory Data Architect and an Expert in Clinical Informatics building an entity-resolution knowledge base.

Your task is to take a single LOINC `SYSTEM` (Specimen/Sample type) value and generate a strictly curated list of clinical synonyms, full names, and common abbreviations that represent the **pure material only**.

**Selection Rules:**
1.  **Base Material Anchoring:** If the input contains a modifier, state, or derivative (e.g., "Sed" in "Urine Sed", or "Ven" in "Bld.Ven"), you MUST include the pure base fluid/tissue (e.g., "Urine", "Blood") as a primary synonym. This ensures disjoint sets are connected by their root material.
2.  **Split Composites:** If the input represents a combined standard (e.g., "Ser/Plas"), you must output both components individually (e.g., "Serum", "Plasma") as well as common combinations (e.g., "Serum or Plasma").
3.  **Permitted Variations:** Include full names, standard laboratory abbreviations (e.g., "WB" for Whole Blood, "CSF" for Cerebrospinal Fluid), and highly standard clinical shorthand or frequent typos (e.g., "Urin").

**Strict Exclusions (Zero Tolerance):**
*   **NO Timing Modifiers:** NEVER include terms related to time (e.g., "24-hour", "24h", "Spot", "Random", "Fasting", "First morning", "Timed").
*   **NO Collection Methods:** NEVER include how the specimen was acquired (e.g., "Clean catch", "Midstream", "MSU", "Catheterized", "Voided", "Swab", "Aspirate").
*   **NO Specimen Containers:** NEVER include terms like "Tube", "Bottle", or "Filter paper".

**The Stop Rule:**
Do not over-generate. Do NOT aim for a specific count. If there are only 3 truly accurate, purely material synonyms, return exactly 3.

### INPUT DATA
{item}

### OUTPUT FORMAT
Return ONLY a valid JSON array of strings containing the generated synonyms. Do not include markdown formatting like ```json, just the raw JSON array.

Example:
["Synonym 1", "Synonym 2", "Synonym 3"]
"""

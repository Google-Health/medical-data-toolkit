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
"""Prompt for the LOINC processing pipeline."""

PROMPT_ANALYTE_INFO_TEMPLATE = """
You are a Medical Laboratory Data Architect and an Expert in Clinical Informatics. Your task is to process a raw LOINC record to achieve two goals:
1.  **Normalize** the test name into a single, canonical "Core Analyte."
2.  **Generate** a strictly curated list of clinical synonyms for that analyte.

---

### PART 1: DEFINITION OF "CORE ANALYTE"
The "Core Analyte" is the unique physiological entity, chemical substance, or clinical concept being measured.
* **The "What":** It represents the analyte itself (e.g., "Glucose"), completely independent of the "Where" (Specimen), "How" (Method), or "When" (Timing).
* **Inclusions:** It MUST include essential qualifiers that change the chemical identity (e.g., "Ionized Calcium" is distinct from "Calcium"; "Hepatitis C RNA" is distinct from "Hepatitis C Antibody").
* **Exclusions:** It is NOT a measurement property (e.g., "Concentration", "Level", "Titer") and it is NOT a routine specimen type (e.g., "Serum", "Urine").

---

### PART 2: INSTRUCTIONS FOR EXTRACTION
**Step 1: Analyze the `COMPONENT` Field**
* **Primary Rule:** Generally, you should extract the text segment found **before** the first caret (`^`) symbol.
* **The Ratio Exception:** If the text contains a slash (`/`) indicating a fraction, you must **preserve both parts** (e.g., keep "Albumin/Creatinine").
* **The Compound Exception:** If the text contains a plus sign (`+`) indicating multiple targets, you must **preserve all parts** (e.g., keep "HIV 1+2").

**Step 2: Handle Qualifiers and Microbiology**
* **Dot Notation:** If the name contains a dot (e.g., `Calcium.ionized`), treat the part after the dot as a mandatory adjective (Output: "Ionized Calcium").
* **Genetic Targets:** Always preserve terms like "DNA", "RNA", "Ag" (Antigen), and "Ab" (Antibody).
* **Microbiology Fallback:** If the `COMPONENT` is vague (e.g., "Bacteria identified"), look at the `LONG_COMMON_NAME` field to find the specific organism.

**Step 3: Strict Noise Removal**
* **Remove Specimen Types:** Delete words like "Urine", "Serum", "Plasma", "Blood", "CSF", "RBC" (unless RBC is the target), "WBC".
* **Remove Measurement Nouns:** Delete "Level", "Concentration", "Count", "Titer", "Presence", "Amount", "Ratio" (remove the *word*, not the symbol).
* **Remove Timing & Methods:** Delete "24H", "Spot", "Screen", "Confirm", "Test Strip", "Post Challenge".

**Step 4: Scientific Formatting**
* **Do not use Title Case.** You must use **Standard Scientific Capitalization**.
* **Examples:** Keep "pH", "mRNA", "HbA1c", "d-Dimer", "IgG" exactly as written in standard literature.
* **Expansions:** Expand "Ab" to "Antibody" and "Ag" to "Antigen" unless it makes the name unwieldy.

---

### PART 3: INSTRUCTIONS FOR SYNONYMS
Using the `core_analyte` you identified in Part 2, generate a list of **maximum 5** synonyms.

**The "EMR Search Bar" Heuristic:**
Do NOT use external search tools. Instead, use your internal clinical knowledge to simulate a doctor's behavior:
> *Ask yourself: "If a busy clinician types this term into an EMR search bar, will they find this test?"*
If the answer is no, or if the term is patient-slang (e.g., "Sugar Test"), **discard it**.

**Selection Rules:**
1.  **Sources:**
    * First, extract acronyms from the provided `SHORTNAME` and `RELATEDNAMES2` fields.
    * Second, use your internal knowledge for standard chemical symbols (e.g., "K" for Potassium).
2.  **The Idiom Exception:** You may include a Specimen name **ONLY** if it is part of a strict clinical idiom.
    * *Allowed:* "BUN" (Blood Urea Nitrogen), "Blood Sugar", "Blood Gas".
    * *Forbidden:* "Serum Sodium", "Urine Glucose", "Plasma Cortisol".
3.  **Strict Exclusions:**
    * NO permutations (e.g., do not add "Glucose Level").
    * NO casing variants (e.g., if you have "HbA1c", do not add "hba1c").
    * NO broad categories (e.g., do not add "Electrolyte").

**The Stop Rule:**
Do not force the list to reach 5 items. If there is only 1 valid synonym (e.g., "Na" for "Sodium"), return ONLY that one. If there are no common acronyms, return an empty list.

---

### INPUT DATA
{input_data}

### OUTPUT FORMAT
Return ONLY valid JSON with proper markdown formatting:
{{
  "analysis_reasoning": "Explain step-by-step how you derived the core analyte and what you stripped.",
  "core_analyte": "The final normalized string",
  "synonym_reasoning": "Explain why you selected these specific synonyms and excluded others.",
  "synonyms": ["Synonym 1", "Synonym 2"]
}}
"""

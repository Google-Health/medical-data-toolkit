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
You are an Expert Clinical Terminologist and Data Standardizer. Your task is to process a list of raw LOINC records to achieve two goals for each record:
1.  **Normalize** the test name into a single, canonical "Core Analyte."
2.  **Generate** a strictly curated list of clinical synonyms for that analyte.

---

### PART 1: DEFINITION OF "CORE ANALYTE"
The "Core Analyte" is the unique physiological entity, chemical substance, or clinical concept being measured.
* **The "What":** It represents the analyte itself (e.g., "Glucose"), entirely independent of the "Where" (Specimen), "How" (Method), or "When" (Timing).
* **Inclusions (Essential Modifiers):** It MUST preserve essential chemical states and clinical modifiers that change the chemical identity (e.g., "Ionized Calcium" is distinct from "Calcium"; keep "Direct Bilirubin", "Non-HDL Cholesterol", "Transferrin Saturation"). Do NOT strip them down to generic roots.
* **Exclusions:** It is NOT a measurement property (e.g., "Concentration", "Level", "Titer") and it is NOT a routine specimen type (e.g., "Serum", "Urine").

---

### PART 2: INSTRUCTIONS FOR EXTRACTION
**Step 1: Analyze the `COMPONENT` Field**
* **Primary Rule:** Generally, you should extract the text segment found **before** the first caret (`^`) symbol.
* **Handling Ratios, Differentials & Slashes (`/`):**
    * **Mathematical Ratios (Keep Both):** For fractions, ratios, and cell differentials (e.g., a differential test named "Neutrophils" with a percentage unit, "Trig/HDL", or "Albumin/Creatinine"), you MUST preserve or construct the core analyte using **BOTH** the numerator and the denominator separated by a slash (e.g., "Neutrophils/Leukocytes", "Triglycerides/HDL", "Albumin/Creatinine").
    * **Synonyms & Subtypes (Keep One):** If a slash simply implies alternative names or subtypes rather than a true ratio, extract ONLY the primary scientific name (e.g., "PCV/Hematocrit" -> "Hematocrit", "Protein/Albumin" -> "Albumin").
* **The Compound Exception:** If the text contains a plus sign (`+`) indicating multiple targets, you must **preserve all parts** (e.g., keep "HIV 1+2").

**Step 2: Indices, Modifiers, and Microbiology**
* **Base Entities for Indices:** For clinical indices and calculated parameters (e.g., MCHC, MCV, MPV, RDW), extract the **fundamental biological entity**. DROP the descriptive measurement modifiers (e.g., 'Mean', 'Corpuscular', 'Concentration', 'Volume', 'Width', 'Distribution').
    * *Examples:* "MCHC" -> "Hemoglobin" | "MCV" -> "Erythrocytes" | "MPV" -> "Platelets".
* **Dot Notation:** If the name contains a dot (e.g., `Calcium.ionized`), treat the part after the dot as a mandatory adjective (Output: "Ionized Calcium").
* **Genetic Targets:** Always preserve terms like "DNA", "RNA", "Ag" (Antigen), and "Ab" (Antibody).
* **Microbiology Fallback:** If the `COMPONENT` is vague (e.g., "Bacteria identified"), look at the `LONG_COMMON_NAME` field to find the specific organism.

**Step 3: Strict Noise Removal**
* **Remove Specimen Types:** Delete words like "Urine", "Serum", "Plasma", "Blood", "CSF", "RBC" (unless RBC is the target), "WBC".
* **Remove Measurement Nouns:** Delete "Level", "Concentration", "Count", "Titer", "Presence", "Amount", "Ratio" (remove the *word*, not the symbol).
* **Remove Timing & Methods:** Delete "PCR", "24H", "Spot", "Screen", "Confirm", "Test Strip", "Post Challenge".

**Step 4: Scientific Formatting**
* **Standardize Casing:** You must use **Standard Scientific Capitalization**. Do not use Title Case.
* **Examples:** Keep "pH", "mRNA", "HbA1c", "d-Dimer", "IgG", "Free T4" exactly as written in standard literature.
* **Expansions:** Expand "Ab" to "Antibody" and "Ag" to "Antigen" unless it makes the name unwieldy.

---

### PART 3: INSTRUCTIONS FOR SYNONYMS
Using the `core_analyte` you identified, generate a strictly curated list of up to 10 valid clinical synonyms.

**1. Combine Provided Data with Your Expert Memory:**
Review the `RELATEDNAMES2`, `SHORTNAME`, and `LONG_COMMON_NAME` fields for inspiration (e.g., abbreviations, alternate chemical names). However, **do not just copy them blindly**. Rely on your own clinical knowledge to generate and select only the best, most universally recognized synonyms.

**2. The "EMR Search Bar" Heuristic:**
Ask yourself: *"If a busy clinician types this exact term into an EMR search bar to find this test, is it standard medical shorthand?"*
* Allow valid clinical abbreviations (e.g., "WBC", "Leuk", "K", "T4").
* Discard patient slang (e.g., "Sugar Test", "Good Cholesterol").

**3. Aggressively Filter LOINC Noise:**
The `RELATEDNAMES2` field contains massive amounts of non-analyte metadata. You MUST completely ignore:
* **Measurement Nouns & Scales:** "QNT", "Absolute", "Count", "Level", "Ratio", "Number concentration", "Cnt".
* **Routine Specimens:** "Blood", "Serum", "Urine", "WB", "Plasma" (Exception: Only keep if it forms a strict idiom like "BUN" or "Blood Gas").
* **Methods & Timing:** "Auto", "Automated detection", "Point in time", "Random", "PCR", "Screen".
* **Categories:** "UniversalLabOrders", "Chemistry", "HEMATOLOGY".

**4. Strict Exclusions (No Redundancy):**
* **No Plurals/Casing Variants:** If you output "WBC", do not add "WBCs". If you output "White blood cell", do not add "White blood cells".
* **No Appended Nouns:** Do not add descriptive nouns to the synonym (e.g., keep "WBC", do NOT add "WBC Count").

---

### INPUT DATA
{input_data}

### OUTPUT FORMAT
Return ONLY a valid JSON array of objects with proper markdown formatting. The array must contain exactly one object for each input record, in the same order as the input.
Each object in the array must follow this structure:
```json
[
  {{
    "LOINC_NUM": "The LOINC_NUM of the input record",
    "analysis_reasoning": "Explain step-by-step how you derived the core analyte and what you stripped.",
    "core_analyte": "The final normalized string",
    "synonym_reasoning": "Explain why you selected these specific synonyms and excluded others.",
    "synonyms": ["Synonym 1", "Synonym 2", ...]
  }}
]
```
"""

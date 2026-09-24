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
* **Clinical Indices as Autonomous Concepts:** For clinical indices and calculated parameters (e.g., MCHC, MCV, MPV, RDW, PCT, PDW, P-LCR, P-LCC), do NOT collapse them into generic cell counts or whole blood proteins. Preserve them as their standard clinical concepts (e.g., "MCHC", "Mean Corpuscular Volume", "Mean Platelet Volume", "Red Cell Distribution Width", "Plateletcrit").
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
Using the `core_analyte` you identified, generate two separate, strictly curated lists of clinical synonyms:

**A. Strict Synonyms (`strict_synonyms`):**
Exact 1:1 clinical aliases, recognized universal acronyms, orthographic variants, and direct abbreviations that refer unambiguously to this specific analyte:
* Examples:
  - Plateletcrit -> ["Platelet Volume Fraction", "Plt Crit", "Platelet Crit", "Thrombocrit", "PCT"]
  - Direct Bilirubin -> ["Conjugated Bilirubin", "Bilirubin Direct", "Direct Bili", "DBIL"]
  - MCHC -> ["Mean Corpuscular Hemoglobin Concentration", "Mean Cell Hemoglobin Concentration", "MCHC"]
  - MCV -> ["Mean Corpuscular Volume", "Mean Cell Volume", "Erythrocyte Mean Volume", "MCV"]
  - Free Thyroxine -> ["Free T4", "FT4", "Unbound Thyroxine"]
  - Hemoglobin -> ["Hgb", "Hb", "Haemoglobin"]
* The "EMR Search Bar" Heuristic: If a clinician searches this exact term in an EMR or lab catalog, does it unambiguously refer to this specific test?

**B. Lenient Synonyms (`lenient_synonyms`):**
The immediate PARENT core-analyte, unfractionated base chemical compound, or underlying cellular entity (biological hypernyms) that allows fallback retrieval if a query only identifies the base substance:
* Examples:
  - Plateletcrit -> ["Platelets", "Thrombocytes"] (parent cell)
  - Direct Bilirubin -> ["Bilirubin", "Total Bilirubin"] (parent compound)
  - MCV -> ["Erythrocytes", "Red Blood Cells", "RBC"] (parent cell)
  - MCHC -> ["Hemoglobin", "Erythrocytes"] (parent protein/cell)
  - Free Thyroxine -> ["Thyroxine", "T4"] (parent hormone)
  - Ionized Calcium -> ["Calcium"] (parent mineral)
* STRICT EXCLUSIONS:
  - NEVER include panel names or test batteries (e.g., NO "CBC", "LFT", "CMP", "BMP", "Lipid Profile").
  - NEVER include specimen types or body fluids (e.g., NO "Serum", "Plasma", "Blood", "Urine").
  - NEVER include measurement units or properties (e.g., NO "Count", "Level", "Concentration", "Ratio").
  - NEVER include sibling or coordinate tests (e.g., Direct Bilirubin must NOT include "Indirect Bilirubin"; MCV must NOT include "MCH").

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
    "synonym_reasoning": "Explain why you selected these specific strict and lenient synonyms and how you decided to partition them.",
    "strict_synonyms": ["Strict 1", "Strict 2", ...],
    "lenient_synonyms": ["Lenient 1", "Lenient 2", ...]
  }}
]
```
"""

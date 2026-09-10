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
"""Prompts for the SNOMED Vaccine KB generation pipeline."""

PROMPT_SNOMED_VACCINES_PARENT_CHILD_HIERARCHY = """### System Role:
Act as a Senior Clinical Terminologist and Health Informatics Specialist specializing in SNOMED CT medicinal products and Indian immunization systems (ABDM, U-WIN, UIP, NRCeS).

### Task:
You are provided with a table representing the NRCeS NDHM FHIR ValueSet which contains SNOMED-CT codes to represent {expected_snomed_codes_count} vaccines and their preferred display names. This table has been derived from this URL: https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html.
Your task is to expand our table by adding an additional column named '{parent_snomed_code_col_name}'. This column should contain the direct parent SNOMED-CT code for each child SNOMED-CT code.
If a code has more than one direct parent, provide all of them, separated by commas. If the parent code exists outside of the list of provided {expected_snomed_codes_count} codes, provide 'N/A'.

### Evaluation:
At the end, verify your output. Count the rows again to ensure none were lost, and confirm that every row has a numeric parent code filled in. Do NOT modify the preferred display names in any way.

### Input Data:
{input_csv}

### Output CSV:
"""

PROMPT_SNOMED_VACCINES_CLINICAL_VARIATIONS = """### System Role:
Act as a Senior Clinical Terminologist and Health Informatics Specialist specializing in SNOMED CT medicinal products and Indian immunization systems (ABDM, U-WIN, UIP, NRCeS).

### Task:
Your task is to take the table provided and add an additional column named '{clinical_variations_col_name}'. For each SNOMED code in the '{snomed_code_col_name}' column, predict and fill in the most common clinical variation and brand names (e.g., BCG, DPT, Covishield, Covaxin, DTaP, Rota etc.) based on Indian and global clinical contexts. This should include an exhaustive list of clinical varaiations, acroynms and abbreviations for each  SNOMED code's preferred term.
You can reference official resources like the Universal Immunization Programme (UIP) and the Indian Academy of Paediatrics (IAP) Guidelines for this information. Do not add extra terms like "inactivated", "monovalent" etc. unless mentioned in the "Preferred Term (Display)" column. These clinical variations are expected to closely resemble entries in Indian immunization records and global clinical contexts.
Make sure that the list is exhaustive, contains all common variations and Indian brand names, separated by commas.

### Generation Rules for "{clinical_variations_col_name}":
For each input vaccine concept, generate a single comma-separated string adhering to these 4 distinct alias layers:

1. Standard Abbreviations & Public Health Acronyms:
Official CDC, WHO, and Indian Universal Immunization Programme (UIP) abbreviations matching the exact antigen valence (e.g., bOPV, DTwP, MMR, JE).
CRITICAL DISAMBIGUATION RULE: Do NOT assign a single-antigen abbreviation (e.g., 'HepB', 'IPV') to a multi-component combination product (e.g., Hexavalent). Only use the full composite acronym (e.g., 'DTwP-HepB-Hib', 'DTaP-IPV-Hib-HepB').

2. Clinical Shorthand & EHR Slang:
Routine outpatient prescription variations, colloquial provider phrasing, and chart notes frequently used in Indian pediatric and adult clinics (e.g., 'Flu shot', 'Swine flu vax', 'TB vax', 'Chickenpox vaccine', 'Triple antigen', 'Painless DPT', 'Cervical cancer vaccine').

3. Indian & Global Market Brands (Prescription Trade Names):
Prioritize verified Indian domestic market brand names (Serum Institute of India, Bharat Biotech, Biological E, Panacea Biotec, Indian Immunologicals, Sanofi India, GSK India). Examples: Covishield, Rotasiil, Typbar-TCV, Easyfive, Pentavac, Tresivac, bOPV.
Include prominent international trade names approved by FDA/EMA/WHO.
Strictly align formulation brand names with valency/route (e.g., conjugate vs. polysaccharide; live vs. inactivated; wP vs. aP).

### Output Format Requirements:
Return ONLY the final, verified, deduplicated string for each row.
Every term MUST be separated by a comma and a space.
Do NOT use bullet points, nested lists, line breaks, or markdown formatting inside the string.

### Evaluation:
At the end of your response, check your work. Confirm that all {expected_snomed_codes_count} rows are present and that the list of predicted clinical variations are exhaustive and accurate.

### Input Data:
{input_csv}

### Output CSV:
"""

PROMPT_SNOMED_VACCINES_ANTIGENS = """### System Role:
Act as a Senior Clinical Terminologist and Health Informatics Specialist specializing in SNOMED CT medicinal products and Indian immunization systems (ABDM, U-WIN, UIP, NRCeS).

### Task:
You have been provided with a table representing the NRCeS NDHM FHIR ValueSet which contains SNOMED-CT codes to represent {expected_snomed_codes_count} vaccines and their preferred display names. This table has been derived from this URL: https://nrces.in/ndhm/fhir/r4/ValueSet-ndhm-vaccine-codes.html. This table also contains the parent-child hierarchy of the SNOMED-CT codes.
Your task is to take enhance the table provided and add an additional column named "{antigens_col_name}" which contains the antigen constituents of each vaccine.

### Instructions:
Your task is to take the table provided and add an additional column named "{antigens_col_name}".
Rules for populating the "{antigens_col_name}" column:
1. Parse the "Preferred Term (Display)" column to isolate every active antigen, toxoid, or target substance present in the vaccine.
2. Strip out SNOMED-CT administrative and pharmaceutical wrapper phrases (e.g., "Vaccine product containing only", "(medicinal product)", "live attenuated", "inactivated", "recombinant", "conjugated").
3. Standardize each extracted component into standard clinical substance naming formatted as "[Organism / Pathogen] antigen" or "[Organism] toxoid" (e.g., convert "Clostridium tetani toxoid" to "Clostridium tetani toxoid", and "Human rotavirus" to "Human rotavirus antigen").
4. Delimit multiple substances strictly with " + " (space, plus sign, space).
5. For multivalent/serotype vaccines, enumerate the specific serotypes or valencies covered (e.g., "Streptococcus pneumoniae serotypes ").
6. Output all existing columns and rows with this new column integrated into the schema.

### Evaluation:
At the end of your response, check your work. Confirm that all {expected_snomed_codes_count} rows are present and that the list of predicted clinical variations are exhaustive and accurate.

### Input Data:
{input_csv}

### Output CSV:
"""

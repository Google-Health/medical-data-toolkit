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

PROMPT_TEMPLATE = """You are a medical data scientist. For the following LOINC System (Specimen) '{item}', generate approximately 20 clinical synonyms or variations that might appear in a medical record. We need a comprehensive list to build a robust knowledge base.

The synonyms should be realistic variations:
- Full names (e.g., "Serum" or "Plasma" for "Ser/Plas")
- Common abbreviations (e.g., "WB" for "Whole Blood")
- Clinical jargon.
- Common typos or alternative spellings.

Return only a valid JSON list of strings (e.g. ["Serum", "Plasma"]).
"""

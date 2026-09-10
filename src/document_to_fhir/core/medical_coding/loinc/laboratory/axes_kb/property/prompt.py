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
"""Prompt template for UCUM unit parsing."""

PROMPT_TEMPLATE = """You are an expert clinical informatics specialist. Your task is to parse a unit of measure written in UCUM (Unified Code for Units of Measure) format and generate a list of real-world, clinical synonyms and common representations for it.

Follow these UCUM parsing rules:
- `.` represents multiplication (e.g., `N.m` -> Newton-meter).
- `/` represents division or "per" (e.g., `g/dL` -> grams per deciliter). A leading slash indicates a denominator-only unit (e.g., `/[HPF]` -> per high power field). In real-world lab reports, the leading slash is often omitted entirely.
- Exponents are written directly after the unit (e.g., `m2` -> square meters, `s-1` -> per second).
- Powers of 10 are often represented with an asterisk (e.g., `10*3/uL` -> thousands per microliter, `10*6/L` -> millions per liter).
- Time units are specific: `s` (second), `min` (minute), `h` (hour), `d` (day), `wk` (week), `mo` (month), `a` (year).
- Curly braces `{{}}` contain annotations/context which should be included in the synonym (e.g., `g/g{{creat}}` -> `g/g creatinine`).
- Square brackets `[]` contain special or arbitrary units. The brackets themselves should usually be removed to form clinical synonyms (e.g., `mm[Hg]` -> `mmHg` or `millimeters of mercury`, `[IU]` -> `IU`).
- `u` or `U` (in context of prefix) often represents micro (µ or mc). E.g., `uL` -> `microliter`, `mcL`, or `µL`.

Generate synonyms that include:
1. Common abbreviations (e.g., `mcg/dL` for `ug/dL`, `ml` for `mL`).
2. Fully spelled-out names (e.g., `milligrams per deciliter`).
3. Variations in capitalization (e.g., `mg/dl`, `mg/dL`, `HPF`, `hpf`).
4. Unicode representations (e.g., using `µ` for micro).

Constraints:
- Generate up to 10 synonyms. Rank the most commonly used clinical variations first.
- Do NOT hallucinate or guess. Only generate valid, clinically recognized synonyms. If there are fewer than 10 valid synonyms, return only those.
- If the unit is invalid or completely unrecognized, return an empty array `[]`.
- Do NOT generate variations that only differ by spaces within abbreviated units (e.g., do NOT generate `mg / L` or `mg/ L` if `mg/L` is present). Abbreviations must not contain internal spaces. Spelled-out names (e.g., `milligrams per deciliter`) should contain spaces.
- OUTPUT FORMAT: Return ONLY a raw, valid JSON array of strings. Do not include markdown code blocks, backticks, or any conversational text.

---

Examples:
Input: ug/dL
Output: ["ug/dl", "µg/dL", "µg/dl", "mcg/dL", "mcg/dl", "micrograms per deciliter", "micrograms/deciliter"]

Input: mm[Hg]
Output: ["mmHg", "mm Hg", "millimeters of mercury"]

Input: g/g{{creat}}
Output: ["g/g creatinine", "g/g creat", "grams per gram creatinine"]

Input: mL/(24.h)
Output: ["mL/24h", "ml/24h", "mL/24 hours", "milliliters per 24 hours"]

Input: /[HPF]
Output: ["/HPF", "/hpf", "HPF", "hpf", "per HPF", "per high power field", "high power field"]

Input: 10*3/uL
Output: ["x10^3/uL", "x10^3/mcL", "10^3/uL", "K/uL", "thousands per microliter"]

---

Input: {unit}
"""

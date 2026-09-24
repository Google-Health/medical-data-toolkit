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
"""Unit tests for AnalytesIndex two-tier strict vs lenient indexing."""

from absl.testing import absltest
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import index


class AnalytesIndexTest(absltest.TestCase):

  def test_strict_synonym_matching(self):
    records = [
        {
            "LOINC_NUM": "718-7",
            "core_analyte": "Hemoglobin",
            "strict_synonyms": ["Hgb", "Hb", "Haemoglobin"],
            "lenient_synonyms": ["Blood", "CBC"],
            "LONG_COMMON_NAME": "Hemoglobin [Mass/volume] in Blood",
            "COMMON_TEST_RANK": 17,
        },
        {
            "LOINC_NUM": "786-4",
            "core_analyte": "MCHC",
            "strict_synonyms": [
                "MCHC",
                "Mean Corpuscular Hemoglobin Concentration",
            ],
            "lenient_synonyms": ["Hemoglobin", "CBC"],
            "LONG_COMMON_NAME": "MCHC in Red Blood Cells",
            "COMMON_TEST_RANK": 13,
        },
    ]
    idx = index.AnalytesIndex()
    idx.load_data(records)

    # Strict search on HGB should return ONLY 718-7, never 786-4 (MCHC)
    hgb_results = idx.search_by_analyte("HGB")
    self.assertLen(hgb_results, 1)
    self.assertEqual(hgb_results[0].loinc_num, "718-7")

    # Strict search on MCHC should return ONLY 786-4
    mchc_results = idx.search_by_analyte("MCHC")
    self.assertLen(mchc_results, 1)
    self.assertEqual(mchc_results[0].loinc_num, "786-4")

  def test_lenient_synonym_fallback(self):
    records = [
        {
            "LOINC_NUM": "1968-7",
            "core_analyte": "Direct Bilirubin",
            "strict_synonyms": ["Direct Bilirubin", "Conjugated Bilirubin"],
            "lenient_synonyms": ["Bilirubin", "Serum Bilirubin"],
            "LONG_COMMON_NAME": "Bilirubin.direct in Serum",
            "COMMON_TEST_RANK": 131,
        },
        {
            "LOINC_NUM": "1975-2",
            "core_analyte": "Total Bilirubin",
            "strict_synonyms": ["Total Bilirubin", "TBIL"],
            "lenient_synonyms": ["Bilirubin", "Serum Bilirubin"],
            "LONG_COMMON_NAME": "Bilirubin.total in Serum",
            "COMMON_TEST_RANK": 22,
        },
    ]
    idx = index.AnalytesIndex()
    idx.load_data(records)

    # Searching broad parent term 'Bilirubin' falls back to lenient index and
    # returns both.
    bili_results = idx.search_by_analyte("Bilirubin", allow_lenient=True)
    self.assertLen(bili_results, 2)
    loincs = {r.loinc_num for r in bili_results}
    self.assertEqual(loincs, {"1968-7", "1975-2"})

    # When allow_lenient is False, broad term returns empty because strict
    # concepts are specific.
    strict_only = idx.search_by_analyte("Bilirubin", allow_lenient=False)
    self.assertEmpty(strict_only)

  def test_non_transitive_isolation(self):
    # Verify that an obscure row with a lenient term does not cross-pollinate
    # strict lookups.
    records = [
        {
            "LOINC_NUM": "4544-3",
            "core_analyte": "Hematocrit",
            "strict_synonyms": ["Hematocrit", "Hct", "PCV"],
            "lenient_synonyms": ["Erythrocytes"],
            "LONG_COMMON_NAME": "Hematocrit of Blood",
            "COMMON_TEST_RANK": 28,
        },
        {
            "LOINC_NUM": "788-0",
            "core_analyte": "RDW",
            "strict_synonyms": ["RDW", "Red Cell Distribution Width"],
            "lenient_synonyms": ["Erythrocytes"],
            "LONG_COMMON_NAME": "Erythrocyte DistWidth in Blood",
            "COMMON_TEST_RANK": 19,
        },
    ]
    idx = index.AnalytesIndex()
    idx.load_data(records)

    # Searching Hematocrit must NEVER return RDW
    hct_results = idx.search_by_analyte("Hematocrit")
    self.assertLen(hct_results, 1)
    self.assertEqual(hct_results[0].loinc_num, "4544-3")

    # Searching RDW must NEVER return Hematocrit
    rdw_results = idx.search_by_analyte("RDW")
    self.assertLen(rdw_results, 1)
    self.assertEqual(rdw_results[0].loinc_num, "788-0")

  def test_get_all_analytes_with_strict_synonyms(self):
    records = [{
        "LOINC_NUM": "718-7",
        "core_analyte": "Hemoglobin",
        "strict_synonyms": ["Hgb", "Hb"],
        "lenient_synonyms": ["Blood"],
        "LONG_COMMON_NAME": "Hemoglobin in Blood",
        "COMMON_TEST_RANK": 17,
    }]
    idx = index.AnalytesIndex()
    idx.load_data(records)

    self.assertEqual(idx.get_core_analyte("718-7"), "Hemoglobin")
    self.assertEqual(
        idx.get_all_analytes("718-7"),
        {"Hemoglobin", "Hgb", "Hb"},
    )
    self.assertEmpty(idx.get_all_analytes("nonexistent-loinc"))

  def test_strict_synonyms_harmonized_across_same_core_analyte(self):
    records = [
        {
            "LOINC_NUM": "3026-2",
            "core_analyte": "Thyroxine",
            "strict_synonyms": ["T4", "Total Thyroxine"],
            "lenient_synonyms": ["Thyroid Hormone"],
            "LONG_COMMON_NAME": "Thyroxine [Mass/volume] in Serum or Plasma",
            "COMMON_TEST_RANK": 394,
        },
        {
            "LOINC_NUM": "14921-1",
            "core_analyte": "Thyroxine",
            "strict_synonyms": ["T4", "Total T4"],
            "lenient_synonyms": ["Thyroid Hormone"],
            "LONG_COMMON_NAME": "Thyroxine [Moles/volume] in Serum or Plasma",
            "COMMON_TEST_RANK": 14432,
        },
    ]
    idx = index.AnalytesIndex()
    idx.load_data(records)

    results = idx.search_by_analyte("Total T4")
    self.assertEqual({r.loinc_num for r in results}, {"3026-2", "14921-1"})


if __name__ == "__main__":
  absltest.main()

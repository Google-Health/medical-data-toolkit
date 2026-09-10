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
"""Unit tests for snomed_vaccine_query_engine."""

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine.index import snomed_vaccine_index
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import base
from src.document_to_fhir.core.medical_coding.snomed.vaccine.query import snomed_vaccine_query_engine


class SnomedVaccineQueryEngineTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    mock_df = pd.DataFrame({
        "preferred_display_term": ["Vaccine_A", "Vaccine_B"],
        "clinical_variations": [
            "Vaccine_A",
            "Vaccine_B",
        ],
        "antigens": ["Vaccine_A", "Vaccine_B"],
        "patient_type": ["UNKNOWN", "UNKNOWN"],
        "snomed_code": ["16627741", "11945993821"],
        "parent_snomed_code": [None, None],
    })
    self.vaccine_index = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(mock_df)
    )
    self.engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=self.vaccine_index
    )

  def test_query_placeholder_returns_correct_result(self):
    imm = resources.Immunization(
        vaccine_name="Vaccine_A",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = self.engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "16627741")
    self.assertEqual(result.display_name, "Vaccine_A")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_engine_is_instance_of_base_vaccine_query_engine(self):
    self.assertIsInstance(self.engine, base.BaseVaccineQueryEngine)

  def test_query_adult_patient_match(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Hepatitis A adult vaccine",
            "Hepatitis A paediatric vaccine",
        ],
        "clinical_variations": [
            "Hepatitis A",
            "Hepatitis A",
        ],
        "antigens": ["Hepatitis A", "Hepatitis A"],
        "patient_type": ["ADULT", "PAEDIATRIC"],
        "snomed_code": ["1001", "1002"],
        "parent_snomed_code": [None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Hepatitis A",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "1001")
    self.assertEqual(result.display_name, "Hepatitis A adult vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_paediatric_patient_match(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Hepatitis A adult vaccine",
            "Hepatitis A paediatric vaccine",
        ],
        "clinical_variations": [
            "Hepatitis A",
            "Hepatitis A",
        ],
        "antigens": ["Hepatitis A", "Hepatitis A"],
        "patient_type": ["ADULT", "PAEDIATRIC"],
        "snomed_code": ["1001", "1002"],
        "parent_snomed_code": [None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Hepatitis A",
        dosage="1",
        patient_type=resources.PatientType.PAEDIATRIC,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "1002")
    self.assertEqual(result.display_name, "Hepatitis A paediatric vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_unknown_patient_general_match(self):
    df = pd.DataFrame({
        "preferred_display_term": ["BCG vaccine"],
        "clinical_variations": ["BCG"],
        "antigens": ["BCG"],
        "patient_type": ["UNKNOWN"],
        "snomed_code": ["3001"],
        "parent_snomed_code": [None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="BCG",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "3001")
    self.assertEqual(result.display_name, "BCG vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_fallback_to_parent_when_multiple_matches(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Covaxin child product",
            "Covaxin parent product",
        ],
        "clinical_variations": [
            "Covaxin",
            "Covaxin",
        ],
        "antigens": ["Covaxin", "Covaxin"],
        "patient_type": ["ADULT", "ADULT"],
        "snomed_code": ["2001", "2002"],
        "parent_snomed_code": ["2002", "836368004"],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Covaxin",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "2002")
    self.assertEqual(result.display_name, "Covaxin parent product")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_fallback_to_parent_stops_at_excluded_codes(self):
    for excluded_code in ["836368004", "836369007", "863950005"]:
      df = pd.DataFrame({
          "preferred_display_term": [
              "Sub vaccine product",
              "Top valid vaccine product",
          ],
          "clinical_variations": [
              "Sub Vaccine",
              "Sub Vaccine",
          ],
          "antigens": ["Sub Vaccine", "Sub Vaccine"],
          "patient_type": ["ADULT", "ADULT"],
          "snomed_code": ["5001", "5002"],
          "parent_snomed_code": ["5002", excluded_code],
      })
      idx = snomed_vaccine_index.SnomedVaccineIndex(
          snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
      )
      engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
          vaccine_index=idx
      )
      imm = resources.Immunization(
          vaccine_name="Sub Vaccine",
          dosage="1",
          patient_type=resources.PatientType.ADULT,
      )
      result = engine.query(imm)
      self.assertEqual(result.code, "5002")
      self.assertEqual(result.display_name, "Top valid vaccine product")
      self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_no_match_returns_empty_result(self):
    imm = resources.Immunization(
        vaccine_name="NonExistentVaccine",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = self.engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_empty_vaccine_name_returns_empty_result(self):
    imm = resources.Immunization(
        vaccine_name="",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = self.engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_with_none_vaccine_index(self):
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=None
    )
    imm = resources.Immunization(
        vaccine_name="Vaccine_A",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_with_empty_master_df(self):
    idx = snomed_vaccine_index.SnomedVaccineIndex(pd.DataFrame())
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Vaccine_A",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_with_missing_required_columns(self):
    df = pd.DataFrame({
        "OnlySomeColumn": ["Val"],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Vaccine_A",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_normalized_variation_matching(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "DTaP-IPV vaccine",
        ],
        "clinical_variations": [
            "DTaP-IPV (Polio, Diphtheria, Pertussis, Tetanus)",
        ],
        "antigens": ["DTaP-IPV"],
        "patient_type": ["UNKNOWN"],
        "snomed_code": ["99991111"],
        "parent_snomed_code": [None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="DTaP-IPV",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "99991111")
    self.assertEqual(result.display_name, "DTaP-IPV vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_with_vaccine_suffix_in_name(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Polio vaccine",
        ],
        "clinical_variations": [
            "Polio",
        ],
        "antigens": ["Polio"],
        "patient_type": ["UNKNOWN"],
        "snomed_code": ["88881111"],
        "parent_snomed_code": [None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(
        snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    )
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Polio Vaccine",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "88881111")
    self.assertEqual(result.display_name, "Polio vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_sorts_by_matching_most_terms(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Measles vaccine",
            "Measles and Mumps vaccine",
            "MMR vaccine",
        ],
        "norm_clinical_variations": [
            "measles",
            "measles, mumps",
            "measles, mumps, rubella",
        ],
        "patient_type": [
            resources.PatientType.UNKNOWN,
            resources.PatientType.UNKNOWN,
            resources.PatientType.UNKNOWN,
        ],
        "snomed_code": ["101", "102", "103"],
        "parent_snomed_code": [None, None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Measles, Mumps, Rubella",
        dosage="1",
        patient_type=resources.PatientType.UNKNOWN,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "103")
    self.assertEqual(result.display_name, "MMR vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_adult_patient_match_with_preprocessed_dataframe(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Hepatitis A adult vaccine",
            "Hepatitis A paediatric vaccine",
        ],
        "norm_clinical_variations": [
            "hepatitisa",
            "hepatitisa",
        ],
        "patient_type": [
            resources.PatientType.ADULT,
            resources.PatientType.PAEDIATRIC,
        ],
        "snomed_code": ["1001", "1002"],
        "parent_snomed_code": [None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Hepatitis A",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "1001")
    self.assertEqual(result.display_name, "Hepatitis A adult vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_paediatric_patient_match_with_preprocessed_dataframe(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Hepatitis A adult vaccine",
            "Hepatitis A paediatric vaccine",
        ],
        "norm_clinical_variations": [
            "hepatitisa",
            "hepatitisa",
        ],
        "patient_type": [
            resources.PatientType.ADULT,
            resources.PatientType.PAEDIATRIC,
        ],
        "snomed_code": ["1001", "1002"],
        "parent_snomed_code": [None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Hepatitis A",
        dosage="1",
        patient_type=resources.PatientType.PAEDIATRIC,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "1002")
    self.assertEqual(result.display_name, "Hepatitis A paediatric vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_fallback_to_parent_when_multiple_matches_preprocessed_df(
      self,
  ):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Covaxin child product",
            "Covaxin parent product",
        ],
        "norm_clinical_variations": [
            "covaxin",
            "covaxin",
        ],
        "patient_type": [
            resources.PatientType.ADULT,
            resources.PatientType.ADULT,
        ],
        "snomed_code": ["2001", "2002"],
        "parent_snomed_code": ["2002", "836368004"],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="Covaxin",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "2002")
    self.assertEqual(result.display_name, "Covaxin parent product")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_multi_term_no_match_returns_empty_result(self):
    imm = resources.Immunization(
        vaccine_name="NonExistent1, NonExistent2",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = self.engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertIsNone(result.code)
    self.assertIsNone(result.display_name)
    self.assertIsNone(result.code_system)

  def test_query_fallback_to_general_when_no_adult_specific_match(self):
    df = pd.DataFrame({
        "preferred_display_term": ["BCG vaccine"],
        "norm_clinical_variations": ["bcg"],
        "patient_type": [resources.PatientType.UNKNOWN],
        "snomed_code": ["3001"],
        "parent_snomed_code": [None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="BCG",
        dosage="1",
        patient_type=resources.PatientType.ADULT,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "3001")
    self.assertEqual(result.display_name, "BCG vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_query_fallback_to_general_when_no_paediatric_specific_match(self):
    df = pd.DataFrame({
        "preferred_display_term": ["BCG vaccine"],
        "norm_clinical_variations": ["bcg"],
        "patient_type": [resources.PatientType.UNKNOWN],
        "snomed_code": ["3001"],
        "parent_snomed_code": [None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    imm = resources.Immunization(
        vaccine_name="BCG",
        dosage="1",
        patient_type=resources.PatientType.PAEDIATRIC,
    )
    result = engine.query(imm)
    self.assertIsInstance(result, base.VaccineResult)
    self.assertEqual(result.code, "3001")
    self.assertEqual(result.display_name, "BCG vaccine")
    self.assertEqual(result.code_system, "SNOMED-CT")

  def test_build_term_pattern(self):
    pattern = self.engine._build_term_pattern("Hib")
    self.assertRegex("Hib", pattern)
    self.assertRegex("DTaP, Hib", pattern)
    self.assertRegex("Hib, HepB", pattern)
    self.assertNotRegex("DTaP-Hib", pattern)

  def test_find_matches_for_terms(self):
    df = pd.DataFrame({
        "preferred_display_term": [
            "Measles vaccine",
            "MMR vaccine",
        ],
        "norm_clinical_variations": [
            "measles",
            "measles, mumps, rubella",
        ],
        "patient_type": [
            resources.PatientType.UNKNOWN,
            resources.PatientType.UNKNOWN,
        ],
        "snomed_code": ["101", "103"],
        "parent_snomed_code": [None, None],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    matches = engine._find_matches_for_terms(["measles", "mumps", "rubella"])
    self.assertFalse(matches.empty)
    self.assertEqual(matches.iloc[0]["snomed_code"], "103")

  def test_find_matches_for_terms_does_not_mutate_master_df(self):
    df = pd.DataFrame({
        "preferred_display_term": ["Vaccine A"],
        "norm_clinical_variations": ["vaccine a"],
        "patient_type": [resources.PatientType.UNKNOWN],
        "snomed_code": ["1001"],
        "parent_snomed_code": [None],
    })
    original_columns = list(df.columns)
    idx = snomed_vaccine_index.SnomedVaccineIndex(df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    _ = engine._find_matches_for_terms(["vaccine a"])
    self.assertEqual(list(idx.master_df.columns), original_columns)
    self.assertNotIn("_match_count", idx.master_df.columns)

  def test_resolve_candidate_matches_missing_parent_code_in_master_df(self):
    master_df = pd.DataFrame({
        "preferred_display_term": ["Candidate 1"],
        "norm_clinical_variations": ["cand"],
        "patient_type": [resources.PatientType.UNKNOWN],
        "snomed_code": ["9001"],
        "parent_snomed_code": ["9999"],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(master_df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    candidates = pd.DataFrame({
        "preferred_display_term": ["Candidate 1", "Candidate 2"],
        "snomed_code": ["9001", "9002"],
        # parent code 9999 does not exist as a snomed_code in master_df
        "parent_snomed_code": ["9999", "9999"],
    })
    result = engine._resolve_candidate_matches(candidates)
    self.assertIsNone(result)

  def test_resolve_candidate_matches_multi_parent_child_not_in_master_df(self):
    master_df = pd.DataFrame({
        "preferred_display_term": ["Other Vaccine"],
        "norm_clinical_variations": ["other"],
        "patient_type": [resources.PatientType.UNKNOWN],
        "snomed_code": ["8001"],
        "parent_snomed_code": ["3333"],
    })
    idx = snomed_vaccine_index.SnomedVaccineIndex(master_df)
    engine = snomed_vaccine_query_engine.SnomedVaccineQueryEngine(
        vaccine_index=idx
    )
    candidates = pd.DataFrame({
        "preferred_display_term": ["Cand 1", "Cand 2"],
        "snomed_code": ["9001", "9002"],
        # multiple parents whose combination doesn't exist as parent_snomed_code
        # in master_df
        "parent_snomed_code": ["1111, 2222", "1111, 2222"],
    })
    result = engine._resolve_candidate_matches(candidates)
    self.assertIsNone(result)


if __name__ == "__main__":
  absltest.main()

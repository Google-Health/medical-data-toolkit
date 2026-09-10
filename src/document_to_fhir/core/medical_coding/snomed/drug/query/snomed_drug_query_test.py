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
"""Tests for snomed_drug_query."""

from unittest import mock

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import snomed_drug_query


class ConcreteSnomedDrugQuery(snomed_drug_query.SnomedDrugQuery):

  def query(self, medication, **kwargs):
    pass


class SnomedDrugQueryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.create_autospec(base.BaseDrugQueryEngine)
    self.query_impl = ConcreteSnomedDrugQuery(self.mock_engine)

  def test_parse_field_empty(self):
    self.assertEqual(self.query_impl.parse_field(None), [])
    self.assertEqual(self.query_impl.parse_field(""), [])

  def test_parse_field_split(self):
    self.assertEqual(self.query_impl.parse_field("A + B"), ["A", "B"])
    self.assertEqual(self.query_impl.parse_field("A | B"), ["A", "B"])
    self.assertEqual(self.query_impl.parse_field("A , B"), ["A", "B"])
    self.assertEqual(self.query_impl.parse_field("A\tB"), ["A", "B"])
    self.assertEqual(self.query_impl.parse_field("A\nB"), ["A", "B"])
    self.assertEqual(self.query_impl.parse_field("  A  +  B  "), ["A", "B"])

  def test_match_dose_form_empty(self):
    df = pd.DataFrame({"Form": ["tablet", "capsule"]})
    result = self.query_impl.match_dose_form(df, None, "Form")
    pd.testing.assert_frame_equal(result, df)

  def test_match_dose_form_exact(self):
    df = pd.DataFrame({"Form": ["tablet", "capsule"]})
    result = self.query_impl.match_dose_form(df, "tablet", "Form")
    expected = pd.DataFrame({"Form": ["tablet"]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

  def test_match_dose_form_fuzzy(self):
    df = pd.DataFrame({"Form": ["tablet", "capsule"]})
    # "tablets" should match "tablet" fuzzy (cutoff 0.6)
    result = self.query_impl.match_dose_form(df, "tablets", "Form")
    expected = pd.DataFrame({"Form": ["tablet"]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

  def test_match_dose_form_no_match(self):
    df = pd.DataFrame({"Form": ["tablet", "capsule"]})
    # "injection" should not match and return original df
    result = self.query_impl.match_dose_form(df, "injection", "Form")
    pd.testing.assert_frame_equal(result, df)

  def test_match_route_empty(self):
    df = pd.DataFrame({"Route": ["oral", "intravenous"]})
    result = self.query_impl.match_route(df, None, "Route")
    pd.testing.assert_frame_equal(result, df)

  def test_match_route_exact(self):
    df = pd.DataFrame({"Route": ["oral", "intravenous"]})
    result = self.query_impl.match_route(df, "oral", "Route")
    expected = pd.DataFrame({"Route": ["oral"]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

  def test_match_route_fuzzy(self):
    df = pd.DataFrame({"Route": ["oral", "intravenous"]})
    result = self.query_impl.match_route(df, "orally", "Route")
    expected = pd.DataFrame({"Route": ["oral"]})
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)

  def test_match_route_no_match(self):
    df = pd.DataFrame({"Route": ["oral", "intravenous"]})
    result = self.query_impl.match_route(df, "topical", "Route")
    pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
  absltest.main()

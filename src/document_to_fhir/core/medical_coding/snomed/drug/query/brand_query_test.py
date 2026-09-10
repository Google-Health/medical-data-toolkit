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
"""Tests for brand_query."""

from unittest import mock

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.drug import config
from src.document_to_fhir.core.medical_coding.snomed.drug.index import brand_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import form_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import product_index
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import brand_query


class BrandQueryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.create_autospec(base.BaseDrugQueryEngine)
    self.mock_brand_index = mock.create_autospec(brand_index.BrandIndex)
    self.mock_product_index = mock.create_autospec(product_index.ProductIndex)
    self.mock_form_index = mock.create_autospec(form_index.FormIndex)

    # Default empty groups
    self.mock_brand_index.groups = {}
    self.mock_product_index.groups = {}

  def _init_engine(self):
    return brand_query.SnomedBrandDrugQueryEngine(
        self.mock_engine,
        self.mock_brand_index,
        self.mock_product_index,
        self.mock_form_index,
    )

  def test_query_empty_brand_and_substance(self):
    engine = self._init_engine()
    med = resources.Medication(brand_name="", active_substances="")
    code, name = engine.query(med, brand_matching_cutoff=0.85)
    self.assertIsNone(code)
    self.assertIsNone(name)

  def test_query_generic_like_brand_name(self):
    engine = self._init_engine()
    # Brand names with '|', '/', '+' are skipped as they are likely generic
    # combinations.
    for name in ["A|B", "A/B", "A+B"]:
      med = resources.Medication(brand_name=name)
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertIsNone(code)
      self.assertIsNone(display)

  def test_query_exact_match_no_brand_candidates(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    self.mock_brand_index.groups = {}

    engine = self._init_engine()
    med = resources.Medication(brand_name="Aspirin")
    code, display = engine.query(med, brand_matching_cutoff=0.85)
    self.assertIsNone(code)
    self.assertIsNone(display)

  def test_query_exact_match_single_brand_candidate_no_filters(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99999"],
        "Brand Name": ["Aspirin 100mg tablet"],
        "Product Identifier": ["12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()
      med = resources.Medication(brand_name="Aspirin")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99999")
      self.assertEqual(display, "Aspirin 100mg tablet")

  def test_query_exact_match_strength_filtering(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99991", "99992"],
        "Brand Name": ["Aspirin 100mg tablet", "Aspirin 500mg tablet"],
        "Product Identifier": ["12345", "12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", True
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):

      engine = self._init_engine()

      # Query for 100mg
      med = resources.Medication(brand_name="Aspirin", strength="100mg")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99991")
      self.assertEqual(display, "Aspirin 100mg tablet")

      # Query for 500mg
      med = resources.Medication(brand_name="Aspirin", strength="500mg")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99992")
      self.assertEqual(display, "Aspirin 500mg tablet")

      # Query for 200mg (no match)
      med = resources.Medication(brand_name="Aspirin", strength="200mg")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertIsNone(code)
      self.assertIsNone(display)

      # Query without strength being provided
      med = resources.Medication(brand_name="Aspirin")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertIsNone(code)
      self.assertIsNone(display)

  def test_query_strength_unit_conversion(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99991"],
        "Brand Name": ["Aspirin 100mg tablet"],
        "Product Identifier": ["12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", True
    ), mock.patch.object(
        config, "BRAND_MATCHING_ENABLE_UNIT_CONVERSION", True
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):

      engine = self._init_engine()

      # Query for 0.1g, should match 100mg
      med = resources.Medication(brand_name="Aspirin", strength="0.1g")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99991")
      self.assertEqual(display, "Aspirin 100mg tablet")

  def test_query_dose_form_filtering(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99991", "99992"],
        "Brand Name": ["Aspirin 100mg tablet", "Aspirin 100mg injection"],
        "Product Identifier": ["12345", "12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", True
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):

      engine = self._init_engine()

      # Query for tablet
      med = resources.Medication(brand_name="Aspirin", medicine_form="tablet")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99991")
      self.assertEqual(display, "Aspirin 100mg tablet")

      # Query for injection
      med = resources.Medication(
          brand_name="Aspirin", medicine_form="injection"
      )
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99992")
      self.assertEqual(display, "Aspirin 100mg injection")

  def test_query_route_filtering(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99991", "99992"],
        "Brand Name": [
            "Aspirin 100mg oral tablet",
            "Aspirin 100mg rectal suppository",
        ],
        "Product Identifier": ["12345", "12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", True
    ):

      engine = self._init_engine()

      # Query for oral
      med = resources.Medication(brand_name="Aspirin", route="oral")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99991")
      self.assertEqual(display, "Aspirin 100mg oral tablet")

  def test_query_multiple_candidates_remaining(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99991", "99992"],
        "Brand Name": ["Aspirin 100mg tablet", "Aspirin 100mg tablet"],
        "Product Identifier": ["12345", "12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):

      engine = self._init_engine()
      med = resources.Medication(brand_name="Aspirin")
      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertIsNone(code)
      self.assertIsNone(display)

  def test_query_fuzzy_brand_match(self):
    self.mock_product_index.groups = {"aspirin": "12345"}
    brand_df = pd.DataFrame({
        "Identifier": ["99999"],
        "Brand Name": ["Aspirin 100mg tablet"],
        "Product Identifier": ["12345"],
    })
    self.mock_brand_index.groups = {"12345": brand_df}

    with mock.patch.object(
        config, "BRAND_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "BRAND_MATCHING_ROUTE", False
    ):

      engine = self._init_engine()

      # Query for "Aspirinn" (typo)
      med = resources.Medication(brand_name="Aspirinn")

      code, display = engine.query(med, brand_matching_cutoff=0.85)
      self.assertEqual(code, "99999")
      self.assertEqual(display, "Aspirin 100mg tablet")

      # Query with very high cutoff, should fail
      code, display = engine.query(med, brand_matching_cutoff=0.99)
      self.assertIsNone(code)
      self.assertIsNone(display)


if __name__ == "__main__":
  absltest.main()

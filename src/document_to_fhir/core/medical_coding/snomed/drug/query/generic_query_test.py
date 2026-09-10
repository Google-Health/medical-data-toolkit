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
"""Tests for generic_query."""

from unittest import mock

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.drug import config
from src.document_to_fhir.core.medical_coding.snomed.drug.index import form_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import generic_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import route_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_variations_index
from src.document_to_fhir.core.medical_coding.snomed.drug.query import base
from src.document_to_fhir.core.medical_coding.snomed.drug.query import generic_query


class GenericQueryTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_engine = mock.create_autospec(base.BaseDrugQueryEngine)
    self.mock_generic_index = mock.create_autospec(generic_index.GenericIndex)
    self.mock_substance_index = mock.create_autospec(
        substance_index.SubstanceIndex
    )
    self.mock_substance_variations_index = mock.create_autospec(
        substance_variations_index.SubstanceVariationsIndex
    )
    self.mock_form_index = mock.create_autospec(form_index.FormIndex)
    self.mock_route_index = mock.create_autospec(route_index.RouteIndex)

    # Default empty groups/dfs
    self.mock_generic_index.groups = {}
    self.mock_substance_index.groups = {}
    self.mock_substance_variations_index.master_df = pd.DataFrame(
        columns=["variations"]
    )
    self.mock_route_index.master_df = pd.DataFrame()

  def _init_engine(self):
    return generic_query.SnomedGenericDrugQueryEngine(
        self.mock_engine,
        self.mock_generic_index,
        self.mock_substance_index,
        self.mock_substance_variations_index,
        self.mock_form_index,
        self.mock_route_index,
    )

  def test_query_no_substances(self):
    engine = self._init_engine()
    med = resources.Medication(active_substances="", brand_name="")
    code, name = engine.query(med, generic_matching_cutoff=0.6)
    self.assertIsNone(code)
    self.assertIsNone(name)

  def test_query_single_substance_exact_match(self):
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["222"],
        "Generic Name": ["Paracetamol 500mg tablet"],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()
      med = resources.Medication(active_substances="Paracetamol")
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg tablet")

  def test_query_multiple_substances_exact_match(self):
    self.mock_substance_index.groups = {
        "paracetamol": "111",
        "ibuprofen": "222",
    }
    generic_df = pd.DataFrame({
        "Identifier": ["333"],
        "Generic Name": ["Paracetamol + Ibuprofen tablet"],
    })
    self.mock_generic_index.groups = {"111+222": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()
      med = resources.Medication(active_substances="Ibuprofen + Paracetamol")
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "333")
      self.assertEqual(display, "Paracetamol + Ibuprofen tablet")

  def test_query_substance_variations(self):
    self.mock_substance_index.groups = {"metformin hydrochloride": "111"}
    self.mock_substance_variations_index.master_df = pd.DataFrame(
        {"variations": [["metformin hydrochloride"]]}, index=["metformin"]
    )
    generic_df = pd.DataFrame({
        "Identifier": ["222"],
        "Generic Name": ["Metformin hydrochloride 500mg tablet"],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_SUPPORT_ALTERNATE_SUBSTANCES", True
    ), mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()
      med = resources.Medication(active_substances="Metformin")
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Metformin hydrochloride 500mg tablet")

  def test_query_strength_matching(self):
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["221", "222"],
        "Generic Name": [
            "Paracetamol 250mg tablet",
            "Paracetamol 500mg tablet",
        ],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", True
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()

      med = resources.Medication(
          active_substances="Paracetamol", strength="500mg"
      )
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg tablet")

  def test_query_strength_unit_conversion(self):
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["222"],
        "Generic Name": ["Paracetamol 500mg tablet"],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", True
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ENABLE_UNIT_CONVERSION", True
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()

      # 0.5g should match 500mg
      med = resources.Medication(
          active_substances="Paracetamol", strength="0.5g"
      )
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg tablet")

  def test_query_dose_form_matching(self):
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["221", "222"],
        "Generic Name": [
            "Paracetamol 500mg tablet",
            "Paracetamol 500mg infusion",
        ],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", True
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()

      med = resources.Medication(
          active_substances="Paracetamol", medicine_form="infusion"
      )
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg infusion")

  def test_query_route_matching(self):
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["221", "222"],
        "Generic Name": [
            "Paracetamol 500mg oral tablet",
            "Paracetamol 500mg rectal suppository",
        ],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", True
    ):
      engine = self._init_engine()

      med = resources.Medication(
          active_substances="Paracetamol", route="rectal"
      )
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg rectal suppository")

  def test_query_use_brand_name_as_substance(self):
    # Active substances is empty, but brand name contains paracetamol
    self.mock_substance_index.groups = {"paracetamol": "111"}
    generic_df = pd.DataFrame({
        "Identifier": ["222"],
        "Generic Name": ["Paracetamol 500mg tablet"],
    })
    self.mock_generic_index.groups = {"111": generic_df}

    with mock.patch.object(
        config, "GENERIC_MATCHING_STRENGTH", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_DOSE_FORM", False
    ), mock.patch.object(
        config, "GENERIC_MATCHING_ROUTE", False
    ):
      engine = self._init_engine()
      med = resources.Medication(active_substances="", brand_name="Paracetamol")
      code, display = engine.query(med, generic_matching_cutoff=0.6)
      self.assertEqual(code, "222")
      self.assertEqual(display, "Paracetamol 500mg tablet")


if __name__ == "__main__":
  absltest.main()

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
"""Tests for SNOMED drug substance variations builder."""

import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug.axes_kb.substance_variations import builder


class SubstanceVariationsBuilderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["101"],
        "Generic Name": ["Amoxicillin 500mg"],
    })
    self.kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )

  @parameterized.named_parameters(
      (
          "match",
          "Amoxicillin (as amoxicillin sodium) + Clavulanic acid",
          "amoxicillin",
          "amoxicillin sodium",
      ),
      (
          "no_match",
          "Paracetamol 500mg tablet",
          "amoxicillin",
          None,
      ),
      (
          "empty_text",
          "",
          "amoxicillin",
          None,
      ),
      (
          "empty_active_substance",
          "Amoxicillin (as amoxicillin sodium) + Clavulanic acid",
          "",
          None,
      ),
      (
          "none_text",
          None,
          "amoxicillin",
          None,
      ),
      (
          "none_active_substance",
          "Amoxicillin (as amoxicillin sodium) + Clavulanic acid",
          None,
          None,
      ),
      (
          "different_substance",
          "Amoxicillin (as amoxicillin sodium) + Clavulanic acid",
          "paracetamol",
          None,
      ),
  )
  def test_find_substance_variant_in_combination_drug(
      self, text, active_substance, expected
  ):
    result = self.kb_builder._find_substance_variant_in_combination_drug(
        text, active_substance
    )
    self.assertEqual(result, expected)

  @parameterized.named_parameters(
      (
          "missing_substance_name",
          pd.DataFrame({"Identifier": ["101"]}),
          pd.DataFrame({
              "Substance Identifier": ["101"],
              "Generic Name": ["Amoxicillin 500mg"],
          }),
          "Column 'Substance Name' not found in substance_df",
      ),
      (
          "missing_generic_name",
          pd.DataFrame({
              "Identifier": ["101"],
              "Substance Name": ["Amoxicillin"],
          }),
          pd.DataFrame({"Substance Identifier": ["101"]}),
          "Column 'Generic Name' not found in generic_df",
      ),
  )
  def test_init_raises_value_error_missing_required_columns(
      self, substance_df, generic_df, expected_regex
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      builder.SubstanceVariationsKBBuilder(
          substance_df=substance_df,
          generic_df=generic_df,
      )

  def test_init_with_pre_normalized_dataframes(self):
    substance_df = pd.DataFrame(
        {"norm_name": ["amoxicillin"]},
        index=["101"],
    )
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102"],
        "norm_name": ["amoxicillin (as amoxicillin sodium) 500mg"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    self.assertIn("norm_name", kb_builder._substance_df.columns)
    self.assertIn("norm_name", kb_builder._generic_df.columns)

  def test_build_kb_clusters_and_returns_index(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101", "102", "103"],
        "Substance Name": [
            "Amoxicillin",
            "Amoxicillin sodium",
            "Paracetamol",
        ],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102", "103"],
        "Generic Name": [
            "Amoxicillin (as amoxicillin sodium) 500mg capsule",
            "Paracetamol 500mg tablet",
        ],
    })

    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )

    temp_dir = self.create_tempdir()
    save_txt_path = os.path.join(temp_dir.full_path, "test_substance_vars.txt")

    df = kb_builder.build_kb(save_txt_path=save_txt_path)

    self.assertIsInstance(
        df, pd.DataFrame
    )
    self.assertTrue(os.path.exists(save_txt_path))

    saved_df = pd.read_csv(save_txt_path, sep="\t")
    self.assertIn("Substance Name", saved_df.columns)
    self.assertIn("variations", saved_df.columns)

    # Check index master_df mappings
    self.assertIn("amoxicillin", df.index)
    self.assertIn("amoxicillin sodium", df.index)
    self.assertEqual(
        df.loc["amoxicillin", "variations"], "amoxicillin sodium"
    )
    self.assertEqual(
        df.loc["amoxicillin sodium", "variations"], "amoxicillin"
    )

  def test_build_kb_combination_drug_fallback(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101", "102"],
        "Substance Name": ["Amoxicillin", "Clavulanic acid"],
    })
    # Generic entry with combination substance identifier '+'
    generic_df = pd.DataFrame({
        "Substance Identifier": ["999+102", "998+102"],
        "Generic Name": [
            "Amoxicillin (as amoxicillin sodium) + Clavulanic acid 625mg",
            "Amoxicillin (as incomplete + Clavulanic acid 500mg",
        ],
    })

    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )

    df = kb_builder.build_kb()
    self.assertIn("amoxicillin", df.index)
    self.assertIn(
        "amoxicillin sodium", df.loc["amoxicillin", "variations"]
    )

  def test_build_kb_with_nan_substance_name_continues(self):
    substance_df = pd.DataFrame({
        "Identifier": ["100", "101"],
        "norm_name": [float("nan"), "amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102"],
        "norm_name": ["amoxicillin (as amoxicillin sodium) 500mg"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    df = kb_builder.build_kb()
    self.assertIn("amoxicillin", df.index)

  def test_build_kb_no_clusters_generated(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["999"],
        "Generic Name": ["Paracetamol 500mg tablet"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    df = kb_builder.build_kb()
    self.assertIsInstance(
        df, pd.DataFrame
    )
    self.assertTrue(df.empty)

  def test_build_kb_save_csv_exception_handled(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101", "102"],
        "Substance Name": ["Amoxicillin", "Amoxicillin sodium"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102"],
        "Generic Name": ["Amoxicillin (as amoxicillin sodium) 500mg capsule"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    with mock.patch.object(
        builder.utils, "open_file", side_effect=IOError("Mock write error")
    ):
      df = kb_builder.build_kb(save_txt_path="/fake/path.txt")
    self.assertIsInstance(
        df, pd.DataFrame
    )

  def test_build_kb_multiple_variations_bidirectional_clustering(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101", "102", "103"],
        "Substance Name": [
            "Amoxicillin",
            "Amoxicillin sodium",
            "Amoxicillin trihydrate",
        ],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102", "103", "101"],
        "Generic Name": [
            "Amoxicillin (as amoxicillin sodium) 500mg capsule",
            "Amoxicillin (as amoxicillin trihydrate) 250mg capsule",
            "Amoxicillin sodium (as amoxicillin) 500mg injection",
        ],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    df = kb_builder.build_kb()
    self.assertIn("amoxicillin", df.index)
    self.assertIn("amoxicillin sodium", df.index)
    self.assertIn("amoxicillin trihydrate", df.index)

  def test_find_generic_matches_for_substance(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["101", "102"],
        "Generic Name": [
            "Amoxicillin (as amoxicillin) 250mg",
            "Amoxicillin (as amoxicillin sodium) 500mg capsule",
        ],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    matches = kb_builder._find_generic_matches_for_substance(
        "amoxicillin", "101"
    )
    self.assertLen(matches, 1)
    self.assertEqual(matches.iloc[0]["Substance Identifier"], "102")

  def test_extract_variants_from_combination_drugs(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["999+102"],
        "norm_name": [
            "amoxicillin (as amoxicillin sodium) + clavulanic acid 625mg"
        ],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    variants = kb_builder._extract_variants_from_combination_drugs(
        generic_df, "amoxicillin"
    )
    self.assertEqual(variants, ["amoxicillin sodium"])

  def test_add_to_clusters_bidirectional(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["101"],
        "Generic Name": ["Amoxicillin 500mg"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    clusters = {}
    kb_builder._add_to_clusters(clusters, "substance_a", ["variant_b"])
    self.assertIn("variant_b", clusters["substance_a"]["variations"])
    self.assertIn("substance_a", clusters["variant_b"]["variations"])

  def test_create_clusters_dataframe(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101"],
        "Substance Name": ["Amoxicillin"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["101"],
        "Generic Name": ["Amoxicillin 500mg"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    clusters = {
        "amoxicillin": {
            "variations": ["amoxicillin sodium", "amoxicillin sodium"]
        },
    }
    df = kb_builder._create_clusters_dataframe(clusters)
    self.assertEqual(df.index.name, "Substance Name")
    # Duplicate variations should be deduplicated
    self.assertEqual(df.loc["amoxicillin", "variations"], "amoxicillin sodium")

  def test_build_kb_save_txt_path(self):
    substance_df = pd.DataFrame({
        "Identifier": ["101", "102"],
        "Substance Name": ["Amoxicillin", "Amoxicillin sodium"],
    })
    generic_df = pd.DataFrame({
        "Substance Identifier": ["102"],
        "Generic Name": ["Amoxicillin (as amoxicillin sodium) 500mg"],
    })
    kb_builder = builder.SubstanceVariationsKBBuilder(
        substance_df=substance_df,
        generic_df=generic_df,
    )
    temp_dir = self.create_tempdir()
    save_txt_path = os.path.join(temp_dir.full_path, "test_substance_vars.txt")
    kb_builder.build_kb(save_txt_path=save_txt_path)
    self.assertTrue(os.path.exists(save_txt_path))
    saved_df = pd.read_csv(save_txt_path, sep="\t")
    self.assertIn("Substance Name", saved_df.columns)
    self.assertIn("variations", saved_df.columns)


if __name__ == "__main__":
  absltest.main()

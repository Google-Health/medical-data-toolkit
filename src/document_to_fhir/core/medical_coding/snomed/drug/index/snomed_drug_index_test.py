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
"""Tests for snomed_drug_index and its subclasses."""

from unittest import mock

from absl import flags
from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.core.medical_coding.snomed.drug.index import brand_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import form_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import generic_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import product_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import route_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import snomed_drug_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_index
from src.document_to_fhir.core.medical_coding.snomed.drug.index import substance_variations_index

flags.FLAGS.mark_as_parsed()


class SnomedDrugIndexTest(absltest.TestCase):

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_generic_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Substance Identifier": ["456+123", "789", None],
        "Generic Name": ["Drug A", "Drug B", "Drug C"],
    })
    mock_read_csv.return_value = mock_df

    idx = generic_index.GenericIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_keys = ["123+456", "789", ""]
    self.assertEqual(list(idx.master_df["substance_key"]), expected_keys)
    self.assertIn("123+456", idx.groups)
    self.assertIn("789", idx.groups)
    self.assertIn("", idx.groups)

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_brand_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Product Identifier": [" 123 ", "456"],
        "Brand Name": ["Brand A", "Brand B"],
    })
    mock_read_csv.return_value = mock_df

    idx = brand_index.BrandIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_norm_ids = ["123", "456"]
    self.assertEqual(list(idx.master_df["norm_product_id"]), expected_norm_ids)
    self.assertIn("123", idx.groups)
    self.assertIn("456", idx.groups)

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_substance_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Substance Name": ["Substance A", "SUBSTANCE B"],
        "Identifier": [" 123 ", "456"],
    })
    mock_read_csv.return_value = mock_df

    idx = substance_index.SubstanceIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_norm_names = ["substance a", "substance b"]
    self.assertEqual(list(idx.master_df["norm_name"]), expected_norm_names)
    self.assertEqual(idx.groups, {"substance a": "123", "substance b": "456"})

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_product_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Product Name": ["Product A", "PRODUCT B"],
        "Identifier": [" 123 ", "456"],
    })
    mock_read_csv.return_value = mock_df

    idx = product_index.ProductIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_norm_names = ["product a", "product b"]
    self.assertEqual(
        list(idx.master_df["normalized_product_name"]), expected_norm_names
    )
    self.assertEqual(idx.groups, {"product a": "123", "product b": "456"})

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_substance_variations_index_load(
      self, mock_read_csv, mock_file_exists
  ):
    mock_df = pd.DataFrame({
        "Substance Name": ["Substance A", "Substance B"],
        "variations": ["var1 | var2", None],
    })
    mock_read_csv.return_value = mock_df

    idx = substance_variations_index.SubstanceVariationsIndex.from_index_file(
        "fake_path.txt"
    )

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    # norm_name should be index
    self.assertEqual(idx.master_df.index.name, "norm_name")
    self.assertEqual(
        idx.master_df.loc["substance a", "variations"], ["var1", "var2"]
    )
    self.assertEqual(idx.master_df.loc["substance b", "variations"], [""])

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_form_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Dose Form": ["Tablet", "CAPSULE"],
    })
    mock_read_csv.return_value = mock_df

    idx = form_index.FormIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_norm_forms = ["tablet", "capsule"]
    self.assertEqual(list(idx.master_df["norm_form"]), expected_norm_forms)

  @mock.patch.object(snomed_drug_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_route_index_load(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "RouteOfAdministration": ["Oral", "INTRAVENOUS"],
    })
    mock_read_csv.return_value = mock_df

    idx = route_index.RouteIndex.from_index_file("fake_path.txt")

    mock_file_exists.assert_called_once_with("fake_path.txt")
    self.assertIsNotNone(idx)
    expected_norm_routes = ["oral", "intravenous"]
    self.assertEqual(list(idx.master_df["norm_route"]), expected_norm_routes)

  def test_large_identifiers_in_tsv_maintain_precision(self):
    tsv_content = (
        "Product Identifier\tBrand Name\n"
        "1966571000189105\tBrand LargeInt\n"
        "196657100018910597682761243\tBrand HugeInt\n"
    )
    temp_file = self.create_tempfile(content=tsv_content)

    idx = brand_index.BrandIndex.from_index_file(temp_file.full_path)

    self.assertIsNotNone(idx)
    # The expected norm_product_id values should be strings with correct
    # precision.
    expected_ids = [
        "1966571000189105",
        "196657100018910597682761243",
    ]
    self.assertEqual(list(idx.master_df["norm_product_id"]), expected_ids)

    # Verify that the groups contain the correct keys.
    for val in expected_ids:
      self.assertIn(val, idx.groups)


if __name__ == "__main__":
  absltest.main()

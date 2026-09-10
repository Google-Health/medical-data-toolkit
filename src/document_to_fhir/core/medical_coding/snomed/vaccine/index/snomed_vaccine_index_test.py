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
"""Unit tests for snomed_vaccine_index."""

from unittest import mock

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.snomed.vaccine.index import snomed_vaccine_index

class SnomedVaccineIndexTest(absltest.TestCase):

  @mock.patch.object(snomed_vaccine_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_from_index_file_success(self, mock_read_csv, mock_file_exists):
    mock_df = pd.DataFrame({
        "Vaccine Name": ["Covaxin", "Polio Vaccine"],
        "Identifier": ["1119349007", "864441000168102"],
    })
    mock_read_csv.return_value = mock_df

    idx = snomed_vaccine_index.SnomedVaccineIndex.from_index_file(
        "fake_path.tsv"
    )

    mock_file_exists.assert_called_once_with("fake_path.tsv")
    mock_read_csv.assert_called_once_with(
        "fake_path.tsv", sep="\t", index_col=False, dtype=str
    )
    self.assertIsNotNone(idx)
    self.assertLen(idx.master_df, 2)
    self.assertEqual(
        list(idx.master_df["Vaccine Name"]), ["Covaxin", "Polio Vaccine"]
    )

  @mock.patch.object(snomed_vaccine_index, "_file_exists", return_value=False)
  @mock.patch.object(pd, "read_csv")
  def test_from_index_file_not_found(self, mock_read_csv, mock_file_exists):
    with self.assertRaisesRegex(
        FileNotFoundError, "File not found: nonexistent.tsv"
    ):
      snomed_vaccine_index.SnomedVaccineIndex.from_index_file("nonexistent.tsv")
    mock_file_exists.assert_called_once_with("nonexistent.tsv")
    mock_read_csv.assert_not_called()

  @mock.patch.object(snomed_vaccine_index, "_file_exists", return_value=True)
  @mock.patch.object(pd, "read_csv")
  def test_from_index_file_custom_kwargs(
      self, mock_read_csv, mock_file_exists
  ):
    mock_df = pd.DataFrame({"Vaccine Name": ["BCG"]})
    mock_read_csv.return_value = mock_df

    idx = snomed_vaccine_index.SnomedVaccineIndex.from_index_file(
        "fake_path.csv", sep=",", on_bad_lines="skip"
    )

    mock_file_exists.assert_called_once_with("fake_path.csv")
    mock_read_csv.assert_called_once_with(
        "fake_path.csv",
        sep=",",
        index_col=False,
        dtype=str,
        on_bad_lines="skip",
    )
    self.assertIsNotNone(idx)

  def test_large_identifiers_in_tsv_maintain_precision(self):
    tsv_content = (
        "Vaccine Name\tIdentifier\n"
        "Vaccine A\t1966571000189105\n"
        "Vaccine B\t196657100018910597682761243\n"
    )
    temp_file = self.create_tempfile(content=tsv_content)

    idx = snomed_vaccine_index.SnomedVaccineIndex.from_index_file(
        temp_file.full_path
    )

    self.assertIsNotNone(idx)
    expected_ids = [
        "1966571000189105",
        "196657100018910597682761243",
    ]
    self.assertEqual(list(idx.master_df["Identifier"]), expected_ids)

  def test_preprocess_clinical_variations_with_antigens(self):
    df = pd.DataFrame({
        "clinical_variations": [
            "polio virus vaccine",
            "hepatitis b vaccine",
        ],
        "antigens": [
            "antigen a + antigen b",
            float("nan"),
        ],
    })
    processed_df = snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    self.assertIn("norm_clinical_variations", processed_df.columns)
    self.assertEqual(
        list(processed_df["norm_clinical_variations"]),
        ["polio, a, b", "hepatitisb"],
    )

  def test_preprocess_without_clinical_variations(self):
    df = pd.DataFrame({
        "Vaccine Name": ["Covaxin", "BCG"],
    })
    processed_df = snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    self.assertIn("norm_clinical_variations", processed_df.columns)
    self.assertEqual(list(processed_df["norm_clinical_variations"]), ["", ""])

  def test_preprocess_patient_type(self):
    df = pd.DataFrame({
        "patient_type": [
            "adult",
            "paediatric",
            "unknown",
            "invalid",
            resources.PatientType.ADULT,
        ]
    })
    processed_df = snomed_vaccine_index.SnomedVaccineIndex._preprocess(df)
    self.assertEqual(
        [pt.value for pt in processed_df["patient_type"]],
        ["ADULT", "PAEDIATRIC", "UNKNOWN", "UNKNOWN", "ADULT"],
    )


if __name__ == "__main__":
  absltest.main()

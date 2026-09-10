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
"""Tests for resources.py."""

import datetime

from absl.testing import absltest

from src.document_to_fhir.common.schema import resources


class ResourcesTest(absltest.TestCase):

  def test_normalize_unit(self):
    self.assertEqual(resources.normalize_unit("500mg"), "500mg")
    self.assertEqual(resources.normalize_unit("1gm"), "1g")
    self.assertEqual(resources.normalize_unit("5gms"), "5g")
    self.assertEqual(resources.normalize_unit("2grams"), "2g")
    self.assertEqual(resources.normalize_unit("10milligrams"), "10mg")
    self.assertEqual(resources.normalize_unit("5ml"), "5ml")
    self.assertEqual(resources.normalize_unit("5mls"), "5ml")
    self.assertEqual(resources.normalize_unit("10milliliters"), "10ml")
    self.assertEqual(resources.normalize_unit("250mcg"), "250mcg")
    self.assertEqual(resources.normalize_unit("100micrograms"), "100mcg")
    self.assertEqual(resources.normalize_unit("1tab"), "1tablet")
    self.assertEqual(resources.normalize_unit("2tabs"), "2tablet")
    self.assertEqual(resources.normalize_unit("3tablets"), "3tablet")
    self.assertEqual(resources.normalize_unit("1cap"), "1capsule")
    self.assertEqual(resources.normalize_unit("2caps"), "2capsule")
    self.assertEqual(resources.normalize_unit("1inj"), "1injection")
    self.assertEqual(resources.normalize_unit("500IU"), "500iu")
    self.assertEqual(resources.normalize_unit("500 i.u."), "500 iu")
    self.assertEqual(resources.normalize_unit("10units"), "10iu")
    self.assertEqual(resources.normalize_unit("5percent"), "5%")
    self.assertEqual(resources.normalize_unit("5pct"), "5%")
    self.assertEqual(resources.normalize_unit("0.5% w/v"), "0.5% w/v")

    # Complex cases
    self.assertEqual(resources.normalize_unit("500mg + 125mg"), "500mg + 125mg")
    self.assertEqual(
        resources.normalize_unit("500 ml + 250 ml"), "500 ml + 250 ml"
    )
    self.assertEqual(resources.normalize_unit("1gm + 500mg"), "1g + 500mg")
    self.assertEqual(resources.normalize_unit("10mcg/ml"), "10mcg/ml")

    med = resources.Medication(brand_name="Test", strength="1gm + 250mg")
    self.assertEqual(med.strength, "1g + 250mg")
    self.assertEqual(med.name, "Test 1g + 250mg")

    med = resources.Medication(brand_name="Test", strength="500 mcg")
    self.assertEqual(med.strength, "500 mcg")
    self.assertEqual(med.name, "Test 500 mcg")

    med = resources.Medication(brand_name="Test", strength="500 microgram")
    self.assertEqual(med.strength, "500 mcg")
    self.assertEqual(med.name, "Test 500 mcg")

  def test_normalize_unit_empty(self):
    self.assertEqual(resources.normalize_unit(""), "")

  def test_medication_brand_name_validation(self):
    med = resources.Medication(brand_name=None)
    self.assertIsNone(med.brand_name)
    self.assertEqual(med.name, "")

    med = resources.Medication(brand_name="")
    self.assertEqual(med.brand_name, "")
    self.assertEqual(med.name, "")

  def test_medication_active_substances_validation(self):
    # Test basic formatting and strength removal
    med = resources.Medication(
        brand_name="Test", active_substances="Paracetamol 500mg"
    )
    self.assertEqual(med.active_substances, "Paracetamol")
    self.assertEqual(med.name, "Test (Paracetamol)")

    # Test when brand_name is None (active_substances should not have
    # parenthesis)
    med = resources.Medication(
        brand_name=None, active_substances="Paracetamol 500mg"
    )
    self.assertEqual(med.active_substances, "Paracetamol")
    self.assertEqual(med.name, "Paracetamol")

    # Test multiple active substances with different separators
    med = resources.Medication(
        brand_name="Test",
        active_substances="Amoxicillin 500mg / Clavulanate 125mg",
    )
    self.assertEqual(med.active_substances, "Amoxicillin + Clavulanate")
    self.assertEqual(med.name, "Test (Amoxicillin + Clavulanate)")

    med = resources.Medication(
        brand_name="Test",
        active_substances="Amoxicillin 500mg | Clavulanate 125mg",
    )
    self.assertEqual(med.active_substances, "Amoxicillin + Clavulanate")
    self.assertEqual(med.name, "Test (Amoxicillin + Clavulanate)")

    # Test None and empty
    med = resources.Medication(brand_name="Test", active_substances=None)
    self.assertIsNone(med.active_substances)
    self.assertEqual(med.name, "Test")

    med = resources.Medication(brand_name="Test", active_substances="")
    self.assertEqual(med.active_substances, "")
    self.assertEqual(med.name, "Test")

  def test_immunization_date_handling(self):
    # Valid date string
    imm = resources.Immunization.model_validate(
        {"vaccine_name": "Polio", "date": "2026-06-25"}
    )
    self.assertEqual(imm.date, datetime.date(2026, 6, 25))

    # ISO datetime string normalized to date
    imm_iso = resources.Immunization.model_validate(
        {"vaccine_name": "Polio", "date": "2026-06-25T12:57:00Z"}
    )
    self.assertEqual(imm_iso.date, datetime.date(2026, 6, 25))

    # Invalid date string cleared to None
    imm_invalid = resources.Immunization.model_validate(
        {"vaccine_name": "Polio", "date": "invalid-date"}
    )
    self.assertIsNone(imm_invalid.date)

  def test_medication_start_date_handling(self):
    # Valid date string
    med = resources.Medication.model_validate(
        {"brand_name": "Amox", "start_date": "2026-06-25"}
    )
    self.assertEqual(med.start_date, datetime.date(2026, 6, 25))

    # ISO datetime string normalized to date
    med_iso = resources.Medication.model_validate(
        {"brand_name": "Amox", "start_date": "2026-06-25T12:57:00Z"}
    )
    self.assertEqual(med_iso.start_date, datetime.date(2026, 6, 25))

    # Invalid date string cleared to None
    med_invalid = resources.Medication.model_validate(
        {"brand_name": "Amox", "start_date": "not-a-date"}
    )
    self.assertIsNone(med_invalid.start_date)


if __name__ == "__main__":
  absltest.main()

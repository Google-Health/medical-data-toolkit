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
"""Unit tests for LOINC ScaleMapper."""

from absl.testing import absltest
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.scale_type import mapper


class ScaleMapperTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mapper = mapper.ScaleMapper()

  def test_none_and_empty_returns_all_scales(self):
    self.assertEqual(self.mapper.get_canonical_scales(None), mapper.ALL_SCALES)
    self.assertEqual(self.mapper.get_canonical_scales(''), mapper.ALL_SCALES)
    self.assertEqual(self.mapper.get_canonical_scales('   '), mapper.ALL_SCALES)

  def test_phase1_exact_ast_terms(self):
    for term in ['Sensitive', 'resistant', 'SUSCEPTIBLE']:
      self.assertEqual(self.mapper.get_canonical_scales(term), {'Ord', 'OrdQn'})

  def test_phase1_exact_ordinal_and_plus_scales(self):
    for val in ['Positive', 'neg', '2+', '+++', '(++)', '(+)', '+/-', '±', '+-']:
      self.assertEqual(
          self.mapper.get_canonical_scales(val),
          {'Ord', 'Nom'},
          msg=f'Failed for {val!r}',
      )
    self.assertEqual(
        self.mapper.get_canonical_scales('0'),
        {'Qn', 'Ord', 'OrdQn', 'SemiQn'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales('(0)'),
        {'Qn', 'Ord', 'OrdQn', 'SemiQn'},
    )

  def test_phase1_exact_quantitative_scales(self):
    expected = {'Qn', 'Ord', 'OrdQn', 'SemiQn'}
    for val in ['12.5', '1.0093', '< 0.5', '>=1,000', '10 - 20', '1:40']:
      self.assertEqual(self.mapper.get_canonical_scales(val), expected)

  def test_phase2_relaxed_dirty_data_fallback(self):
    # Dirty numeric with flag/units does not leak 'Nom'
    self.assertEqual(
        self.mapper.get_canonical_scales('4.0092 L'),
        {'Qn', 'Ord', 'OrdQn', 'SemiQn'},
    )
    # Dirty parenthesized/digit plus or equivocal symbol in text does not leak 'Qn'
    self.assertEqual(
        self.mapper.get_canonical_scales('Protein (++)'),
        {'Ord', 'Nom', 'Nar'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales('Protein 2+'),
        {'Ord', 'Nom', 'Nar'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales('Ketones +-'),
        {'Ord', 'Nom', 'Nar'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales('Glucose +/-'),
        {'Ord', 'Nom', 'Nar'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales('Bilirubin ±'),
        {'Ord', 'Nom', 'Nar'},
    )
    # AST in text
    self.assertEqual(
        self.mapper.get_canonical_scales('Resistant to penicillin'),
        {'Ord', 'OrdQn', 'Nar'},
    )
    # Short nominal text vs long narrative text
    self.assertEqual(
        self.mapper.get_canonical_scales('Staphylococcus aureus'),
        {'Nom'},
    )
    self.assertEqual(
        self.mapper.get_canonical_scales(
            'This is a long narrative description of the test result.'
        ),
        {'Nom', 'Nar', 'Doc'},
    )

  def test_phase3_unrecognizable_symbols_fallback(self):
    self.assertEqual(self.mapper.get_canonical_scales('???'), mapper.ALL_SCALES)


if __name__ == '__main__':
  absltest.main()

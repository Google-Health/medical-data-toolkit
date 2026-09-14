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
"""Tests for the LOINC query engine."""

from unittest import mock

from absl.testing import absltest
import pandas as pd

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc.common import schema
from src.document_to_fhir.core.medical_coding.loinc.common import utils
from src.document_to_fhir.core.medical_coding.loinc.laboratory import query
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte import index
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property import mapper as property_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.scale_type import mapper as scale_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system import mapper as system_mapper_lib

# Consolidate Mock Data for all tests
MOCK_LOINC_RECORDS = [
    {
        'LOINC_NUM': '2',
        'core_analyte': 'Glucose',
        'LONG_COMMON_NAME': 'Glucose Random',
        'SYSTEM': 'Ser/Plas',
        'SCALE_TYP': 'Ord',
        'PROPERTY': 'SCnc',
        'COMMON_TEST_RANK': 20,
    },
    {
        'LOINC_NUM': '4',
        'core_analyte': 'Glucose',
        'LONG_COMMON_NAME': 'Glucose Ser Nar',
        'SYSTEM': 'Ser',
        'SCALE_TYP': 'Nar',
        'PROPERTY': 'MCnc',
        'COMMON_TEST_RANK': 40,
    },
    {
        'LOINC_NUM': '1',
        'core_analyte': 'Glucose',
        'LONG_COMMON_NAME': 'Glucose Ser/Plas Qn Mass',
        'SYSTEM': 'Ser/Plas',
        'SCALE_TYP': 'Qn',
        'PROPERTY': 'MCnc',
        'COMMON_TEST_RANK': 10,
    },
    {
        'LOINC_NUM': '3',
        'core_analyte': 'Glucose',
        'LONG_COMMON_NAME': 'Glucose Urine Nom',
        'SYSTEM': 'Urine',
        'SCALE_TYP': 'Nom',
        'PROPERTY': 'MCnc',
        'COMMON_TEST_RANK': 30,
    },
]


def mock_search_by_analyte(analyte_name):
  if analyte_name in ['Glucose', 'Sugar']:
    return [schema.LoincRow.model_validate(r) for r in MOCK_LOINC_RECORDS]
  return []


class LaboratoryLoincQueryEngineAnalyteTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte
    self.engine = query.LaboratoryLoincQueryEngine(self.mock_index)

  def test_query_no_core_analyte(self):
    test_input = resources.LabTest(core_analyte='', name='', result='100')
    results = self.engine.query(test_input)
    self.assertEqual(results, [])

  def test_query_exact_match_default_ranking(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', name='', result='100'
    )
    results = self.engine.query(test_input)
    self.mock_index.search_by_analyte.assert_called_once_with('Glucose')
    self.assertLen(results, 4)
    # Default ranking is by COMMON_TEST_RANK ascending (10, 20, 30, 40)
    self.assertEqual(results[0].loinc_num, '1')
    self.assertEqual(results[1].loinc_num, '2')
    self.assertEqual(results[2].loinc_num, '3')
    self.assertEqual(results[3].loinc_num, '4')

  def test_query_synonym_match(self):
    test_input = resources.LabTest(core_analyte='Sugar', name='', result='100')
    results = self.engine.query(test_input)
    self.mock_index.search_by_analyte.assert_called_once_with('Sugar')
    self.assertLen(results, 4)
    self.assertEqual(results[0].loinc_num, '1')

  def test_ranking_with_name(self):
    # Should rank 'Glucose Random' higher due to name similarity
    test_input = resources.LabTest(
        core_analyte='Glucose', name='Glucose Random', result='100'
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 4)
    # 'Glucose Random' is most similar to 'Glucose Random', so it should be
    # first.
    self.assertEqual(results[0].long_common_name, 'Glucose Random')
    self.assertEqual(results[0].loinc_num, '2')

  def test_query_batch(self):
    test_inputs = [
        resources.LabTest(core_analyte='Glucose', name='', result='100'),
        resources.LabTest(core_analyte='Sugar', name='', result='100'),
    ]
    results = self.engine.query_batch(test_inputs)
    self.assertLen(results, 2)
    self.assertLen(results[0], 4)
    self.assertEqual(results[0][0].loinc_num, '1')
    self.assertLen(results[1], 4)
    self.assertEqual(results[1][0].loinc_num, '1')

  def test_query_with_unit_but_no_property_mapper_succeeds(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', unit='mg/dL', name='', result='100'
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 4)


class LaboratoryLoincQueryEngineSystemTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte

    # Setup Systems KB with sets to support multi-mapping
    self.systems_kb = {
        'urine': {'Urine'},
        'serum': {'Ser/Plas', 'Ser'},
    }
    self.system_mapper = system_mapper_lib.SpecimenToSystemMapper(
        self.systems_kb
    )

  def test_query_with_system_filter(self):
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, system_mapper=self.system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='urine', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].loinc_num, '3')

  def test_query_without_system_filter_yields_wrong_default(self):
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, system_mapper=None
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='urine', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 4)
    self.assertEqual(results[0].loinc_num, '1')

  def test_query_fallback_when_no_match(self):
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, system_mapper=self.system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='blood', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 4)
    self.assertEqual(results[0].loinc_num, '1')

  def test_query_with_multi_mapping_filter(self):
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, system_mapper=self.system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='serum', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 3)
    loinc_nums = [r.loinc_num for r in results]
    self.assertIn('1', loinc_nums)
    self.assertIn('2', loinc_nums)
    self.assertIn('4', loinc_nums)
    self.assertNotIn('3', loinc_nums)

  def test_query_without_specimen_does_not_call_system_mapper(self):
    mock_system_mapper = mock.create_autospec(
        system_mapper_lib.SpecimenToSystemMapper, instance=True
    )
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, system_mapper=mock_system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', name='', result='100'
    )
    engine.query(test_input)
    mock_system_mapper.get_canonical_systems.assert_not_called()


class LaboratoryLoincQueryEngineScaleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte
    self.scale_mapper = scale_mapper_lib.ScaleMapper()
    self.engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, scale_mapper=self.scale_mapper
    )

  def test_query_with_numeric_result_picks_qn(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', result='100.5', name=''
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].scale_typ, 'Qn')
    self.assertEqual(results[0].loinc_num, '1')

  def test_query_with_text_result_picks_ord(self):
    test_input = resources.LabTest(core_analyte='Glucose', result='++', name='')
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].scale_typ, 'Ord')
    self.assertEqual(results[0].loinc_num, '2')

  def test_query_with_modified_numeric_result_picks_qn(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', result='< 0.35', name=''
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 3)
    loinc_nums = [r.loinc_num for r in results]
    self.assertIn('2', loinc_nums)
    self.assertIn('3', loinc_nums)
    self.assertIn('4', loinc_nums)

  def test_query_with_range_result_picks_qn(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', result='10 - 20', name=''
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 3)
    loinc_nums = [r.loinc_num for r in results]
    self.assertIn('2', loinc_nums)
    self.assertIn('3', loinc_nums)
    self.assertIn('4', loinc_nums)

  def test_query_with_positive_result_picks_ord(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', result='Positive', name=''
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].scale_typ, 'Ord')
    self.assertEqual(results[0].loinc_num, '2')

  def test_query_with_short_text_result_picks_nom(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', result='Staphylococcus aureus', name=''
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 3)
    loinc_nums = [r.loinc_num for r in results]
    self.assertIn('2', loinc_nums)
    self.assertIn('3', loinc_nums)
    self.assertIn('4', loinc_nums)

  def test_query_with_long_text_result_picks_nar(self):
    test_input = resources.LabTest(
        core_analyte='Glucose',
        result='This is a long narrative description of the test result.',
        name='',
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 3)
    loinc_nums = [r.loinc_num for r in results]
    self.assertIn('2', loinc_nums)
    self.assertIn('3', loinc_nums)
    self.assertIn('4', loinc_nums)

  def test_query_without_result_does_not_call_scale_mapper(self):
    mock_scale_mapper = mock.create_autospec(
        scale_mapper_lib.ScaleMapper, instance=True
    )
    engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, scale_mapper=mock_scale_mapper
    )
    test_input = resources.LabTest(core_analyte='Glucose', name='', result='')
    engine.query(test_input)
    mock_scale_mapper.get_canonical_scales.assert_not_called()


class LaboratoryLoincQueryEnginePropertyTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte
    self.property_kb = {
        'mg/dl': {'MCnc'},
        'mmol/l': {'SCnc'},
    }
    self.property_mapper = property_mapper_lib.UnitToPropertyMapper(
        self.property_kb
    )
    self.engine = query.LaboratoryLoincQueryEngine(
        self.mock_index, property_mapper=self.property_mapper
    )

  def test_query_with_mass_unit_picks_mcnc(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', unit='mg/dL', name='', result='100'
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 3)
    self.assertEqual(results[0].property, 'MCnc')
    self.assertEqual(results[0].loinc_num, '1')
    self.assertEqual(results[1].property, 'MCnc')
    self.assertEqual(results[1].loinc_num, '3')
    self.assertEqual(results[2].property, 'MCnc')
    self.assertEqual(results[2].loinc_num, '4')

  def test_query_with_molar_unit_picks_scnc(self):
    test_input = resources.LabTest(
        core_analyte='Glucose', unit='mmol/L', name='', result='100'
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].property, 'SCnc')
    self.assertEqual(results[0].loinc_num, '2')


class LaboratoryLoincQueryEngineIntegrationTest(absltest.TestCase):
  """Integration tests with all mappers configured."""

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte

    self.systems_kb = {
        'urine': {'Urine'},
        'serum': {'Ser/Plas', 'Ser'},
    }
    self.system_mapper = system_mapper_lib.SpecimenToSystemMapper(
        self.systems_kb
    )
    self.scale_mapper = scale_mapper_lib.ScaleMapper()
    self.property_kb = {
        'mg/dl': {'MCnc'},
        'mmol/l': {'SCnc'},
    }
    self.property_mapper = property_mapper_lib.UnitToPropertyMapper(
        self.property_kb
    )

    self.engine = query.LaboratoryLoincQueryEngine(
        analyte_index=self.mock_index,
        system_mapper=self.system_mapper,
        scale_mapper=self.scale_mapper,
        property_mapper=self.property_mapper,
    )

  def test_integration_query_matches_all_criteria(self):
    test_input = resources.LabTest(
        core_analyte='Glucose',
        specimen='urine',
        result='100.5',
        unit='mg/dL',
        name='',
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].loinc_num, '3')

  def test_integration_query_prioritizes_best_match(self):
    test_input = resources.LabTest(
        core_analyte='Glucose',
        specimen='serum',
        result='100.5',
        unit='mmol/L',
        name='',
    )
    results = self.engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].loinc_num, '1')


class UnitToPropertyMapperTest(absltest.TestCase):

  @mock.patch.object(utils, 'read_csv', autospec=True, spec_set=True)
  def test_load_mapping_from_csv(self, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        'property': ['MCnc', 'SCnc'],
        'synonym': ['mg/dl', 'mmol/l'],
    })
    mapper_inst = property_mapper_lib.UnitToPropertyMapper.from_csv(
        csv_path='dummy_path.csv'
    )
    props = mapper_inst.get_canonical_properties('mg/dl')
    self.assertIn('MCnc', props)
    props = mapper_inst.get_canonical_properties('mmol/l')
    self.assertIn('SCnc', props)

  def test_get_inferred_properties(self):
    mapper_inst = property_mapper_lib.UnitToPropertyMapper({'mg/dl': {'MCnc'}})
    props = mapper_inst.get_canonical_properties('mg/dL')
    self.assertIn('MCnc', props)


class SpecimenToSystemMapperTest(absltest.TestCase):

  @mock.patch.object(utils, 'read_csv', autospec=True, spec_set=True)
  def test_load_mapping_from_csv(self, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        'canonical': ['Ser/Plas', 'Urine'],
        'synonym': ['serum', 'urine'],
    })
    sys_mapper = system_mapper_lib.SpecimenToSystemMapper.from_csv(
        csv_path='dummy_path.csv'
    )
    systems = sys_mapper.get_canonical_systems('serum')
    self.assertIn('Ser/Plas', systems)
    systems = sys_mapper.get_canonical_systems('urine')
    self.assertIn('Urine', systems)


class RankingRegressionTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.records = [
        # Bilirubin Total vs Direct
        {
            'LOINC_NUM': '1975-2',
            'core_analyte': 'Bilirubin',
            'LONG_COMMON_NAME': (
                'Bilirubin.total [Mass/volume] in Serum or Plasma'
            ),
            'SYSTEM': 'Ser/Plas',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'MCnc',
            'COMMON_TEST_RANK': 22,
        },
        {
            'LOINC_NUM': '1968-7',
            'core_analyte': 'Bilirubin',
            'LONG_COMMON_NAME': (
                'Bilirubin.direct [Mass/volume] in Serum or Plasma'
            ),
            'SYSTEM': 'Ser/Plas',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'MCnc',
            'COMMON_TEST_RANK': 131,
        },
        # Albumin Serum vs Urine
        {
            'LOINC_NUM': '1751-7',
            'core_analyte': 'Albumin',
            'LONG_COMMON_NAME': 'Albumin [Mass/volume] in Serum or Plasma',
            'SYSTEM': 'Ser/Plas',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'MCnc',
            'COMMON_TEST_RANK': 39,
        },
        {
            'LOINC_NUM': '1754-1',
            'core_analyte': 'Albumin',
            'LONG_COMMON_NAME': 'Albumin [Mass/volume] in Urine',
            'SYSTEM': 'Urine',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'MCnc',
            'COMMON_TEST_RANK': 272,
        },
        # Hemoglobin vs MCH
        {
            'LOINC_NUM': '718-7',
            'core_analyte': 'Hemoglobin',
            'LONG_COMMON_NAME': 'Hemoglobin [Mass/volume] in Blood',
            'SYSTEM': 'Bld',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'MCnc',
            'COMMON_TEST_RANK': 17,
        },
        {
            'LOINC_NUM': '785-6',
            'core_analyte': 'MCH',
            'LONG_COMMON_NAME': 'MCH [Entitic mass] by Automated count',
            'SYSTEM': 'Bld',
            'SCALE_TYP': 'Qn',
            'PROPERTY': 'EntMass',
            'COMMON_TEST_RANK': 15,
        },
    ]
    self.mock_index = mock.create_autospec(
        index.AnalytesIndex, instance=True, spec_set=True
    )

    def search_analyte(analyte):
      res = [
          r
          for r in self.records
          if r['core_analyte'].lower() == analyte.lower()
      ]
      return [schema.LoincRow.model_validate(r) for r in res]

    self.mock_index.search_by_analyte.side_effect = search_analyte
    self.engine = query.LaboratoryLoincQueryEngine(self.mock_index)

  def test_bilirubin_direct_disambiguation(self):
    # When query contains 'Direct', 1968-7 should outrank 1975-2 despite lower
    # rank.
    test_input = resources.LabTest(
        core_analyte='Bilirubin', name='Bilirubin Direct', result='0.2'
    )
    results = self.engine.query(test_input)
    self.assertNotEmpty(results)
    self.assertEqual(results[0].loinc_num, '1968-7')
    self.assertEqual(results[1].loinc_num, '1975-2')

  def test_bilirubin_total_disambiguation(self):
    # When query contains 'Total', 1975-2 should be ranked #1
    test_input = resources.LabTest(
        core_analyte='Bilirubin', name='Bilirubin Total', result='0.8'
    )
    results = self.engine.query(test_input)
    self.assertNotEmpty(results)
    self.assertEqual(results[0].loinc_num, '1975-2')
    self.assertEqual(results[1].loinc_num, '1968-7')

  def test_albumin_serum_vs_urine_rank_prior(self):
    # When specimen is unspecified, rank prior breaks tie in favor of Serum
    # (rank 39) over Urine (rank 272).
    test_input = resources.LabTest(
        core_analyte='Albumin', name='Albumin', result='4.2'
    )
    results = self.engine.query(test_input)
    self.assertNotEmpty(results)
    self.assertEqual(results[0].loinc_num, '1751-7')
    self.assertEqual(results[1].loinc_num, '1754-1')

  def test_haemoglobin_commonwealth_spelling(self):
    # 'Haemoglobin' should fuzzy match 'Hemoglobin' and rank 718-7 #1
    test_input = resources.LabTest(
        core_analyte='Hemoglobin', name='Haemoglobin', result='13.5'
    )
    results = self.engine.query(test_input)
    self.assertNotEmpty(results)
    self.assertEqual(results[0].loinc_num, '718-7')

  def test_long_query_token_overlap_dominates_rank(self):
    # For a long query (>20 tokens), 1 token match must still beat a 0-token
    # match even when comparing against a higher LOINC popularity rank.
    long_name = ' '.join([f'modifier{i}' for i in range(25)] + ['Direct'])
    test_input = resources.LabTest(
        core_analyte='Bilirubin', name=long_name, result='0.2'
    )
    results = self.engine.query(test_input)
    self.assertNotEmpty(results)
    # 1968-7 matches 'direct' from query; 1975-2 matches 0 tokens.
    # 1968-7 must rank #1 despite lower rank (131 vs 22).
    self.assertEqual(results[0].loinc_num, '1968-7')


if __name__ == '__main__':
  absltest.main()

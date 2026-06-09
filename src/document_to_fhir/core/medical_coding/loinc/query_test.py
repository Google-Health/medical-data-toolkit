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

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.medical_coding.loinc import query
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import index
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.property import mapper as property_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.scale_type import mapper as scale_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.system import mapper as system_mapper_lib
from src.document_to_fhir.core.medical_coding.loinc.common import schema

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


class LoincQueryEngineAnalyteTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte
    self.engine = query.LoincQueryEngine(self.mock_index)

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


class LoincQueryEngineSystemTest(absltest.TestCase):

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
    engine = query.LoincQueryEngine(
        self.mock_index, system_mapper=self.system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='urine', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 1)
    self.assertEqual(results[0].loinc_num, '3')

  def test_query_without_system_filter_yields_wrong_default(self):
    engine = query.LoincQueryEngine(self.mock_index, system_mapper=None)
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='urine', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 4)
    self.assertEqual(results[0].loinc_num, '1')

  def test_query_fallback_when_no_match(self):
    engine = query.LoincQueryEngine(
        self.mock_index, system_mapper=self.system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', specimen='blood', name='', result='100'
    )
    results = engine.query(test_input)
    self.assertLen(results, 4)
    self.assertEqual(results[0].loinc_num, '1')

  def test_query_with_multi_mapping_filter(self):
    engine = query.LoincQueryEngine(
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
    engine = query.LoincQueryEngine(
        self.mock_index, system_mapper=mock_system_mapper
    )
    test_input = resources.LabTest(
        core_analyte='Glucose', name='', result='100'
    )
    engine.query(test_input)
    mock_system_mapper.get_canonical_systems.assert_not_called()


class LoincQueryEngineScaleTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_index = mock.create_autospec(index.AnalytesIndex, instance=True)
    self.mock_index.search_by_analyte.side_effect = mock_search_by_analyte
    self.scale_mapper = scale_mapper_lib.ScaleMapper()
    self.engine = query.LoincQueryEngine(
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
    engine = query.LoincQueryEngine(
        self.mock_index, scale_mapper=mock_scale_mapper
    )
    test_input = resources.LabTest(core_analyte='Glucose', name='', result='')
    engine.query(test_input)
    mock_scale_mapper.get_canonical_scales.assert_not_called()


class LoincQueryEnginePropertyTest(absltest.TestCase):

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
    self.engine = query.LoincQueryEngine(
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


class LoincQueryEngineIntegrationTest(absltest.TestCase):
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

    self.engine = query.LoincQueryEngine(
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

  @mock.patch(
      'builtins.open',
      new_callable=mock.mock_open,
      read_data='property,synonym\nMCnc,mg/dl\nSCnc,mmol/l\n',
  )
  def test_load_mapping_from_csv(self, mock_file):
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

  @mock.patch(
      'builtins.open',
      new_callable=mock.mock_open,
      read_data='canonical,synonym\nSer/Plas,serum\nUrine,urine\n',
  )
  def test_load_mapping_from_csv(self, mock_file):
    sys_mapper = system_mapper_lib.SpecimenToSystemMapper.from_csv(
        csv_path='dummy_path.csv'
    )
    systems = sys_mapper.get_canonical_systems('serum')
    self.assertIn('Ser/Plas', systems)
    systems = sys_mapper.get_canonical_systems('urine')
    self.assertIn('Urine', systems)


if __name__ == '__main__':
  absltest.main()

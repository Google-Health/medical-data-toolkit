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
"""Tests for rest_server API entry points."""

import io
import unittest.mock

from absl.testing import absltest
import PIL.Image
import pypdfium2

from src import rest_server


class RestServerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    rest_server.flask_app.testing = True
    self.client = rest_server.flask_app.test_client()
    rest_server._composite_standardizer = None
    self.max_pages_patcher = unittest.mock.patch(
        'src.rest_server._get_max_pdf_pages',
        return_value=40,
    )
    self.max_pages_patcher.start()

  def tearDown(self):
    super().tearDown()
    self.max_pages_patcher.stop()

  def test_healthcheck(self):
    response = self.client.get('/')
    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.data, b'Healthcheck OK')

  def test_document_to_fhir_empty_body(self):
    response = self.client.post('/document_to_fhir', data=b'')
    self.assertEqual(response.status_code, 400)
    self.assertIn(rest_server._EMPTY_BODY_ERROR.encode('utf-8'), response.data)

  def test_document_to_fhir_zero_pages(self):
    # Create a pdf document with no pagesusing pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()
    response = self.client.post('/document_to_fhir', data=pdf_bytes)
    self.assertEqual(response.status_code, 400)
    self.assertIn(
        rest_server._NOT_PDF_OR_IMAGE_BYTES_ERROR.encode('utf-8'), response.data
    )

  def test_document_to_fhir_invalid_both(self):
    response = self.client.post('/document_to_fhir', data=b'some data')
    self.assertEqual(response.status_code, 400)
    self.assertIn(
        rest_server._NOT_PDF_OR_IMAGE_BYTES_ERROR.encode('utf-8'), response.data
    )

  @unittest.mock.patch(
      'src.rest_server.get_composite_standardizer'
  )
  def test_document_to_fhir_valid_pdf_missing_standardizer(
      self, mock_get_standardizer):
    mock_get_standardizer.return_value = None
    # Create a valid 1 page empty pdf using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    response = self.client.post('/document_to_fhir', data=pdf_bytes)
    self.assertEqual(response.status_code, 500)
    self.assertIn(
        b'CompositeDocumentStandardizer not initialized', response.data
    )

  def test_document_to_fhir_success(self):
    # Create a valid 1 page empty pdf using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    mock_standardizer = unittest.mock.MagicMock()
    mock_result = unittest.mock.MagicMock()
    mock_result.model_dump.return_value = {'status': 'success'}
    mock_standardizer.standardize.return_value = mock_result

    with unittest.mock.patch(
        'src.rest_server.get_composite_standardizer',
        return_value=mock_standardizer,
    ):
      response = self.client.post('/document_to_fhir', data=pdf_bytes)
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.json, {'status': 'success'})
      mock_standardizer.standardize.assert_called_once_with(
          pdf_bytes, mime_type='application/pdf'
      )

  @unittest.mock.patch(
      'src.rest_server.get_composite_standardizer'
  )
  def test_document_to_fhir_with_job_id_logging(self, mock_get_standardizer):
    # Mock standardizer
    mock_standardizer = unittest.mock.MagicMock()
    mock_result = unittest.mock.MagicMock()
    mock_result.model_dump.return_value = {'status': 'success'}
    mock_standardizer.standardize.return_value = mock_result
    mock_get_standardizer.return_value = mock_standardizer

    # Create a valid 1 page empty pdf
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    # We want to assert that logs printed during this request contain the job ID.
    with self.assertLogs(level='INFO') as log_watcher:
      response = self.client.post(
          '/document_to_fhir',
          data=pdf_bytes,
          headers={'X-Job-Id': 'test-job-123'},
      )
      self.assertEqual(response.status_code, 200)

    # Check that at least one log message contains the job ID prefix
    has_job_id = False
    for log in log_watcher.output:
      if '[job=test-job-123]' in log:
        has_job_id = True
        break
    self.assertTrue(
        has_job_id, f'Job ID not found in logs: {log_watcher.output}'
    )

  def test_document_to_fhir_validation_error(self):
    # Create a valid 1 page empty pdf using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    mock_standardizer = unittest.mock.MagicMock()
    mock_standardizer.standardize.side_effect = ValueError('Invalid data')

    with unittest.mock.patch(
        'src.rest_server.get_composite_standardizer',
        return_value=mock_standardizer,
    ):
      response = self.client.post('/document_to_fhir', data=pdf_bytes)
      self.assertEqual(response.status_code, 422)
      self.assertIn(
          b'Document standardization failed: Invalid data', response.data
      )

  def test_document_to_fhir_valid_image(self):
    # Create an 5x5 image using PIL
    with io.BytesIO() as img_byte_arr:
      with PIL.Image.new('RGB', (5, 5)) as img:
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

    mock_standardizer = unittest.mock.MagicMock()
    mock_result = unittest.mock.MagicMock()
    mock_result.model_dump.return_value = {'status': 'success'}
    mock_standardizer.standardize.return_value = mock_result

    with unittest.mock.patch(
        'src.rest_server.get_composite_standardizer',
        return_value=mock_standardizer,
    ):
      response = self.client.post('/document_to_fhir', data=img_bytes)
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.json, {'status': 'success'})
      mock_standardizer.standardize.assert_called_once_with(
          img_bytes, mime_type='image/png'
      )

  def test_document_to_fhir_pdf_with_header(self):
    # Create a valid 1 page empty pdf using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    mock_standardizer = unittest.mock.MagicMock()
    mock_result = unittest.mock.MagicMock()
    mock_result.model_dump.return_value = {'status': 'success'}
    mock_standardizer.standardize.return_value = mock_result

    with unittest.mock.patch(
        'src.rest_server.get_composite_standardizer',
        return_value=mock_standardizer,
    ):
      response = self.client.post(
          '/document_to_fhir',
          data=pdf_bytes,
          headers={'Content-Type': 'application/pdf'},
      )
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.json, {'status': 'success'})
      mock_standardizer.standardize.assert_called_once_with(
          pdf_bytes, mime_type='application/pdf'
      )

  def test_document_to_fhir_image_with_header(self):
    # Create an 5x5 image using PIL
    with io.BytesIO() as img_byte_arr:
      with PIL.Image.new('RGB', (5, 5)) as img:
        img.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()

    mock_standardizer = unittest.mock.MagicMock()
    mock_result = unittest.mock.MagicMock()
    mock_result.model_dump.return_value = {'status': 'success'}
    mock_standardizer.standardize.return_value = mock_result

    with unittest.mock.patch(
        'src.rest_server.get_composite_standardizer',
        return_value=mock_standardizer,
    ):
      response = self.client.post(
          '/document_to_fhir',
          data=img_bytes,
          headers={'Content-Type': 'image/jpeg'},
      )
      self.assertEqual(response.status_code, 200)
      self.assertEqual(response.json, {'status': 'success'})
      mock_standardizer.standardize.assert_called_once_with(
          img_bytes, mime_type='image/jpeg'
      )

  def test_document_to_fhir_pdf_mismatch_header(self):
    # Create a valid 1 page empty pdf using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    response = self.client.post(
        '/document_to_fhir',
        data=pdf_bytes,
        headers={'Content-Type': 'image/png'},
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn(
        b'Data is not a valid image but header claims it is.', response.data
    )

  def test_document_to_fhir_unsupported_header(self):
    response = self.client.post(
        '/document_to_fhir',
        data=b'some data',
        headers={'Content-Type': 'text/plain'},
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn(b'Unsupported Content-Type: text/plain', response.data)

  def test_document_to_fhir_image_unsupported_format(self):
    with unittest.mock.patch(
        'src.rest_server.'
        '_is_image',
        return_value=True,
    ), unittest.mock.patch(
        'PIL.Image.open',
        side_effect=Exception('Failed to read format'),
    ):
      response = self.client.post('/document_to_fhir', data=b'some data')
      self.assertEqual(response.status_code, 400)
      self.assertIn(b'Unsupported image format.', response.data)

  def test_document_to_fhir_tiff_image_no_header(self):
    with io.BytesIO() as img_byte_arr:
      with PIL.Image.new('RGB', (5, 5)) as img:
        img.save(img_byte_arr, format='TIFF')
        img_bytes = img_byte_arr.getvalue()

    response = self.client.post('/document_to_fhir', data=img_bytes)
    self.assertEqual(response.status_code, 400)
    self.assertIn(b'Unsupported image format.', response.data)

  def test_document_to_fhir_tiff_image_with_header(self):
    with io.BytesIO() as img_byte_arr:
      with PIL.Image.new('RGB', (5, 5)) as img:
        img.save(img_byte_arr, format='TIFF')
        img_bytes = img_byte_arr.getvalue()

    response = self.client.post(
        '/document_to_fhir',
        data=img_bytes,
        headers={'Content-Type': 'image/tiff'},
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn(b'Unsupported Content-Type: image/tiff', response.data)

  @unittest.mock.patch(
      'src.rest_server._CONFIG_FILE'
  )
  @unittest.mock.patch(
      'src.rest_server.os.path.exists'
  )
  @unittest.mock.patch(
      'src.rest_server.yaml.safe_load'
  )
  @unittest.mock.patch(
      'builtins.open', new_callable=unittest.mock.mock_open, read_data='dummy'
  )
  @unittest.mock.patch(
      'src.rest_server.composite_document_standardizer.CompositeDocumentStandardizer'
  )
  @unittest.mock.patch(
      'src.rest_server._create_llm_client'
  )
  def test_get_composite_standardizer_initialization(
      self,
      mock_create_client,
      mock_standardizer_class,
      mock_open,
      mock_yaml_load,
      mock_exists,
      mock_config_file,
  ):
    rest_server._composite_standardizer = None  # Ensure re-initialization
    mock_exists.return_value = True
    mock_create_client.return_value = unittest.mock.MagicMock()
    mock_config_file.value = 'dummy.yaml'
    mock_yaml_load.return_value = {
        'classifier_llm_client': {
            'type': 'GeminiClient',
            'parameters': {'model': 'gemini-1.5-flash'},
        },
        'extractor_llm_client': {
            'type': 'GeminiClient',
            'parameters': {'model': 'gemini-1.5-pro'},
        },
        'supported_types': ['LABORATORY_REPORT'],
        'document_standardization_policy': 'ALLOW_ONLY_SUPPORTED',
        'loinc_analaytes_index_csv_path': 'dummy.csv',
        'loinc_system_kb_csv_path': 'dummy_system.csv',
        'loinc_property_kb_csv_path': 'dummy_property.csv',
        'loinc_rsna_playbook_index_csv_path': 'dummy_rsna.csv',
        'snomed_drug_brand_kb_path': 'dummy_brand.txt',
        'snomed_drug_generic_kb_path': 'dummy_generic.txt',
        'snomed_drug_substance_kb_path': 'dummy_substance.txt',
        'snomed_drug_product_kb_path': 'dummy_product.txt',
        'snomed_drug_substance_variations_path': 'dummy_variations.txt',
        'snomed_drug_form_kb_path': 'dummy_form.txt',
        'snomed_drug_route_kb_path': 'dummy_route.txt',
        'snomed_vaccine_kb_path': 'dummy_vaccine.tsv',
    }

    with (
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte.index.AnalytesIndex.from_csv'
        ) as mock_from_csv,
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system.mapper.SpecimenToSystemMapper.from_csv'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property.mapper.UnitToPropertyMapper.from_csv'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.loinc.radiology.query.RsnaPlaybookQueryEngine.from_csv'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.brand_index.BrandIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.generic_index.GenericIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_index.SubstanceIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.product_index.ProductIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_variations_index.SubstanceVariationsIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.form_index.FormIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.drug.index.route_index.RouteIndex.from_index_file'
        ),
        unittest.mock.patch(
            'src.document_to_fhir.core.medical_coding.snomed.vaccine.index.snomed_vaccine_index.SnomedVaccineIndex.from_index_file'
        ),
    ):
      mock_from_csv.return_value = unittest.mock.MagicMock()
      standardizer = rest_server.get_composite_standardizer()
      mock_standardizer_class.assert_called_once_with(
          classifier=unittest.mock.ANY,
          standardizers=unittest.mock.ANY,
          document_standardization_policy=rest_server.document_types.DocumentStandardizationPolicy.ALLOW_ONLY_SUPPORTED,
          attach_document_to_bundle=False,
          return_metadata=False,
      )
      self.assertIsNotNone(standardizer)
      self.assertEqual(standardizer, mock_standardizer_class.return_value)
      # test singleton
      standardizer2 = rest_server.get_composite_standardizer()
      self.assertIs(standardizer, standardizer2)
      self.assertEqual(mock_standardizer_class.call_count, 1)

  @unittest.mock.patch.dict(
      'src.rest_server.os.environ',
      {'LITELLM_API_KEY': 'fake_key'},
  )
  @unittest.mock.patch(
      'src.document_to_fhir.common.model_client.LiteLLMClient'
  )
  def test_create_llm_client_litellm_with_config(self, mock_litellm_client):
    config = {
        'type': 'LiteLLMClient',
        'parameters': {
            'model': 'openai/google/gemma-4-31B-it',
            'api_key_env': 'LITELLM_API_KEY',
            'api_base': 'http://localhost:8000/v1',
            'config': {
                'extra_body': {
                    'mm_processor_kwargs': {'max_soft_tokens': 560}
                }
            }
        }
    }
    rest_server._create_llm_client(config)
    mock_litellm_client.assert_called_once_with(
        model='openai/google/gemma-4-31B-it',
        api_base='http://localhost:8000/v1',
        api_key='fake_key',
        temperature=0.0,
        config={
            'extra_body': {'mm_processor_kwargs': {'max_soft_tokens': 560}}
        },
        verbose=False,
        timeout=300.0,
        max_retries=3,
        supports_pdf=False,
        enable_thinking=False,
    )

  def test_pdf_generation_for_page_limit(self):
    max_pages = 40
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      for _ in range(max_pages + 1):
        pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    actual_count = rest_server._pdf_page_count(pdf_bytes)
    self.assertEqual(actual_count, max_pages + 1)

  def test_document_to_fhir_pdf_too_many_pages(self):
    max_pages = 40
    # Create a PDF with 41 pages using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      for _ in range(max_pages + 1):
        pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    response = self.client.post('/document_to_fhir', data=pdf_bytes)
    self.assertEqual(response.status_code, 400)
    self.assertIn(
        'PDF exceeds the maximum allowed number of pages.'.encode('utf-8'),
        response.data,
    )

  def test_document_to_fhir_pdf_too_many_pages_with_header(self):
    max_pages = 40
    # Create a PDF with 41 pages using pypdfium2
    with io.BytesIO() as out_buffer:
      pdf = pypdfium2.PdfDocument.new()
      for _ in range(max_pages + 1):
        pdf.new_page(612, 792)
      pdf.save(out_buffer)
      pdf_bytes = out_buffer.getvalue()

    response = self.client.post(
        '/document_to_fhir',
        data=pdf_bytes,
        headers={'Content-Type': 'application/pdf'},
    )
    self.assertEqual(response.status_code, 400)
    self.assertIn(
        'PDF exceeds the maximum allowed number of pages.'.encode('utf-8'),
        response.data,
    )

  @unittest.mock.patch(
      'src.rest_server._CONFIG_FILE'
  )
  @unittest.mock.patch(
      'src.rest_server.os.path.exists'
  )
  @unittest.mock.patch(
      'src.rest_server.yaml.safe_load'
  )
  @unittest.mock.patch(
      'builtins.open', new_callable=unittest.mock.mock_open, read_data='dummy'
  )
  def test_get_max_pdf_pages_initialization_and_caching(
      self,
      _,
      mock_yaml_load,
      mock_exists,
      mock_config_file,
  ):
    self.max_pages_patcher.stop()
    rest_server._max_pdf_pages = None  # Ensure re-initialization
    mock_exists.return_value = True
    mock_config_file.value = 'dummy.yaml'
    mock_yaml_load.return_value = {'max_pdf_pages': 55}

    # First call should load from config
    max_pages = rest_server._get_max_pdf_pages()
    self.assertEqual(max_pages, 55)
    mock_yaml_load.assert_called_once()

    # Second call should use cached value and NOT call yaml_load again
    mock_yaml_load.reset_mock()
    max_pages2 = rest_server._get_max_pdf_pages()
    self.assertEqual(max_pages2, 55)
    mock_yaml_load.assert_not_called()

  def test_get_config_variable_success(self):
    config = {'test_key': 'test_value'}
    self.assertEqual(
        rest_server._get_config_variable(config, 'test_key'), 'test_value'
    )

  def test_get_config_variable_missing_key(self):
    config = {'other_key': 'value'}
    with self.assertRaisesRegex(
        ValueError, 'test_key is required in config\\.'
    ):
      rest_server._get_config_variable(config, 'test_key')

  def test_get_config_variable_empty_value(self):
    for empty_val in ['', None, {}, []]:
      with self.subTest(empty_val=empty_val):
        config = {'test_key': empty_val}
        with self.assertRaisesRegex(
            ValueError, 'test_key is required in config\\.'
        ):
          rest_server._get_config_variable(config, 'test_key')

  def test_get_config_variable_invalid_config(self):
    for invalid_config in [None, 'not a mapping']:
      with self.subTest(invalid_config=invalid_config):
        with self.assertRaisesRegex(
            ValueError, 'test_key is required in config\\.'
        ):
          rest_server._get_config_variable(invalid_config, 'test_key')

  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.core_analyte.index.AnalytesIndex.from_csv'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.system.mapper.SpecimenToSystemMapper.from_csv'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.loinc.laboratory.axes_kb.property.mapper.UnitToPropertyMapper.from_csv'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.loinc.laboratory.query.LaboratoryLoincQueryEngine'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.loinc.radiology.query.RsnaPlaybookQueryEngine.from_csv'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.mapper.terminology_mappers.loinc_terminology_mapper.LoincTerminologyMapper'
  )
  def test_create_loinc_mapper(
      self,
      mock_mapper_cls,
      mock_rsna_mapper_csv,
      mock_query_engine_cls,
      mock_property_mapper_csv,
      mock_system_mapper_csv,
      mock_analyte_index_csv,
  ):
    config = {
        'loinc_analaytes_index_csv_path': 'dummy_analyte.csv',
        'loinc_system_kb_csv_path': 'dummy_system.csv',
        'loinc_property_kb_csv_path': 'dummy_property.csv',
        'loinc_rsna_playbook_index_csv_path': 'dummy_rsna.csv',
    }
    mapper = rest_server._create_loinc_mapper(config)
    mock_analyte_index_csv.assert_called_once_with('dummy_analyte.csv')
    mock_system_mapper_csv.assert_called_once_with('dummy_system.csv')
    mock_property_mapper_csv.assert_called_once_with('dummy_property.csv')
    mock_rsna_mapper_csv.assert_called_once_with('dummy_rsna.csv')
    mock_query_engine_cls.assert_called_once_with(
        analyte_index=mock_analyte_index_csv.return_value,
        system_mapper=mock_system_mapper_csv.return_value,
        scale_mapper=unittest.mock.ANY,
        property_mapper=mock_property_mapper_csv.return_value,
    )
    mock_mapper_cls.assert_called_once_with(
        lab_query_engine=mock_query_engine_cls.return_value,
        radiology_query_engine=mock_rsna_mapper_csv.return_value,
    )
    self.assertEqual(mapper, mock_mapper_cls.return_value)

  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.brand_index.BrandIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.generic_index.GenericIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_index.SubstanceIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.product_index.ProductIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.substance_variations_index.SubstanceVariationsIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.form_index.FormIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.index.route_index.RouteIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.drug.query.snomed_drug_query_engine.SnomedDrugQueryEngine'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.mapper.terminology_mappers.snomed_drug_terminology_mapper.SnomedDrugTerminologyMapper'
  )
  def test_create_snomed_drug_mapper(
      self,
      mock_mapper_cls,
      mock_query_engine_cls,
      mock_route_idx,
      mock_form_idx,
      mock_variations_idx,
      mock_product_idx,
      mock_substance_idx,
      mock_generic_idx,
      mock_brand_idx,
  ):
    config = {
        'snomed_drug_brand_kb_path': 'dummy_brand.txt',
        'snomed_drug_generic_kb_path': 'dummy_generic.txt',
        'snomed_drug_substance_kb_path': 'dummy_substance.txt',
        'snomed_drug_product_kb_path': 'dummy_product.txt',
        'snomed_drug_substance_variations_path': 'dummy_variations.txt',
        'snomed_drug_form_kb_path': 'dummy_form.txt',
        'snomed_drug_route_kb_path': 'dummy_route.txt',
    }
    mapper = rest_server._create_snomed_drug_mapper(config)
    mock_brand_idx.assert_called_once_with('dummy_brand.txt')
    mock_generic_idx.assert_called_once_with('dummy_generic.txt')
    mock_substance_idx.assert_called_once_with('dummy_substance.txt')
    mock_product_idx.assert_called_once_with('dummy_product.txt')
    mock_variations_idx.assert_called_once_with('dummy_variations.txt')
    mock_form_idx.assert_called_once_with('dummy_form.txt')
    mock_route_idx.assert_called_once_with('dummy_route.txt')
    mock_query_engine_cls.assert_called_once_with(
        brand_idx=mock_brand_idx.return_value,
        generic_idx=mock_generic_idx.return_value,
        substance_idx=mock_substance_idx.return_value,
        product_idx=mock_product_idx.return_value,
        substance_variations_idx=mock_variations_idx.return_value,
        form_idx=mock_form_idx.return_value,
        route_idx=mock_route_idx.return_value,
    )
    mock_mapper_cls.assert_called_once_with(mock_query_engine_cls.return_value)
    self.assertEqual(mapper, mock_mapper_cls.return_value)

  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.vaccine.index.snomed_vaccine_index.SnomedVaccineIndex.from_index_file'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.snomed.vaccine.query.snomed_vaccine_query_engine.SnomedVaccineQueryEngine'
  )
  @unittest.mock.patch(
      'src.document_to_fhir.core.medical_coding.mapper.terminology_mappers.snomed_vaccine_terminology_mapper.SnomedVaccineTerminologyMapper'
  )
  def test_create_snomed_vaccine_mapper(
      self,
      mock_mapper_cls,
      mock_query_engine_cls,
      mock_vaccine_idx,
  ):
    config = {
        'snomed_vaccine_kb_path': 'dummy_vaccine.tsv',
    }
    mapper = rest_server._create_snomed_vaccine_mapper(config)
    mock_vaccine_idx.assert_called_once_with('dummy_vaccine.tsv')
    mock_query_engine_cls.assert_called_once_with(
        vaccine_index=mock_vaccine_idx.return_value,
    )
    mock_mapper_cls.assert_called_once_with(mock_query_engine_cls.return_value)
    self.assertEqual(mapper, mock_mapper_cls.return_value)

  def test_create_snomed_vaccine_mapper_missing_path(self):
    for invalid_config in [
        {},
        {'snomed_vaccine_kb_path': ''},
        {'snomed_vaccine_kb_path': None},
    ]:
      with self.subTest(invalid_config=invalid_config):
        with self.assertRaisesRegex(
            ValueError, 'snomed_vaccine_kb_path is required in config\\.'
        ):
          rest_server._create_snomed_vaccine_mapper(invalid_config)

  @unittest.mock.patch(
      'src.rest_server._create_loinc_mapper'
  )
  @unittest.mock.patch(
      'src.rest_server._create_snomed_drug_mapper'
  )
  @unittest.mock.patch(
      'src.rest_server._create_snomed_vaccine_mapper'
  )
  def test_get_mapper_registry(
      self, mock_vaccine_mapper, mock_snomed_mapper, mock_loinc_mapper
  ):
    mock_loinc = unittest.mock.MagicMock()
    mock_snomed = unittest.mock.MagicMock()
    mock_vaccine = unittest.mock.MagicMock()
    mock_loinc_mapper.return_value = mock_loinc
    mock_snomed_mapper.return_value = mock_snomed
    mock_vaccine_mapper.return_value = mock_vaccine

    config = {'dummy': 'config'}
    registry = rest_server._get_mapper_registry(config)

    mock_loinc_mapper.assert_called_once_with(config)
    mock_snomed_mapper.assert_called_once_with(config)
    mock_vaccine_mapper.assert_called_once_with(config)
    self.assertEqual(
        registry,
        {
            rest_server.resources.LabTest: mock_loinc,
            rest_server.medical_documents.RadiologyReport: mock_loinc,
            rest_server.abdm_medical_documents.AbdmRadiologyReport: mock_loinc,
            rest_server.resources.Medication: mock_snomed,
            rest_server.resources.Immunization: mock_vaccine,
        },
    )

  def test_document_type_mapping_immunization(self):
    self.assertIn('IMMUNIZATION', rest_server._DOCUMENT_TYPE_MAPPING)
    mapping = rest_server._DOCUMENT_TYPE_MAPPING['IMMUNIZATION']
    self.assertIs(
        mapping['extractor_class'],
        rest_server.immunization_extractor.ImmunizationExtractor,
    )
    self.assertIs(
        mapping['schema'],
        rest_server.abdm_medical_documents.AbdmImmunizationRecord,
    )
    self.assertIs(
        mapping['fhir_generator_class'],
        rest_server.abdm_immunization_fhir_generator.AbdmImmunizationFhirGenerator,
    )

  @unittest.mock.patch(
      'src.rest_server._get_mapper_registry'
  )
  @unittest.mock.patch(
      'src.rest_server.medical_document_standardizer.MedicalDocumentStandardizer'
  )
  def test_create_standardizer_map_immunization(
      self,
      mock_standardizer_cls,
      mock_get_mapper_registry,
  ):
    mock_registry = {
        rest_server.resources.Immunization: unittest.mock.MagicMock()
    }
    mock_get_mapper_registry.return_value = mock_registry
    mock_client = unittest.mock.MagicMock()

    mock_extractor_cls = unittest.mock.MagicMock()
    mock_fhir_gen_cls = unittest.mock.MagicMock()

    with unittest.mock.patch.dict(
        rest_server._DOCUMENT_TYPE_MAPPING,
        {
            'IMMUNIZATION': {
                'extractor_class': mock_extractor_cls,
                'schema': rest_server.abdm_medical_documents.AbdmImmunizationRecord,
                'fhir_generator_class': mock_fhir_gen_cls,
            }
        },
    ):
      config = {'supported_types': ['IMMUNIZATION']}
      standardizer_map = rest_server._create_standardizer_map(
          config, mock_client
      )

      mock_extractor_cls.assert_called_once_with(
          mock_client, rest_server.abdm_medical_documents.AbdmImmunizationRecord
      )
      mock_fhir_gen_cls.assert_called_once_with()
      mock_standardizer_cls.assert_called_once_with(
          extractor=mock_extractor_cls.return_value,
          mapper_registry=mock_registry,
          fhir_generator=mock_fhir_gen_cls.return_value,
      )
      self.assertIn(
          rest_server.document_types.MedicalDocumentType.IMMUNIZATION,
          standardizer_map,
      )
      self.assertEqual(
          standardizer_map[
              rest_server.document_types.MedicalDocumentType.IMMUNIZATION
          ],
          mock_standardizer_cls.return_value,
      )


if __name__ == '__main__':
  absltest.main()

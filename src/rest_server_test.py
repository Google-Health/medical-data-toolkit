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
    }

    with unittest.mock.patch(
        'src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte.index.AnalytesIndex.from_csv'
    ) as mock_from_csv, unittest.mock.patch(
        'src.document_to_fhir.core.medical_coding.loinc.axes_kb.system.mapper.SpecimenToSystemMapper.from_csv'
    ), unittest.mock.patch(
        'src.document_to_fhir.core.medical_coding.loinc.axes_kb.property.mapper.UnitToPropertyMapper.from_csv'
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
      {'LITELLM_API_KEY': 'fake_key'}
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


if __name__ == '__main__':
  absltest.main()

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
"""REST server for the MDDAS."""

from collections.abc import Mapping, Sequence
import io
import os
import threading
from typing import Any

from absl import app as absl_app
from absl import flags
from absl import logging
import gunicorn.app.base
import PIL.Image
import pydantic
import pypdfium2
import yaml

import flask
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_medical_documents
from src.document_to_fhir.core.classification import classifier
from src.document_to_fhir.core.extraction.extractors import lab_report_extractor
from src.document_to_fhir.core.fhir.abdm import abdm_lab_report_fhir_generator
from src.document_to_fhir.core.medical_coding.loinc import query as loinc_query
from src.document_to_fhir.core.medical_coding.loinc.axes_kb.core_analyte import index as analyte_index
from src.document_to_fhir.core.medical_coding.mapper.terminology_mappers import loinc_terminology_mapper
from src.document_to_fhir.core.orchestrator import composite_document_standardizer
from src.document_to_fhir.core.orchestrator import medical_document_standardizer


flask_app = flask.Flask(__name__)
_CONFIG_FILE = flags.DEFINE_string(
    'config_file', None, 'Path to the YAML configuration file.'
)

_DOCUMENT_TYPE_MAPPING = {
    'LABORATORY_REPORT': {
        'extractor_class': lab_report_extractor.LabReportExtractor,
        'schema': abdm_medical_documents.AbdmLabReport,
        'fhir_generator_class': (
            abdm_lab_report_fhir_generator.AbdmLabReportFhirGenerator
        ),
    },
}

_EMPTY_BODY_ERROR = 'Message body is empty.'
_NOT_PDF_OR_IMAGE_BYTES_ERROR = (
    'Message body does not encode PDF or image bytes.'
)

_composite_standardizer: (
    composite_document_standardizer.CompositeDocumentStandardizer | None
) = None
_standardizer_lock = threading.Lock()


def _init_fork_module_state():
  """Re-initializes module state in forked child processes."""
  global _composite_standardizer
  global _standardizer_lock
  _composite_standardizer = None
  _standardizer_lock = threading.Lock()


def get_composite_standardizer() -> (
    composite_document_standardizer.CompositeDocumentStandardizer | None
):
  """Returns the CompositeDocumentStandardizer instance, initializing it if necessary.

  The path to the configuration file is determined by the --config_file flag.

  Raises:
    ValueError: If no configuration file is specified by flag.
    FileNotFoundError: If the specified configuration file does not exist.
  """
  global _composite_standardizer
  try:
    config_file = _CONFIG_FILE.value
  except flags.UnparsedFlagAccessError:
    config_file = None

  if not config_file:
    raise ValueError("Configuration file not specified via flag.")

  if not os.path.exists(config_file):
    raise FileNotFoundError(f"Configuration file not found at: {config_file}")

  with _standardizer_lock:
    if _composite_standardizer is None and config_file:
      with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
      classifier_client = _create_llm_client(
          config.get('classifier_llm_client', {})
      )
      extractor_client = _create_llm_client(
          config.get('extractor_llm_client', {})
      )
      standardizer_map = _create_standardizer_map(config, extractor_client)
      classifier_inst = classifier.MultiDocumentClassifier(
          client=classifier_client
      )
      _composite_standardizer = (
          composite_document_standardizer.CompositeDocumentStandardizer(
              classifier=classifier_inst,
              standardizers=standardizer_map,
              document_standardization_policy=document_types.DocumentStandardizationPolicy(
                  config.get('document_standardization_policy', 'ACCEPT_ALL')
              ),
              attach_document_to_bundle=config.get(
                  'ATTACH_DOCUMENT_TO_BUNDLE', False
              ),
          )
      )
  return _composite_standardizer


def _flask_error(msg: str, status_code: int) -> flask.Response:
  return flask.make_response(flask.jsonify({'error': msg}), status_code)


def _document_to_fhir(file_bytes: bytes, mime_type: str) -> flask.Response:
  """Returns the FHIR representation of a document."""
  composite_standardizer = get_composite_standardizer()
  if composite_standardizer is None:
    return _flask_error('CompositeDocumentStandardizer not initialized', 500)

  try:
    result = composite_standardizer.standardize(file_bytes, mime_type=mime_type)
    return flask.make_response(flask.jsonify(result.model_dump()), 200)
  except (
      ValueError,
      pydantic.ValidationError,
      model_client.ResponseParsingError,
  ) as e:
    logging.exception(
        'Document standardization failed with validation/parsing error: %s', e
    )
    return _flask_error(f'Document standardization failed: {e}', 422)
  except Exception as e:  # pylint: disable=broad-except
    logging.exception('Unexpected error during document standardization: %s', e)
    return _flask_error(f'Failed to convert document to FHIR: {e}', 500)


def _pdf_page_count(file_bytes: bytes) -> int:
  """Returns the number of pages in a PDF file, or -1 if the file is not a PDF."""
  try:
    # Attempt to open the document
    doc = pypdfium2.PdfDocument(file_bytes)
    # Optional: Perform an additional check like getting the page count
    page_count = len(doc)
    doc.close()
    return page_count
  except Exception:  # pylint: disable=broad-except
    # pypdfium2 raises a generic exception or PdfiumError if loading fails
    # not a pdf file
    return -1


def _is_image(file_bytes: bytes) -> bool:
  """Returns whether the file is an image."""
  try:
    # Attempt to open the document
    with io.BytesIO(file_bytes) as image_bytes:
      with PIL.Image.open(image_bytes) as img:
        img.verify()
      return True
  except Exception:  # pylint: disable=broad-except
    return False


def _get_image_mime_type(file_bytes: bytes) -> str:
  """Returns the mime type of the image, defaulting to image/png."""
  try:
    with io.BytesIO(file_bytes) as image_bytes:
      with PIL.Image.open(image_bytes) as img:
        return f'image/{img.format.lower()}'
  except Exception:  # pylint: disable=broad-except
    return 'image/png'


def _validate_or_infer_content_type(
    data: bytes, header_type: str | None
) -> str:
  """Validates the header content type or infers it from data.

  Args:
    data: The raw bytes of the document.
    header_type: The Content-Type header value, or None.

  Returns:
    The determined mime type string.

  Raises:
    ValueError: If validation fails or type cannot be determined.
  """
  if header_type:
    if header_type == 'application/pdf':
      if _pdf_page_count(data) > 0:
        return header_type
      raise ValueError('Data is not a valid PDF but header claims it is.')
    elif header_type.startswith('image/'):
      if _is_image(data):
        return header_type
      raise ValueError('Data is not a valid image but header claims it is.')
    else:
      raise ValueError(f'Unsupported Content-Type: {header_type}')

  if _pdf_page_count(data) > 0:
    return 'application/pdf'
  if _is_image(data):
    return _get_image_mime_type(data)

  raise ValueError(_NOT_PDF_OR_IMAGE_BYTES_ERROR)


@flask_app.route(
    '/document_to_fhir', methods=['POST'], endpoint='document_to_fhir'
)
def document_to_fhir() -> flask.Response:
  """Document to FHIR endpoint for the server (flask.Response)."""
  binary_data = flask.request.get_data()
  if not binary_data:
    return _flask_error(_EMPTY_BODY_ERROR, 400)
  header_type = flask.request.headers.get('Content-Type')
  try:
    content_type = _validate_or_infer_content_type(binary_data, header_type)
  except ValueError as e:
    return _flask_error(str(e), 400)

  return _document_to_fhir(binary_data, content_type)


@flask_app.route('/', methods=['GET', 'POST'], endpoint='healthcheck')
def healthcheck() -> flask.Response:
  """Healthcheck endpoint for the server (flask.Response)."""
  return 'Healthcheck OK'


class GunicornApplication(gunicorn.app.base.BaseApplication):
  """gunicorn WSGI wrapper for the Flask server.

  More info: https://docs.gunicorn.org/en/stable/custom.html#custom-application
  """

  def __init__(self, app: flask.Flask):
    self.application = app
    super().__init__()

  def load_config(self):
    num_workers = os.cpu_count() or 1
    self.cfg.set('worker_class', 'gthread')
    self.cfg.set('workers', str(num_workers))
    self.cfg.set('threads', '5')
    self.cfg.set('bind', 'unix:/tmp/gunicorn.sock')
    self.cfg.set('accesslog', '-')
    self.cfg.set(
        'access_log_format', '%(u)s "%(r)s" %(s)s "%(f)s" "%({body}i)s"'
    )

  def load(self) -> flask.Flask:
    return self.application


def _create_llm_client(config: Mapping[str, Any]) -> model_client.LLMClient:
  """Creates an LLMClient based on configuration."""
  client_type = config.get('type')
  params = config.get('parameters', {})

  api_key = params.get('api_key')
  api_key_env = params.get('api_key_env')
  if api_key_env:
    api_key = os.environ.get(api_key_env)

  if client_type == 'GeminiClient':
    if 'model' not in params:
      raise ValueError(
          f"'model' is required for {client_type} in configuration"
      )
    return model_client.GeminiClient(
        api_key=api_key,
        model=params['model'],
        verbose=params.get('verbose', False),
    )
  elif client_type == 'GemmaClient':
    if 'model' not in params:
      raise ValueError(
          f"'model' is required for {client_type} in configuration"
      )
    return model_client.GemmaClient(
        api_key=api_key,
        model=params['model'],
        verbose=params.get('verbose', False),
    )
  elif client_type == 'LiteLLMClient':
    if 'model' not in params:
      raise ValueError(
          f"'model' is required for {client_type} in configuration"
      )
    return model_client.LiteLLMClient(
        model=params['model'],
        api_base=params.get('api_base'),
        api_key=api_key,
        temperature=params.get('temperature', 0.0),
        verbose=params.get('verbose', False),
        timeout=params.get('timeout', 300.0),
        max_retries=params.get('max_retries', 3),
        supports_pdf=params.get('supports_pdf', False),
    )
  else:
    raise ValueError(f'Unsupported LLMClient type: {client_type}')


def _create_standardizer_map(
    config: Mapping[str, Any], client: model_client.LLMClient
) -> Mapping[
    document_types.MedicalDocumentType,
    medical_document_standardizer.MedicalDocumentStandardizer,
]:
  """Creates the map of standardizers based on configuration."""
  standardizer_map = {}
  loinc_analaytes_index_csv_path = config.get('loinc_analaytes_index_csv_path')

  if not loinc_analaytes_index_csv_path:
    raise ValueError(
        'loinc_analaytes_index_csv_path is required in config for terminology'
        ' mapping.'
    )

  idx = analyte_index.AnalytesIndex.from_csv(loinc_analaytes_index_csv_path)
  query_engine = loinc_query.LoincQueryEngine(idx)
  mapper = loinc_terminology_mapper.LoincTerminologyMapper(query_engine)
  mapper_registry = {resources.LabTest: mapper}

  supported_types = config.get('supported_types', [])
  for doc_type_str in supported_types:
    if doc_type_str not in _DOCUMENT_TYPE_MAPPING:
      raise ValueError(
          f'Document type {doc_type_str} is not supported or mapped in code.'
      )

    mapping = _DOCUMENT_TYPE_MAPPING[doc_type_str]
    doc_type = document_types.MedicalDocumentType(doc_type_str)

    extractor_class = mapping['extractor_class']
    extractor = extractor_class(client, mapping['schema'])

    generator_class = mapping.get('fhir_generator_class')
    fhir_generator = generator_class() if generator_class else None  # pytype: disable=missing-parameter

    standardizer = medical_document_standardizer.MedicalDocumentStandardizer(
        extractor=extractor,
        mapper_registry=mapper_registry,
        fhir_generator=fhir_generator,
    )
    standardizer_map[doc_type] = standardizer

  return standardizer_map


def main(unused_argv: Sequence[str]):
  """Main entry point for the server."""
  GunicornApplication(flask_app).run()


# os.register_at_fork must be called before Gunicorn forks child processes.
# This is required to re-initialize module-level state (e.g. threading locks)
# in each child process via _init_fork_module_state.
os.register_at_fork(after_in_child=_init_fork_module_state)

if __name__ == '__main__':
  absl_app.run(main)

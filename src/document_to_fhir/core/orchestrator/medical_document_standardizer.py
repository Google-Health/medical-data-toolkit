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
"""Standardizes medical document images into structured data and FHIR."""

from typing import Any

from absl import logging
import pydantic

from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.core.extraction import medical_extractor
from src.document_to_fhir.core.fhir import fhir_generator as fhir_generator_lib
from src.document_to_fhir.core.medical_coding.mapper import terminology_mapper


def enrich_terminology_recursive(
    model: Any,
    mapper_registry: dict[
        type[resources.MedicalData], terminology_mapper.ITerminologyMapper
    ],
):
  """Walks the Pydantic tree to apply terminology mapping."""
  if model is None:
    return
  if isinstance(model, resources.MedicalData):
    mapper = mapper_registry.get(type(model))
    if mapper:
      mapper.map_inplace(model)

  for _, value in model:
    if isinstance(value, list):
      for item in value:
        if isinstance(item, (pydantic.BaseModel, resources.MedicalData)):
          enrich_terminology_recursive(item, mapper_registry)
    elif isinstance(value, (pydantic.BaseModel, resources.MedicalData)):
      enrich_terminology_recursive(value, mapper_registry)


class MedicalDocumentStandardizer:
  """Orchestrates the lifecycle of a single medical document segment."""

  def __init__(
      self,
      extractor: medical_extractor.MedicalExtractor,
      mapper_registry: dict[
          type[resources.MedicalData], terminology_mapper.ITerminologyMapper
      ],
      fhir_generator: fhir_generator_lib.IFhirGenerator,
  ):
    self.extractor = extractor
    self.mapper_registry = mapper_registry
    self.fhir_generator = fhir_generator

  def standardize(
      self, images: list[bytes], mime_type: str = "image/png"
  ) -> tuple[resources.MedicalData, Any]:
    """Standardizes medical document images into structured data and FHIR.

    Args:
      images: A list of image bytes to extract data from.
      mime_type: The MIME type of the images.

    Returns:
      A tuple of (medical_document, fhir_bundle).
    """
    # 1. Extraction (LLM Call).
    medical_document = self.extractor.extract(images, mime_type=mime_type)

    # 2. Terminology Enrichment.
    enrich_terminology_recursive(medical_document, self.mapper_registry)

    # 3. FHIR Generation.
    fhir_bundle = self.fhir_generator.generate_fhir(medical_document)

    return medical_document, fhir_bundle

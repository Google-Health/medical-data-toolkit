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
"""Standardized composite medical document with context."""

import threading
from typing import Any, Optional

import pydantic

from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import medical_documents


class DocumentSegment(pydantic.BaseModel):
  reasoning: str = pydantic.Field(description="Reasoning for the prediction.")
  handwritten_content_percent: int = pydantic.Field(
      default=0,
      description=(
          "Percentage of handwritten content in the document segment in the"
          " range 0-100%."
      ),
  )
  document_type: document_types.MedicalDocumentType = pydantic.Field(
      description="Predicted document type."
  )
  start_page: int = pydantic.Field(
      description="Start page number of the document."
  )
  end_page: int = pydantic.Field(description="End page number of the document.")

  @pydantic.field_validator("document_type", mode="before")
  @classmethod
  def validate_document_type(cls, v: Any) -> Any:
    """Safely coerces to MedicalDocumentType falling back to NON_MEDICAL if unknown."""
    try:
      return document_types.MedicalDocumentType(v)
    except ValueError:
      return document_types.MedicalDocumentType.NON_MEDICAL


class CompositeDocument(pydantic.BaseModel):
  segments: list[DocumentSegment] = pydantic.Field(default_factory=list)


class StandardizedMedicalDocumentWithContext(pydantic.BaseModel):
  """Represents a standardized medical document with its context.

  This model holds a medical document, its type, page range, and optionally
  a FHIR bundle representation. It includes a validator to hydrate the
  `medical_document` field from raw dictionary data based on the
  `document_type`.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  document_type: Optional[document_types.MedicalDocumentType] = None
  start_page: Optional[int] = None
  end_page: Optional[int] = None
  # We hint as Any to prevent Pydantic from trying to validate
  # until OUR logic runs in the validator.
  medical_document: Optional[Any] = None

  # We can store the FHIR Bundle as a JSON here for easy serialization
  fhir_bundle: Optional[dict[str, Any]] = pydantic.Field(default=None)

  @pydantic.model_validator(mode="before")
  @classmethod
  def hydrate_medical_document(cls, data: Any) -> Any:
    if not isinstance(data, dict):
      return data

    med_doc_raw = data.get("medical_document")
    doc_type = data.get("document_type")
    fhir_bundle = data.get("fhir_bundle")

    # Logic: If we have a dict and a known type, convert to Pydantic object
    if med_doc_raw and isinstance(med_doc_raw, dict) and doc_type:
      match doc_type:
        case document_types.MedicalDocumentType.LABORATORY_REPORT:
          model_class = medical_documents.LabReport
        case _:
          model_class = None

      if model_class:
        # This performs the specific validation for LabReport/Prescription
        data["medical_document"] = model_class.model_validate(med_doc_raw)

    if fhir_bundle and isinstance(fhir_bundle, dict):
      data["fhir_bundle"] = fhir_bundle
    return data


class PipelineMetadata(pydantic.BaseModel):
  latency_ms: dict[str, Any]
  token_usage: Optional[dict[str, Any]] = None


class StandardizedCompositeMedicalDocumentWithContext(pydantic.BaseModel):
  """A collection of StandardizedMedicalDocumentWithContext objects.

  This class is used to group multiple standardized medical documents,
  providing methods to manage and access them.
  """

  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

  standardized_medical_documents: list[
      StandardizedMedicalDocumentWithContext
  ] = pydantic.Field(default_factory=list)
  metadata: Optional[PipelineMetadata] = None
  lock: threading.Lock = pydantic.Field(
      default_factory=threading.Lock, exclude=True
  )

  @property
  def n_documents(self) -> int:
    return len(self.standardized_medical_documents)

  @property
  def n_pages(self) -> int:
    return sum([
        doc.end_page - doc.start_page + 1
        for doc in self.standardized_medical_documents
    ])

  def add_standardized_document(
      self, doc: StandardizedMedicalDocumentWithContext
  ):
    with self.lock:
      self.standardized_medical_documents.append(doc)

  def sort_documents_by_page_number(self):
    self.standardized_medical_documents.sort(key=lambda doc: doc.start_page)

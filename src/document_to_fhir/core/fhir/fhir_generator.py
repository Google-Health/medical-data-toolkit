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
"""Abstract base class for FHIR generators."""

import abc

from google.fhir.r4.proto.core.resources import bundle_and_contained_resource_pb2 as fhir_pb2
from src.document_to_fhir.common.schema import medical_documents


ABC = abc.ABC
abstractmethod = abc.abstractmethod


class IFhirGenerator(ABC):
  """Interface for FHIR generators.

  This class defines the interface for FHIR generators, which are responsible
  for converting medical documents into FHIR bundles.
  """

  def __init__(self, version: str = "", fhir_profile: str = ""):
    """Initializes the FHIR generator.

    Args:
      version: The version of the FHIR generator.
      fhir_profile: The FHIR profile to use for generation.
    """
    pass

  @abstractmethod
  def generate_fhir(
      self, medical_document: medical_documents.MedicalDocument
  ) -> fhir_pb2.Bundle:
    """Generates a FHIR bundle from a medical document.

    Args:
      medical_document: The medical document to convert.

    Returns:
      A FHIR bundle representing the medical document.
    """
    pass

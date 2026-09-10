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
"""Base classes and protocols for SNOMED vaccine query engines."""

import abc
from typing import Any

import pydantic

from src.document_to_fhir.common.schema import resources


class VaccineResult(pydantic.BaseModel):
  """Represents a candidate SNOMED vaccine record returned by query engines."""

  code: str | None = None
  display_name: str | None = None
  code_system: str | None = None


class BaseVaccineQueryEngine(abc.ABC):
  """Abstract base class for vaccine query engines."""

  @abc.abstractmethod
  def query(
      self, immunization: resources.Immunization, *args: Any, **kwargs: Any
  ) -> VaccineResult:
    """Queries vaccine codes based on immunization criteria.

    Args:
      immunization: Immunization data containing vaccine name and demographics.
      *args: Additional positional arguments.
      **kwargs: Additional keyword arguments.

    Returns:
      VaccineResult containing matching SNOMED code and display name.
    """


class BaseVaccineQuery(abc.ABC):
  """Base class for vaccine queries."""

  def __init__(self, engine: BaseVaccineQueryEngine):
    """Initializes the vaccine query with the query engine."""
    self.engine = engine

  @abc.abstractmethod
  def query(
      self,
      immunization: resources.Immunization,
      verbose: bool = False,
      **kwargs: Any,
  ) -> tuple[str | None, str | None]:
    """Maps an immunization to a SNOMED vaccine code.

    Args:
      immunization: Immunization data containing vaccine name and demographics.
      verbose: Whether to output verbose query details.
      **kwargs: Additional keyword arguments.

    Returns:
      A tuple of (code, display_name) or (None, None) if not found.
    """

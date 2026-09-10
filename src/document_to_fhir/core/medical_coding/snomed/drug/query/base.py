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
"""Base classes and protocols for SNOMED drug query engines."""

import abc
import pydantic
from src.document_to_fhir.common.schema import resources


class DrugResult(pydantic.BaseModel):
  """Represents a candidate SNOMED drug record returned by query engines."""

  code: str | None = None
  display_name: str | None = None
  code_system: str | None = None


class BaseDrugQueryEngine(abc.ABC):
  """Abstract base class for drug query engines."""

  @abc.abstractmethod
  def query(
      self, medication: resources.Medication, *args, **kwargs
  ) -> DrugResult:
    """Queries drug codes based on medication criteria."""


class BaseDrugQuery(abc.ABC):
  """Base class for drug queries."""

  def __init__(self, engine: BaseDrugQueryEngine):
    self.engine = engine

  @abc.abstractmethod
  def query(
      self,
      medication: resources.Medication,
      verbose: bool = False,
      **kwargs,
  ) -> tuple[str | None, str | None]:
    """Maps a medication to a SNOMED drug code."""

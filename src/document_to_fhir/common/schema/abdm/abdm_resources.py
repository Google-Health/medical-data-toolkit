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
"""Pydantic models for representing ABDM specific medical document resources."""

from typing import Optional

import pydantic

from src.document_to_fhir.common.schema import resources


class AbdmPractitionerIdentifiers(resources.BasePractitionerIdentifiers):
  """ABDM identifiers for a practitioner."""

  md: Optional[str] = pydantic.Field(
      default=None,
      description='Medical Doctor Registration or License number (ABDM).',
  )


class AbdmOrganizationIdentifiers(resources.BaseOrganizationIdentifiers):
  """ABDM identifiers for an organization."""

  prn: Optional[str] = pydantic.Field(
      default=None, description='Provider Number.'
  )

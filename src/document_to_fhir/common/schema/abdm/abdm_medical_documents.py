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
"""Pydantic models for representing ABDM specific medical documents."""

from typing import Optional

import pydantic

from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources


class AbdmLabReport(medical_documents.LabReport):
  """ABDM specific Laboratory Report."""

  service_provider: Optional[
      resources.Organization[abdm_resources.AbdmOrganizationIdentifiers]
  ] = pydantic.Field(default=None, description='Service provider details.')
  practitioner: Optional[
      resources.Practitioner[abdm_resources.AbdmPractitionerIdentifiers]
  ] = pydantic.Field(default=None, description='Clinical Practitioner details.')

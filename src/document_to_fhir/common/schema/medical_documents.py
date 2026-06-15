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
"""Pydantic models for representing medical documents."""

# Rationale for schema design:
# 1. Gemini Compatibility:
#    - Default values: Gemini API fails when schemas contain default values.
#      Pydantic's `Field(default=...)` is avoided.
#    - Optional syntax: `Optional[str]` is used instead of `str | None` as
#      suggested by Gemini's structured output documentation:
#      https://ai.google.dev/gemini-api/docs/structured-output
#    - Docstrings: Class docstrings were found to be included in the LLM prompt,
#      increasing complexity. Field descriptions are sufficient and are used
#      instead.
#
# 2. Flexibility vs. Structure with Optional:
#    - Models like `LabReport.service_provider` or `LabReport.practitioner` are
#      `Optional` because entire sections of data may be missing from a given
#      lab report PDF, and we want to allow Gemini to omit them if not found.
#    - However, if Gemini *does* identify and extract an entity (e.g., an
#      Organization), that entity must adhere to a minimal structure. For
#      example, `Organization.name` is mandatory (`str`), ensuring that if an
#      organization is returned, it at least includes a name.
#    - Similarly, leaf fields within a structure (e.g.,
#      `PatientIdentifiers.mrn`) can be `Optional` if that specific piece of
#      data is often missing, even if the containing object
#      (`PatientIdentifiers`) is present.
#    - This approach provides flexibility for Gemini to handle missing data or
#      entire sections, while enforcing a core structure for entities that
#      *are* extracted.

import datetime
from typing import Annotated, Optional

import pydantic

from src.document_to_fhir.common.schema import resources


class MedicalDocument(resources.MedicalData):
  """Base class for all medical document types.

  Inherits schema customization from MedicalData.
  """


def _serialize_datetime_to_isoformat(
    dt: Optional[datetime.datetime],
) -> Optional[str]:
  if dt is None:
    return None
  # Ensure the datetime is in UTC.
  if dt.tzinfo is not None:
    dt = dt.astimezone(datetime.timezone.utc)
  else:
    dt = dt.replace(tzinfo=datetime.timezone.utc)
  return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


CustomDateTime = Annotated[
    datetime.datetime,
    pydantic.PlainSerializer(_serialize_datetime_to_isoformat, return_type=str),
]


class LabReport(MedicalDocument):
  """Medical Laboratory Report for a patient."""

  sample_collection_time: Optional[CustomDateTime] = pydantic.Field(
      default=None, description='Sample collection time of the lab report.'
  )
  patient: resources.Patient = pydantic.Field(description='Patient details.')
  service_provider: Optional[
      resources.Organization[resources.BaseOrganizationIdentifiers]
  ] = pydantic.Field(default=None, description='Service provider details.')
  practitioner: Optional[
      resources.Practitioner[resources.BasePractitionerIdentifiers]
  ] = pydantic.Field(default=None, description='Clinical Practitioner details.')
  lab_tests: list[resources.LabTest] = pydantic.Field(
      description='List of lab tests.'
  )

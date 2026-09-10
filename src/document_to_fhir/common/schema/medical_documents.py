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
from pydantic import json_schema

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
      default=None,
      description=(
          'Sample collection date and time of the lab report, often labeled as'
          " 'Collected On', 'Collection Date/Time', or 'Sample Date'. Do not"
          " confuse with registration ('Registered On') or reporting"
          " ('Reported On') timestamps."
      ),
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


class RadiologyReport(MedicalDocument):
  """Medical Radiology/Imaging Report for a patient."""

  study_date: Optional[CustomDateTime] = pydantic.Field(
      default=None,
      description=(
          'Date and time of the imaging study, often labeled as'
          " 'Study Date', 'Date of Study', 'Exam Date', 'Examination Date', or"
          " 'Investigation Date'. Do not confuse with reporting"
          " ('Reported On', 'Report Date') or registration"
          " ('Registered On') timestamps."
      ),
  )
  patient: resources.Patient = pydantic.Field(description='Patient details.')
  service_provider: Optional[
      resources.Organization[resources.BaseOrganizationIdentifiers]
  ] = pydantic.Field(default=None, description='Service provider details.')
  practitioner: Optional[
      resources.Practitioner[resources.BasePractitionerIdentifiers]
  ] = pydantic.Field(
      default=None,
      description='Clinical Practitioner details (e.g., Radiologist).',
  )

  modality_type: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Imaging modality type only (e.g., CT, MR, XR, US, MG, NM, PT, RF).'
          ' For compound procedures like CTA, extract the base modality "CT"'
          ' here and the technique "Angio" in modality_subtype.'
      ),
  )
  modality_subtype: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Imaging modality subtype or specific technique (e.g., Angio, Doppler,'
          ' Tomosynthesis, SPECT, Portable). For example, for CTA or MRA,'
          ' extract "Angio" here.'
      ),
  )

  region_imaged: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Anatomic region and specific imaging focus if present (e.g., Head,'
          ' Chest, Spine, Cervical Spine, Lumbar Spine, Knee, Abdomen and'
          ' Pelvis, Wrist).'
      ),
  )
  contrast_used: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Contrast administration and timing (e.g., "W contrast", "WO'
          ' contrast", "W and WO contrast", "with IV contrast").'
      ),
  )
  view_type: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Radiographic view projection or view count/aggregation (e.g., "PA",'
          ' "AP", "Lateral", "PA and Lateral", "Single view", "2 Views",'
          ' "3 Views").'
      ),
  )
  laterality: Optional[str] = pydantic.Field(
      default=None,
      description='Laterality (e.g., Left, Right, Bilateral, Unspecified).',
  )
  procedure_loinc_code: Annotated[
      Optional[str], json_schema.SkipJsonSchema()
  ] = pydantic.Field(
      default=None,
      description='LOINC code for the specific imaging procedure/study.',
  )
  procedure_loinc_common_name: Annotated[
      Optional[str], json_schema.SkipJsonSchema()
  ] = pydantic.Field(
      default=None,
      description='LOINC Long Common Name for the imaging procedure/study.',
  )
  exam_name: str = pydantic.Field(
      description=(
          'Name of the examination (e.g., "CT Head and Neck WO contrast").'
      )
  )
  findings: Optional[str] = pydantic.Field(
      default=None, description='Detailed findings of the radiology study.'
  )
  conclusion: Optional[str] = pydantic.Field(
      default=None, description='Summary or clinical impression of the study.'
  )
  conclusion_code: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(
          default=None,
          description='Coded conclusion (e.g. SNOMED code for the diagnosis).',
      )
  )


class Prescription(MedicalDocument):
  """Medical Prescription for a patient."""

  prescription_date: Optional[CustomDateTime] = pydantic.Field(
      default=None,
      description=(
          'Date and time of the prescription, often labeled as'
          " 'Date', 'Prescription Date', 'Prescribed On', or"
          " 'Consultation Date'. Do not confuse with follow-up or review"
          " dates ('Review On', 'Next Visit') or dispensing timestamps."
      ),
  )
  diagnosis: Optional[list[str]] = pydantic.Field(
      default=None, description='List of diagnoses of the patient.'
  )
  patient: resources.Patient = pydantic.Field(description='Patient details.')
  service_provider: Optional[
      resources.Organization[resources.BaseOrganizationIdentifiers]
  ] = pydantic.Field(default=None, description='Service provider details.')
  practitioner: Optional[
      resources.Practitioner[resources.BasePractitionerIdentifiers]
  ] = pydantic.Field(default=None, description='Clinical Practitioner details.')
  medications: list[resources.Medication] = pydantic.Field(
      description='List of medications.'
  )


class ImmunizationRecord(MedicalDocument):
  """Immunization record for a patient."""

  patient: resources.Patient = pydantic.Field(description='Patient details.')
  service_provider: Optional[
      resources.Organization[resources.BaseOrganizationIdentifiers]
  ] = pydantic.Field(default=None, description='Service provider details.')
  practitioner: Optional[
      resources.Practitioner[resources.BasePractitionerIdentifiers]
  ] = pydantic.Field(default=None, description='Clinical Practitioner details.')
  immunizations: list[resources.Immunization] = pydantic.Field(
      description='List of immunizations.'
  )

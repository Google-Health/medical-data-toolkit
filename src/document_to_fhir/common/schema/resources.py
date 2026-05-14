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
"""Pydantic models for representing medical document resources.

These models define the structure for various entities found in medical
documents, such as patients, practitioners, lab tests, and organizations.
"""

import datetime
import logging
from typing import Annotated, Any, Generic, Optional, TypeVar, Union, get_args, get_origin

import pydantic
from pydantic import json_schema


class MedicalData(pydantic.BaseModel):
  """Base class for all medical data models.

  This class provides custom JSON schema generation to remove 'default' values,
  which is necessary for compatibility with the google-generativeai library.
  It also handles relaxed datetime validation for optional fields.
  """

  @pydantic.model_validator(mode='before')
  @classmethod
  def clear_invalid_optional_datetimes(cls, data: Any) -> Any:
    """Clears invalid datetime strings for optional fields."""
    if not isinstance(data, dict):
      return data

    for field_name, field_info in cls.model_fields.items():
      annotation = field_info.annotation
      # Check if field's annotation is Optional[datetime.datetime] or
      # Optional[datetime.date]
      if get_origin(annotation) is Union:
        args = get_args(annotation)
        # Optional[T] is Union[T, NoneType]
        if type(None) in args:
          target_type = next(
              (a for a in args if not isinstance(a, type(None))), None
          )
          if target_type in (datetime.datetime, datetime.date):
            value = data.get(field_name)
            if value and isinstance(value, str):
              try:
                if target_type is datetime.datetime:
                  # Use pydantic's internal parsing if possible, or just a
                  # basic check. For simplicity, we just try to let pydantic
                  # validate it in a sub-try.
                  datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
                else:
                  datetime.date.fromisoformat(value)
              except (ValueError, TypeError):
                logging.warning(
                    'Clearing invalid %s for optional field %s: %s',
                    target_type.__name__,
                    field_name,
                    value,
                )
                data[field_name] = None
    return data

  @classmethod
  def __get_pydantic_json_schema__(
      cls, core_schema_obj: Any, handler: json_schema.GetJsonSchemaHandler
  ) -> json_schema.JsonSchemaValue:
    """Customizes Pydantic's JSON schema generation for MedicalData models.

    This method intercepts the schema generation process to remove the 'default'
    field from the resulting JSON schema. This is necessary because the
    google-generativeai library, when converting a JSON schema to its internal
    Protobuf representation for structured output (JSON mode), does not
    support the 'default' keyword. If 'default': None is present in the schema
    for optional fields, it causes a ValueError during the conversion process
    in the library.

    Args:
      core_schema_obj: The core schema being processed. This is an instance of
        pydantic_core.core_schema.CoreSchema, but we use Any due to import
        restrictions.
      handler: The handler for resolving references and generating the schema.
    """
    # The handler instance is callable to continue default schema generation.
    json_schema_obj = handler(core_schema_obj)

    # The handler instance also has the resolve_ref_schema method.
    json_schema_obj = handler.resolve_ref_schema(json_schema_obj)

    def remove_defaults_from_schema(schema_obj: Any) -> None:
      """Recursively traverses the schema and removes 'default' keys."""
      if isinstance(schema_obj, dict):
        schema_obj.pop('default', None)
        for value in schema_obj.values():
          remove_defaults_from_schema(value)
      elif isinstance(schema_obj, list):
        for item in schema_obj:
          remove_defaults_from_schema(item)

    # Remove 'default' keys from the generated schema before returning it.
    remove_defaults_from_schema(json_schema_obj)
    return json_schema_obj


class PatientIdentifiers(MedicalData):
  """Represents identifiers for a patient."""

  mr: Optional[str] = pydantic.Field(
      default=None, description='Medical Record Number.'
  )


class Patient(MedicalData):
  """Represents a patient."""

  name: str = pydantic.Field(description='Name of the patient.')
  identifiers: Optional[PatientIdentifiers] = pydantic.Field(
      default=None, description='Identifiers of the patient.'
  )
  dob: Optional[datetime.date] = pydantic.Field(
      default=None, description='Date of birth of the patient.'
  )
  gender: Optional[str] = pydantic.Field(
      default=None, description='Gender of the patient.'
  )


class LabTest(MedicalData):
  """Individual tests within a lab report."""

  core_analyte: str = pydantic.Field(
      description=(
          """The unique physiological entity or chemical substance measured, independent of specimen, method, or timing.
          Use Standard Scientific Capitalization (e.g., "HbA1c", "pH", "d-Dimer", "Free T4").
          Exclude specimen types (e.g., "Blood", "Serum"), methods (e.g., "PCR"), and generic nouns (e.g., "Level", "Count").
          DISAMBIGUATION: For names with slashes ("/"), keep the slash ONLY if it represents a mathematical ratio
          (e.g., "Albumin/Creatinine"); if it implies synonyms or subtypes, extract the specific scientific name
          (e.g., "PCV/Hematocrit" -> "Hematocrit", "Protein/Albumin" -> "Albumin")."""
      ),
  )
  name: str = pydantic.Field(description='The formal name of the lab test.')
  result: str = pydantic.Field(
      description='The actual value or description recorded for the test.'
  )
  unit: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'The unit of measurement (e.g., "mg/dL"). Use empty string if none.'
      ),
  )
  specimen: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'The type of biological sample (e.g., "Blood", "Serum", "Urine").'
      ),
  )
  method: Optional[str] = pydantic.Field(
      default=None, description='Method used for the lab test.'
  )
  panel_name: Optional[str] = pydantic.Field(
      default=None, description='Name of the lab panel.'
  )
  # This is temporary for now to capture the LOINC code of the lab panel.
  panel_loinc_code: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(
          default=None,
          description='LOINC code of the lab panel if panel is present.',
      )
  )
  reference_range: Optional[list[str]] = pydantic.Field(
      default=None,
      description=(
          'Reference ranges of the lab test. This should be an array of string'
          ' ranges (e.g., ["13.17 - 17 g/dL"])'
      ),
  )
  loinc_code: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(default=None, description='LOINC code of the lab test.')
  )
  loinc_common_name: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(
          default=None, description='LOINC common name of the lab test.'
      )
  )


class BasePractitionerIdentifiers(MedicalData):
  """Base represents identifiers for a practitioner."""


PractitionerIdentifiersT = TypeVar(
    'PractitionerIdentifiersT', bound=BasePractitionerIdentifiers
)


class Practitioner(MedicalData, Generic[PractitionerIdentifiersT]):
  """Represents a healthcare practitioner."""

  name: str = pydantic.Field(description='Name of the practitioner.')
  identifiers: Optional[PractitionerIdentifiersT] = pydantic.Field(
      default=None, description='Identifiers of the practitioner.'
  )
  qualification: Optional[str] = pydantic.Field(
      default=None, description='Qualification of the practitioner.'
  )


class BaseOrganizationIdentifiers(MedicalData):
  """Base represents identifiers for an organization."""


OrganizationIdentifiersT = TypeVar(
    'OrganizationIdentifiersT', bound=BaseOrganizationIdentifiers
)


class Address(MedicalData):
  """Represents a physical address."""

  street: Optional[str] = pydantic.Field(
      default=None, description='Street address.'
  )
  city: Optional[str] = pydantic.Field(
      default=None, description='City of the address.'
  )
  state: Optional[str] = pydantic.Field(
      default=None, description='State or province of the address.'
  )
  postal_code: Optional[str] = pydantic.Field(
      default=None, description='Postal or ZIP code of the address.'
  )
  country: Optional[str] = pydantic.Field(
      default=None, description='Country of the address.'
  )


class Contact(MedicalData):
  """Represents contact information."""

  phone: Optional[str] = pydantic.Field(
      default=None, description='Phone number.'
  )
  email: Optional[str] = pydantic.Field(
      default=None, description='Email address.'
  )
  fax: Optional[str] = pydantic.Field(default=None, description='Fax number.')
  website: Optional[str] = pydantic.Field(
      default=None, description='Website URL.'
  )


class Organization(MedicalData, Generic[OrganizationIdentifiersT]):
  """Represents an organization, such as a hospital or lab."""

  name: str = pydantic.Field(description='Name of the organization.')
  identifiers: Optional[OrganizationIdentifiersT] = pydantic.Field(
      default=None, description='Identifiers of the organization.'
  )
  address: Optional[Address] = pydantic.Field(
      default=None, description='Address of the organization.'
  )
  contact: Optional[Contact] = pydantic.Field(
      default=None, description='Contact information of the organization.'
  )

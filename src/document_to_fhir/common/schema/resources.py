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
import enum
import logging
import re
import typing
from typing import Annotated, Any, Generic, Optional, TypeVar, Union

import pydantic
from pydantic import json_schema


def normalize_unit(unit_str: str) -> str:
  """Normalizes informal/non-standard unit strings to standard SI/UCUM representations."""
  if not unit_str:
    return unit_str

  unit = unit_str.strip().lower()

  # Common unit mappings in prescriptions
  # Using (?<=\d|\b) instead of \b at start to handle squashed units like "50mg"
  mappings = [
      (r'(?<=\d)(gm(s)?|gram(s)?)\b|\b(gm(s)?|gram(s)?)\b', 'g'),
      (r'(?<=\d)(milligram(s)?)\b|\b(milligram(s)?)\b', 'mg'),
      (r'(?<=\d)(milliliter(s)?|ml(s)?)\b|\b(milliliter(s)?|ml(s)?)\b', 'ml'),
      (r'(?<=\d)(mcg|microgram(s)?)\b|\b(mcg|microgram(s)?)\b', 'mcg'),
      (r'(?<=\d)(tab(s)?|tablet(s)?)\b|\b(tab(s)?|tablet(s)?)\b', 'tablet'),
      (r'(?<=\d)(cap(s)?|capsule(s)?)\b|\b(cap(s)?|capsule(s)?)\b', 'capsule'),
      (r'(?<=\d)inj(ection(s)?)?\b|\binj(ection(s)?)?\b', 'injection'),
      (r'(?<=\d)(i\.?u\.?|units?)(?!\w)|\b(i\.?u\.?|units?)(?!\w)', 'iu'),
      (r'(?<=\d)(percent(age)?|pct)\b|\b(percent(age)?|pct)\b', '%'),
      (r'(?<=\d)(w\/v)\b|\b(w\/v)\b', 'w/v'),
      (r'(?<=\d)(v\/v)\b|\b(v\/v)\b', 'v/v'),
  ]

  for pattern, replacement in mappings:
    unit = re.sub(pattern, replacement, unit, flags=re.IGNORECASE)

  return unit


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
      if typing.get_origin(annotation) is Union:
        args = typing.get_args(annotation)
        # Optional[T] is Union[T, NoneType]
        if type(None) in args:
          target_type = next(
              (a for a in args if not isinstance(a, type(None))), None
          )
          # Handle Annotated[datetime.datetime, ...] or
          # Annotated[datetime.date, ...]
          if typing.get_origin(target_type) is Annotated:
            target_type = typing.get_args(target_type)[0]
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
                  try:
                    datetime.date.fromisoformat(value)
                  except ValueError:
                    parsed_dt = datetime.datetime.fromisoformat(
                        value.replace('Z', '+00:00')
                    )
                    data[field_name] = parsed_dt.date().isoformat()
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


class ReferenceRangeAppliesTo(str, enum.Enum):
  MALE = 'Male'
  FEMALE = 'Female'
  PREGNANCY_CONFIRMED = 'Pregnancy confirmed'


class AgeRange(MedicalData):
  low: Optional[float] = pydantic.Field(
      default=None, description='Lower limit of age (in years).'
  )
  high: Optional[float] = pydantic.Field(
      default=None, description='Upper limit of age (in years).'
  )


class ReferenceRange(MedicalData):
  """Represents a reference range for a lab test."""

  low: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Lower limit if numeric. Applicable for standard non-conditional'
          ' cases (e.g., "4.0" from "4.0-6.5", "5.0" from ">5") and for'
          ' Gender/Age conditional cases only (e.g., "4.5" from "Male'
          ' 4.5-6.0", "5.0" from "Adults >17 years 5-6"). Do NOT extract from'
          ' clinical interpretation/risk-level cues (e.g., "Pre-diabetic:'
          ' 5.7-6.4").'
      ),
  )
  high: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Upper limit if numeric. Applicable for standard non-conditional'
          ' cases (e.g., "6.5" from "4.0-6.5", "6.0" from "<=6") and for'
          ' Gender/Age conditional cases only (e.g., "6.0" from "Male'
          ' 4.5-6.0", "6.0" from "Adults >17 years 5-6"). Do NOT extract from'
          ' clinical interpretation/risk-level cues (e.g., "Pre-diabetic:'
          ' 5.7-6.4").'
      ),
  )
  text: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Contains all the reference range text, cues, and non-parseable'
          ' info. If multiple risk-level ranges (e.g., Non-diabetic,'
          ' Pre-diabetic, Diabetic) are present, even if they appear in'
          ' different parts of the report, e.g., one in the main table and'
          ' others in comments, they MUST be clubbed. Concatenate all texts'
          ' (standard range text and cues) separated by semicolon ";"'
          ' (e.g., "4 - 6.5; Non-diabetic: <= 5.6; Pre-diabetic: 5.7-6.4;'
          '  Diabetic: >= 6.5").'
      ),
  )


class LabTest(MedicalData):
  """Individual tests within a lab report."""

  core_analyte: str = pydantic.Field(
      description='Core Analyte as described in the prompt.',
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
  reference_range: Optional[list[ReferenceRange]] = pydantic.Field(
      default=None,
      description=(
          'List of reference ranges or clinical interpretation cues. Handles'
          ' standard intervals (4-6.5), thresholds (<5.7, >=6.5), and textual'
          ' categories (Prediabetes 5.7-6.4).'
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


class PatientType(str, enum.Enum):
  """Represents the age bracket of the patient."""
  ADULT = 'ADULT'
  PAEDIATRIC = 'PAEDIATRIC'
  UNKNOWN = 'UNKNOWN'


class Immunization(MedicalData):
  """Represents an individual immunization."""

  # Setting dosage to Optional[str] causes it to be omitted even when the value
  # is present in the prompt. Hence, it is kept as a mandatory field and a
  # validator is used to set it to "unknown" if not specified. Keeping
  # 'dosage' as the first field is very useful because it directs the model to
  # decouple the dosage from the vaccine name when they are present together in
  # the record, e.g. 'Hep A1' where '1' indicates the dosage.
  dosage: str = pydantic.Field(
      description=(
          'Type of dosage e.g. 1, 2, booster etc. Use numbers or'
          ' "booster". Set to "unknown" if not specified.'
      ),
  )
  vaccine_name: str = pydantic.Field(
      description=(
          'Standardized name of the vaccine without dosage specified. In the'
          ' event that there is a choice of vaccine variations, pick the first'
          ' one. For example, Hep A1 should be extracted as Hep A.'
      )
  )
  date: Optional[datetime.date] = pydantic.Field(
      default=None,
      description=(
          'Date of vaccine administration, often labeled as'
          " 'Date Given', 'Given On', 'Administered On', 'Date of"
          " Administration', or 'Vaccination Date'. Do not confuse with"
          " 'Due Date', 'Next Due Date', or vaccine expiration date."
      ),
  )
  patient_type: PatientType = pydantic.Field(
      default=PatientType.UNKNOWN,
      description=(
          'Type of patient based on age when the vaccine was administered. If'
          ' the patient age is 18 or above, classify as ADULT. If the patient'
          ' age is less than 18, classify as PAEDIATRIC. If the age is not'
          ' available, classify as UNKNOWN.'
      ),
  )
  vaccine_code: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(default=None, description='SNOMED-CT Code of the vaccine.')
  )
  vaccine_code_name: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(
          default=None,
          description='Name of the vaccine in the SNOMED-CT code system.',
      )
  )

  @pydantic.model_validator(mode='before')
  @classmethod
  def set_default_dosage(cls, data: Any) -> Any:
    if isinstance(data, dict):
      if not data.get('dosage'):
        data = dict(data)
        data['dosage'] = 'unknown'
    return data


class Dosage(MedicalData):
  """Represents dosage instructions for a medication."""

  instruction: Optional[str] = pydantic.Field(
      default=None,
      description='Timing or condition for dosage (e.g., "in the morning").',
  )
  dose_value: Optional[float] = pydantic.Field(
      default=None, description='Amount of dose.'
  )
  dose_unit: Optional[str] = pydantic.Field(
      default=None, description='Unit for dose amount (e.g., "ml", "TAB).'
  )
  frequency: Optional[int] = pydantic.Field(
      default=None,
      description=(
          'The number of times the medication should be taken per period.'
      ),
  )
  frequency_period: Optional[float] = pydantic.Field(
      default=None,
      description=(
          'The period over which the medication should be taken `frequency`'
          ' times.'
      ),
  )
  frequency_period_unit: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'The unit for `frequency_period` (e.g., "day", "week", "month").'
      ),
  )
  as_needed: Optional[bool] = pydantic.Field(
      default=None,
      description=(
          'Boolean flag indicating if the medication should be taken on an "as'
          ' needed" basis, inferred from dosage instructions.'
      ),
  )
  as_needed_for: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'The reason for taking the medication, to be used only when'
          ' as_needed is true (e.g., "for pain", "for nausea").'
      ),
  )

  # When dose_value is not specified, we assume dose_value prescribed is 1.
  # This todo is not open sourced as this falls in the copybara remove block.
  # TODO: Clinical review if this is safe to assume.
  @pydantic.field_validator('dose_value', mode='before')
  @classmethod
  def replace_none_with_default(cls, v):
    if v is None:
      return 1
    return v


# TODO: Add name_type enum field with values ["GENERIC", "BRAND", "SALT_FORM",
# "OTHER"]. for clinical coding purposes.
# TODO: Update medicine_form instruction to exclude route information.
# TODO: Separate fields for strength_value and strength_unit will be better for
# clinical coding.
class Medication(MedicalData):
  """Represents a prescribed or administered medication."""

  name: Annotated[str, json_schema.SkipJsonSchema()] = pydantic.Field(
      default=None, description='Full name of the medication.'
  )
  medicine_form: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Form as a complete word without abbreviations (e.g., "tablet",'
          ' "capsule", "syrup", "drop" etc.).'
      ),
  )
  brand_name: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Brand name ONLY. Exclude strength (e.g. 500mg), form (e.g. tablet,'
          ' syrup etc.), and active substances.'
      ),
  )
  active_substances: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Active ingredients extracted VERBATIM without strengths or form. DO'
          ' NOT add unwritten salt names or modify the names in any way. Join'
          ' multiple items strictly using'
          ' " + ".'
      ),
  )
  strength: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'Strength (e.g. "500mcg", "500mg + 125mg", "1g", "5ml" etc.). Join'
          ' multiples with " + ".'
      ),
  )
  medication_code: Annotated[Optional[str], json_schema.SkipJsonSchema()] = (
      pydantic.Field(default=None, description='SNOMED code of the medication.')
  )
  medication_code_name: Annotated[
      Optional[str], json_schema.SkipJsonSchema()
  ] = pydantic.Field(
      default=None, description='Name of the medication as per SNOMED.'
  )
  dosage_instructions: Optional[Dosage] = pydantic.Field(
      default=None, description='Dosage instructions for the medication.'
  )
  route: Optional[str] = pydantic.Field(
      default=None,
      description='Route of administration (e.g., "oral", "intravenous").',
  )
  start_date: Optional[datetime.date] = pydantic.Field(
      default=None,
      description=(
          'Start date of the medication course, often labeled as'
          " 'Start Date' or 'From'. If not explicitly specified,"
          ' use the prescription date.'
      ),
  )
  quantity_value: Optional[float] = pydantic.Field(
      default=None,
      description=(
          'The numeric value of the quantity of medication prescribed or'
          ' dispensed.'
      ),
  )
  quantity_unit: Optional[str] = pydantic.Field(
      default=None,
      description=(
          'The unit of measurement for the quantity (e.g., "TAB", "mL").'
      ),
  )

  @pydantic.field_validator('brand_name', mode='before')
  @classmethod
  def sanitize_brand_name(cls, v: Optional[str]) -> Optional[str]:
    """Strips pharmaceutical form terms and leftover unit noise from brand names."""
    if not v:
      return v

    # Common Indian prescription form keywords & abbreviations
    form_keywords = (
        r'\b('
        r'tab|tabs|tablet|tablets|'
        r'cap|caps|capsule|capsules|'
        r'syp|syrup|syr|'
        r'inj|injection|'
        r'oint|ointment|crm|cream|gel|'
        r'drop|drops|drp|'
        r'soln|solution|susp|suspension|'
        r'respule|respules|rotacap|rotacaps|inhaler'
        r')\b'
    )

    # 1. Remove form words (case-insensitive)
    v = re.sub(form_keywords, '', v, flags=re.IGNORECASE)

    # 2. Clean up extra punctuation, trailing hyphens, and whitespace
    v = re.sub(r'[\,\.\/\-]+', ' ', v)
    v = ' '.join(v.split())

    return v

  @pydantic.field_validator('active_substances', mode='before')
  @classmethod
  def sanitize_active_substances(cls, v: Optional[str]) -> Optional[str]:
    if not v:
      return v
    v = re.sub(r'[\|\/]', ' + ', v)
    v = re.sub(
        r'\b\d+(\.\d+)?\s*(mg|g|ml|mcg|i\.?u\.?)\b', '', v, flags=re.IGNORECASE
    )
    return ' + '.join([part.strip() for part in v.split('+') if part.strip()])

  @pydantic.field_validator('strength', mode='before')
  @classmethod
  def sanitize_strength(cls, v: Optional[str]) -> Optional[str]:
    if not v:
      return v
    return normalize_unit(v)

  def model_post_init(self, context: Any) -> None:
    """Constructs full name using base_name, strength, and medicine_form."""
    if self.name:
      return
    parts = []
    if self.brand_name:
      parts.append(self.brand_name)
    if self.active_substances:
      if parts:
        parts.append(f'({self.active_substances})')
      else:
        parts.append(self.active_substances)
    if self.strength:
      parts.append(self.strength)
    if self.medicine_form:
      parts.append(self.medicine_form)
    self.name = ' '.join(parts)

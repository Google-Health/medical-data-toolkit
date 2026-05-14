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
"""Utility functions for FHIR proto conversions and parsing."""

import datetime
from typing import Optional

from google.fhir.r4.proto.core import codes_pb2
from google.fhir.r4.proto.core import datatypes_pb2
from src.document_to_fhir.common.schema import resources


_MICROSECONDS_PER_SECOND = 1000000


def to_fhir_datetime(dt: datetime.datetime) -> datatypes_pb2.DateTime:
  """Converts a datetime object to a FHIR DateTime proto."""
  return datatypes_pb2.DateTime(
      value_us=int(dt.timestamp() * _MICROSECONDS_PER_SECOND),
      timezone=dt.tzname() if dt.tzname() else "UTC",
      precision=datatypes_pb2.DateTime.SECOND,
  )


def to_fhir_date(date_obj: datetime.date) -> datatypes_pb2.Date:
  """Converts a date object to a FHIR Date proto."""
  dt = datetime.datetime.combine(date_obj, datetime.time.min)
  # Set time to start of day in UTC for timestamp calculation
  dt_utc = dt.replace(tzinfo=datetime.timezone.utc)
  return datatypes_pb2.Date(
      value_us=int(dt_utc.timestamp() * _MICROSECONDS_PER_SECOND),
      timezone="UTC",
      precision=datatypes_pb2.Date.DAY,
  )


def parse_name(name_str: str) -> datatypes_pb2.HumanName:
  """Parses a name string into a FHIR HumanName proto."""
  parts = name_str.split(" ")
  prefix = None
  if parts[0] in ["Mr.", "Mrs.", "Ms.", "Dr."]:
    prefix = parts[0]
    parts = parts[1:]

  family = parts[-1]
  given = parts[:-1]

  human_name = datatypes_pb2.HumanName()
  if prefix:
    human_name.prefix.add().value = prefix
  for g in given:
    human_name.given.add().value = g
  human_name.family.value = family
  human_name.text.value = name_str
  return human_name


def to_fhir_administrative_gender(
    gender_str: Optional[str],
) -> codes_pb2.AdministrativeGenderCode.Value:
  """Converts a gender string to a FHIR AdministrativeGenderCode."""
  gender_lower = gender_str.lower() if gender_str else ""
  if gender_lower in ("male", "m"):
    return codes_pb2.AdministrativeGenderCode.MALE
  elif gender_lower in ("female", "f"):
    return codes_pb2.AdministrativeGenderCode.FEMALE
  elif gender_lower in ("other", "o"):
    return codes_pb2.AdministrativeGenderCode.OTHER
  else:
    return codes_pb2.AdministrativeGenderCode.UNKNOWN


def to_fhir_address(address_model: resources.Address) -> datatypes_pb2.Address:
  """Converts an Address model to a FHIR Address proto."""
  address = datatypes_pb2.Address()
  if address_model.city:
    address.city.value = address_model.city
  if address_model.state:
    address.state.value = address_model.state
  if address_model.postal_code:
    address.postal_code.value = address_model.postal_code
  if address_model.country:
    address.country.value = address_model.country
  if address_model.street:
    address.line.add(value=address_model.street)
  return address


def to_fhir_telecom(
    contact_model: resources.Contact,
) -> list[datatypes_pb2.ContactPoint]:
  """Converts a Contact model to a list of FHIR ContactPoint protos."""
  telecoms = []
  if contact_model.phone:
    telecom = datatypes_pb2.ContactPoint()
    telecom.value.value = contact_model.phone
    telecom.system.value = codes_pb2.ContactPointSystemCode.PHONE
    telecoms.append(telecom)
  if contact_model.email:
    telecom = datatypes_pb2.ContactPoint()
    telecom.value.value = contact_model.email
    telecom.system.value = codes_pb2.ContactPointSystemCode.EMAIL
    telecoms.append(telecom)
  if contact_model.fax:
    telecom = datatypes_pb2.ContactPoint()
    telecom.value.value = contact_model.fax
    telecom.system.value = codes_pb2.ContactPointSystemCode.FAX
    telecoms.append(telecom)
  if contact_model.website:
    telecom = datatypes_pb2.ContactPoint()
    telecom.value.value = contact_model.website
    telecom.system.value = codes_pb2.ContactPointSystemCode.URL
    telecoms.append(telecom)
  return telecoms

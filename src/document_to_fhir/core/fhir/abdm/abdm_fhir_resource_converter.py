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
"""Converts internal medical document models to ABDM FHIR resources."""

import datetime
import decimal
import re

from google.fhir.r4.proto.core import codes_pb2
from google.fhir.r4.proto.core import datatypes_pb2
from google.fhir.r4.proto.core.resources import diagnostic_report_pb2
from google.fhir.r4.proto.core.resources import document_reference_pb2
from google.fhir.r4.proto.core.resources import encounter_pb2
from google.fhir.r4.proto.core.resources import observation_pb2
from google.fhir.r4.proto.core.resources import organization_pb2
from google.fhir.r4.proto.core.resources import patient_pb2
from google.fhir.r4.proto.core.resources import practitioner_pb2
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir import fhir_utils

# Regular expression to parse reference range strings like '1.0-2.0' or '1 - 2'.
_RANGE_REGEX = re.compile(r"\s*([0-9.]+)\s*-\s*([0-9.]+)\s*")
# Regular expression to parse reference range threshold strings like '>=1.0' or
# '< 2.0'.
_THRESHOLD_REGEX = re.compile(r"\s*(<|>|<=|>=)\s*([0-9.]+)\s*")


def _normalize_fhir_decimal(val_str: str) -> str | None:
  try:
    d = decimal.Decimal(val_str)
    if d.is_finite():
      return str(d)
  except (decimal.InvalidOperation, ValueError):
    pass
  return None


def _add_identifier(
    resource_fhir, system: str, value: str, code: str, display: str
):
  """Adds an identifier to a FHIR resource."""
  resource_fhir.identifier.add(
      system=datatypes_pb2.Uri(value=system),
      value=datatypes_pb2.String(value=value),
      type=datatypes_pb2.CodeableConcept(
          coding=[
              datatypes_pb2.Coding(
                  system=datatypes_pb2.Uri(
                      value="http://terminology.hl7.org/CodeSystem/v2-0203"
                  ),
                  code=datatypes_pb2.Code(value=code),
                  display=datatypes_pb2.String(value=display),
              )
          ]
      ),
  )


def create_patient(
    patient: resources.Patient,
    patient_id: str,
    identifier_system_uri: str = "http://hospital.smarthealthit.org",
) -> patient_pb2.Patient:
  """Creates a FHIR Patient resource."""
  if not patient:
    raise ValueError("Patient cannot be None")
  patient_fhir = patient_pb2.Patient(
      id=datatypes_pb2.Id(value=patient_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient"
              )
          ]
      ),
  )
  if patient.identifiers and patient.identifiers.mr:
    _add_identifier(
        patient_fhir,
        identifier_system_uri,
        patient.identifiers.mr,
        "MR",
        "Medical Record Number",
    )
  else:
    _add_identifier(
        patient_fhir,
        identifier_system_uri,
        "0",
        "MR",
        "Medical Record Number",
    )
  if patient.name:
    patient_fhir.name.append(fhir_utils.parse_name(patient.name))
  if patient.dob:
    patient_fhir.birth_date.CopyFrom(fhir_utils.to_fhir_date(patient.dob))
  patient_fhir.gender.value = fhir_utils.to_fhir_administrative_gender(
      patient.gender
  )
  return patient_fhir


def create_practitioner(
    practitioner: resources.Practitioner[
        abdm_resources.AbdmPractitionerIdentifiers
    ],
    practitioner_id: str,
    identifier_system_uri: str = "http://hospital.smarthealthit.org",
) -> practitioner_pb2.Practitioner:
  """Creates a FHIR Practitioner resource."""
  if not practitioner:
    raise ValueError("Practitioner cannot be None")
  practitioner_fhir = practitioner_pb2.Practitioner(
      id=datatypes_pb2.Id(value=practitioner_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner"
              )
          ]
      ),
  )
  identifiers = practitioner.identifiers
  if identifiers and identifiers.md:
    _add_identifier(
        practitioner_fhir,
        "https://doctor.ndhm.gov.in",
        identifiers.md,
        "MD",
        "Medical License number",
    )
  else:
    _add_identifier(
        practitioner_fhir,
        "https://doctor.ndhm.gov.in",
        "0",
        "MD",
        "Medical License number",
    )
  if practitioner.name:
    practitioner_fhir.name.append(fhir_utils.parse_name(practitioner.name))
  return practitioner_fhir


def create_organization(
    organization: resources.Organization[
        abdm_resources.AbdmOrganizationIdentifiers
    ],
    organization_id: str,
) -> organization_pb2.Organization:
  """Creates a FHIR Organization resource."""
  if not organization:
    raise ValueError("Organization cannot be None")
  org_fhir = organization_pb2.Organization(
      id=datatypes_pb2.Id(value=organization_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Organization"
              )
          ]
      ),
      active=datatypes_pb2.Boolean(value=True),
  )
  if organization.name:
    org_fhir.name.value = organization.name

  identifiers = organization.identifiers
  if identifiers and identifiers.prn:
    _add_identifier(
        org_fhir,
        "https://facility.ndhm.gov.in",
        identifiers.prn,
        "PRN",
        "Provider number",
    )
  else:
    _add_identifier(
        org_fhir,
        "https://facility.ndhm.gov.in",
        "0",
        "PRN",
        "Provider number",
    )

  if organization.address:
    org_fhir.address.add().CopyFrom(
        fhir_utils.to_fhir_address(organization.address)
    )

  if organization.contact:
    for telecom in fhir_utils.to_fhir_telecom(organization.contact):
      org_fhir.telecom.add().CopyFrom(telecom)
  return org_fhir


def create_encounter(
    encounter_id: str,
    patient_uri_ref: str,
    period_start_dt: datatypes_pb2.DateTime | None,
    practitioner_uri_ref: str | None = None,
    organization_uri_ref: str | None = None,
) -> encounter_pb2.Encounter:
  """Creates a FHIR Encounter resource."""
  encounter = encounter_pb2.Encounter(
      id=datatypes_pb2.Id(value=encounter_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Encounter"
              )
          ]
      ),
      status=encounter_pb2.Encounter.StatusCode(
          value=codes_pb2.EncounterStatusCode.FINISHED
      ),
      class_value=datatypes_pb2.Coding(
          system=datatypes_pb2.Uri(
              value="http://terminology.hl7.org/CodeSystem/v3-ActCode"
          ),
          code=datatypes_pb2.Code(value="AMB"),
          display=datatypes_pb2.String(value="ambulatory"),
      ),
      period=datatypes_pb2.Period(),
  )
  if period_start_dt:
    encounter.period.start.CopyFrom(period_start_dt)
  encounter.subject.uri.value = patient_uri_ref
  if practitioner_uri_ref:
    participant = encounter.participant.add()
    participant.individual.uri.value = practitioner_uri_ref
    participant.type.add(
        coding=[
            datatypes_pb2.Coding(
                system=datatypes_pb2.Uri(
                    value="http://terminology.hl7.org/CodeSystem/v3-ParticipationType"
                ),
                code=datatypes_pb2.Code(value="PPRF"),
                display=datatypes_pb2.String(value="primary performer"),
            )
        ]
    )
  if organization_uri_ref:
    encounter.service_provider.uri.value = organization_uri_ref
  return encounter


def create_lab_observation(
    lab_test: resources.LabTest,
    observation_id: str,
    patient_uri_ref: str,
    specimen_ref: str | None,
    effective_dt: datatypes_pb2.DateTime | None,
    organization_ref: str | None = None,
) -> observation_pb2.Observation:
  """Creates a FHIR Observation resource."""
  if not lab_test:
    raise ValueError("LabTest cannot be None")
  specimen_reference = None
  if specimen_ref:
    specimen_reference = datatypes_pb2.Reference()
    specimen_reference.uri.value = specimen_ref

  obs = observation_pb2.Observation(
      id=datatypes_pb2.Id(value=observation_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"
              )
          ]
      ),
      status=observation_pb2.Observation.StatusCode(
          value=codes_pb2.ObservationStatusCode.FINAL
      ),
      category=[
          datatypes_pb2.CodeableConcept(
              coding=[
                  datatypes_pb2.Coding(
                      system=datatypes_pb2.Uri(
                          value="http://terminology.hl7.org/CodeSystem/observation-category"
                      ),
                      code=datatypes_pb2.Code(value="laboratory"),
                      display=datatypes_pb2.String(value="Laboratory"),
                  )
              ]
          )
      ],
  )
  if effective_dt:
    obs.effective.date_time.CopyFrom(effective_dt)
  if lab_test.loinc_code:
    obs.code.coding.add(
        system=datatypes_pb2.Uri(value="http://loinc.org"),
        code=datatypes_pb2.Code(value=lab_test.loinc_code),
        display=datatypes_pb2.String(
            value=lab_test.loinc_common_name or lab_test.name
        ),
    )
  if lab_test.loinc_common_name:
    obs.code.text.value = lab_test.loinc_common_name
  elif lab_test.name:
    obs.code.text.value = lab_test.name
  obs.subject.uri.value = patient_uri_ref
  if specimen_reference:
    obs.specimen.CopyFrom(specimen_reference)

  if organization_ref:
    obs.performer.add().uri.value = organization_ref

  if lab_test.result is not None and lab_test.result.strip() != "":
    norm_result = _normalize_fhir_decimal(lab_test.result)
    if norm_result is not None:
      obs.value.quantity.value.value = norm_result
      if lab_test.unit:
        obs.value.quantity.unit.value = lab_test.unit
        obs.value.quantity.system.value = "http://unitsofmeasure.org"
        obs.value.quantity.code.value = lab_test.unit
    else:
      obs.value.string_value.value = str(lab_test.result)

  if lab_test.reference_range:
    range_str = lab_test.reference_range[0]
    if range_str:
      rr = obs.reference_range.add()

      range_str_without_unit = (
          range_str.removesuffix(lab_test.unit)
          if lab_test.unit
          else range_str
      )
      # Range regex supports formats like '1.0-2.0' and '1 - 2'.
      match = _RANGE_REGEX.fullmatch(range_str_without_unit)
      if match:
        low_val, high_val = match.groups()
        norm_low = _normalize_fhir_decimal(low_val)
        norm_high = _normalize_fhir_decimal(high_val)
        if norm_low and norm_high:
          rr.low.value.value = norm_low
          rr.low.system.value = "http://unitsofmeasure.org"
          rr.high.value.value = norm_high
          rr.high.system.value = "http://unitsofmeasure.org"
          if lab_test.unit:
            rr.low.unit.value = lab_test.unit
            rr.low.code.value = lab_test.unit
            rr.high.unit.value = lab_test.unit
            rr.high.code.value = lab_test.unit
      # Threshold regex supports formats like '>=1.0' or '< 2.0'.
      match = _THRESHOLD_REGEX.fullmatch(range_str_without_unit)
      if match:
        operator, value = match.groups()
        norm_val = _normalize_fhir_decimal(value)
        if norm_val:
          if operator == ">=" or operator == ">":
            rr.low.value.value = norm_val
            rr.low.system.value = "http://unitsofmeasure.org"
            if lab_test.unit:
              rr.low.unit.value = lab_test.unit
              rr.low.code.value = lab_test.unit
          elif operator == "<" or operator == "<=":
            rr.high.value.value = norm_val
            rr.high.system.value = "http://unitsofmeasure.org"
            if lab_test.unit:
              rr.high.unit.value = lab_test.unit
              rr.high.code.value = lab_test.unit
      # If no pattern matches for range, set the text value.
      if not rr.low.value.value and not rr.high.value.value:
        rr.text.value = range_str

  return obs


def create_panel_observation(
    observation_id: str,
    patient_uri_ref: str,
    effective_dt: datatypes_pb2.DateTime | None,
    panel_name: str,
    panel_loinc_code: str | None,
    member_refs: list[str],
) -> observation_pb2.Observation:
  """Creates a FHIR Observation resource for a panel (Master Observation)."""
  obs = observation_pb2.Observation(
      id=datatypes_pb2.Id(value=observation_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation"
              )
          ]
      ),
      status=observation_pb2.Observation.StatusCode(
          value=codes_pb2.ObservationStatusCode.FINAL
      ),
      category=[
          datatypes_pb2.CodeableConcept(
              coding=[
                  datatypes_pb2.Coding(
                      system=datatypes_pb2.Uri(
                          value="http://terminology.hl7.org/CodeSystem/observation-category"
                      ),
                      code=datatypes_pb2.Code(value="laboratory"),
                      display=datatypes_pb2.String(value="Laboratory"),
                  )
              ]
          )
      ],
  )
  if effective_dt:
    obs.effective.date_time.CopyFrom(effective_dt)

  if panel_loinc_code:
    obs.code.coding.add(
        system=datatypes_pb2.Uri(value="http://loinc.org"),
        code=datatypes_pb2.Code(value=panel_loinc_code),
        display=datatypes_pb2.String(value=panel_name),
    )
  obs.code.text.value = panel_name

  obs.subject.uri.value = patient_uri_ref

  for member_ref in member_refs:
    obs.has_member.add().uri.value = member_ref

  return obs


def create_diagnostic_report(
    report_id: str,
    patient_uri_ref: str,
    practitioner_ref: str | None,
    organization_ref: str | None,
    encounter_uri_ref: str | None,
    observation_refs: list[str],
    specimen_refs: list[str],
    effective_dt: datatypes_pb2.DateTime | None,
    panel_name: str | None = None,
    panel_loinc_code: str | None = None,
    conclusion: str | None = None,
) -> diagnostic_report_pb2.DiagnosticReport:
  """Creates a FHIR DiagnosticReport resource."""
  codeable_concept = datatypes_pb2.CodeableConcept()
  if panel_loinc_code:
    coding = codeable_concept.coding.add(
        system=datatypes_pb2.Uri(value="http://loinc.org"),
        code=datatypes_pb2.Code(value=panel_loinc_code),
    )
    if panel_name:
      coding.display.value = panel_name
      codeable_concept.text.value = panel_name
    else:
      codeable_concept.text.value = panel_loinc_code
  elif panel_name:
    codeable_concept.text.value = panel_name
  else:
    codeable_concept.text.value = "Laboratory Report"

  encounter_reference = None
  if encounter_uri_ref:
    encounter_reference = datatypes_pb2.Reference()
    encounter_reference.uri.value = encounter_uri_ref

  issued_precision = datatypes_pb2.Instant.SECOND
  if effective_dt:
    if effective_dt.precision == datatypes_pb2.DateTime.MICROSECOND:
      issued_precision = datatypes_pb2.Instant.MICROSECOND
    elif effective_dt.precision == datatypes_pb2.DateTime.MILLISECOND:
      issued_precision = datatypes_pb2.Instant.MILLISECOND

  report = diagnostic_report_pb2.DiagnosticReport(
      id=datatypes_pb2.Id(value=report_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportLab"
              )
          ]
      ),
      status=diagnostic_report_pb2.DiagnosticReport.StatusCode(
          value=codes_pb2.DiagnosticReportStatusCode.FINAL
      ),
      category=[
          datatypes_pb2.CodeableConcept(
              coding=[
                  datatypes_pb2.Coding(
                      system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
                      code=datatypes_pb2.Code(value="4241000179101"),
                      display=datatypes_pb2.String(value="Laboratory report"),
                  )
              ]
          )
      ],
      code=codeable_concept,
  )
  if effective_dt:
    report.effective.date_time.CopyFrom(effective_dt)
    report.issued.value_us = effective_dt.value_us
    report.issued.timezone = effective_dt.timezone
    report.issued.precision = issued_precision
  report.subject.uri.value = patient_uri_ref
  if encounter_reference:
    report.encounter.CopyFrom(encounter_reference)
  for obs_ref in observation_refs:
    report.result.add().uri.value = obs_ref
  for spec_ref in specimen_refs:
    report.specimen.add().uri.value = spec_ref
  if practitioner_ref:
    report.results_interpreter.add().uri.value = practitioner_ref
    report.performer.add().uri.value = practitioner_ref
  if organization_ref:
    report.performer.add().uri.value = organization_ref

  report.conclusion.value = conclusion or "NA"
  return report


def create_document_reference(
    doc_ref_id: str,
    patient_ref: str,
    mime_type: str,
    encoded_data: str,
    creation_time: datetime.datetime,
    type_code: str = "4241000179101",
    type_display: str = "Laboratory report",
    type_system: str = "http://snomed.info/sct",
) -> document_reference_pb2.DocumentReference:
  """Creates a FHIR DocumentReference resource."""
  doc_ref = document_reference_pb2.DocumentReference(
      id=datatypes_pb2.Id(value=doc_ref_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentReference"
              )
          ]
      ),
      status=document_reference_pb2.DocumentReference.StatusCode(
          value=codes_pb2.DocumentReferenceStatusCode.CURRENT
      ),
      doc_status=document_reference_pb2.DocumentReference.DocStatusCode(
          value=codes_pb2.CompositionStatusCode.FINAL
      ),
      type=datatypes_pb2.CodeableConcept(
          coding=[
              datatypes_pb2.Coding(
                  system=datatypes_pb2.Uri(value=type_system),
                  code=datatypes_pb2.Code(value=type_code),
                  display=datatypes_pb2.String(value=type_display),
              )
          ],
          text=datatypes_pb2.String(value=type_display),
      ),
      subject=datatypes_pb2.Reference(
          uri=datatypes_pb2.String(value=patient_ref)
      ),
  )

  content = doc_ref.content.add()
  content.attachment.content_type.value = mime_type
  content.attachment.language.value = "en-IN"
  content.attachment.data.value = encoded_data.encode("utf-8")
  content.attachment.title.value = type_display
  content.attachment.creation.CopyFrom(
      fhir_utils.to_fhir_datetime(creation_time)
  )

  return doc_ref

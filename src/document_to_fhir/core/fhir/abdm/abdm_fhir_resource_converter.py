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
import logging

from google.fhir.r4.proto.core import codes_pb2
from google.fhir.r4.proto.core import datatypes_pb2
from google.fhir.r4.proto.core import valuesets_pb2
from google.fhir.r4.proto.core.resources import diagnostic_report_pb2
from google.fhir.r4.proto.core.resources import document_reference_pb2
from google.fhir.r4.proto.core.resources import encounter_pb2
from google.fhir.r4.proto.core.resources import immunization_pb2
from google.fhir.r4.proto.core.resources import medication_request_pb2
from google.fhir.r4.proto.core.resources import observation_pb2
from google.fhir.r4.proto.core.resources import organization_pb2
from google.fhir.r4.proto.core.resources import patient_pb2
from google.fhir.r4.proto.core.resources import practitioner_pb2
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir import fhir_utils


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
    for ref_range in lab_test.reference_range:
      rr = obs.reference_range.add()
      if ref_range.low:
        norm_low = _normalize_fhir_decimal(ref_range.low)
        if norm_low:
          rr.low.value.value = norm_low
          rr.low.system.value = "http://unitsofmeasure.org"
          if lab_test.unit:
            rr.low.unit.value = lab_test.unit
            rr.low.code.value = lab_test.unit
      if ref_range.high:
        norm_high = _normalize_fhir_decimal(ref_range.high)
        if norm_high:
          rr.high.value.value = norm_high
          rr.high.system.value = "http://unitsofmeasure.org"
          if lab_test.unit:
            rr.high.unit.value = lab_test.unit
            rr.high.code.value = lab_test.unit
      if ref_range.text:
        rr.text.value = ref_range.text

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


def create_imaging_diagnostic_report(
    report_id: str,
    patient_uri_ref: str,
    practitioner_ref: str | None,
    organization_ref: str | None,
    encounter_uri_ref: str | None,
    effective_dt: datatypes_pb2.DateTime | None,
    exam_name: str,
    modality: str | None = None,
    findings: str | None = None,
    conclusion: str | None = None,
    conclusion_code: str | None = None,
    modality_subtype: str | None = None,
    procedure_loinc_code: str | None = None,
    procedure_loinc_common_name: str | None = None,
) -> diagnostic_report_pb2.DiagnosticReport:
  """Creates a FHIR DiagnosticReport resource for Imaging."""
  codeable_concept = datatypes_pb2.CodeableConcept()
  if procedure_loinc_code:
    coding = codeable_concept.coding.add(
        system=datatypes_pb2.Uri(value="http://loinc.org"),
        code=datatypes_pb2.Code(value=procedure_loinc_code),
    )
    display_name = procedure_loinc_common_name or exam_name
    if display_name:
      coding.display.value = display_name
  text_value = exam_name or procedure_loinc_common_name
  if text_value:
    codeable_concept.text.value = text_value

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

  category_concept = datatypes_pb2.CodeableConcept(
      coding=[
          datatypes_pb2.Coding(
              system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
              code=datatypes_pb2.Code(value="721981007"),
              display=datatypes_pb2.String(value="Diagnostic studies report"),
          )
      ]
  )
  # Modality and modality_subtype are combined to store in category.text.
  modality_parts = [m for m in (modality, modality_subtype) if m]
  if modality_parts:
    category_concept.text.value = " ".join(modality_parts)

  categories = [category_concept]

  report = diagnostic_report_pb2.DiagnosticReport(
      id=datatypes_pb2.Id(value=report_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportImaging"
              )
          ]
      ),
      status=diagnostic_report_pb2.DiagnosticReport.StatusCode(
          value=codes_pb2.DiagnosticReportStatusCode.FINAL
      ),
      category=categories,
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
  if practitioner_ref:
    report.results_interpreter.add().uri.value = practitioner_ref
    report.performer.add().uri.value = practitioner_ref
  if organization_ref:
    report.performer.add().uri.value = organization_ref

  conclusions = []
  if findings:
    conclusions.append(f"Findings: {findings}")
  if conclusion:
    conclusions.append(f"Conclusion: {conclusion}")

  if conclusions:
    report.conclusion.value = "\n".join(conclusions)
  else:
    report.conclusion.value = "NA"

  if conclusion_code:
    display_val = conclusion if conclusion else "Coded Conclusion"
    report.conclusion_code.add(
        coding=[
            datatypes_pb2.Coding(
                system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
                code=datatypes_pb2.Code(value=conclusion_code),
                display=datatypes_pb2.String(value=display_val),
            )
        ]
    )

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


def create_medication_request(
    medication: resources.Medication | None,
    medication_request_id: str,
    patient_uri_ref: str,
    practitioner_uri_ref: str | None = None,
    authored_on_date: datetime.datetime | None = None,
    encounter_uri_ref: str | None = None,
) -> medication_request_pb2.MedicationRequest:
  """Creates a FHIR MedicationRequest resource from a Medication object."""
  if not medication:
    raise ValueError("Medication cannot be None")
  med_req = medication_request_pb2.MedicationRequest(
      id=datatypes_pb2.Id(value=medication_request_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/MedicationRequest"
              )
          ]
      ),
      status=medication_request_pb2.MedicationRequest.StatusCode(
          value=codes_pb2.MedicationrequestStatusCode.ACTIVE
      ),
      intent=medication_request_pb2.MedicationRequest.IntentCode(
          value=codes_pb2.MedicationRequestIntentCode.ORDER
      ),
  )
  med_req.subject.uri.value = patient_uri_ref
  if practitioner_uri_ref:
    med_req.requester.uri.value = practitioner_uri_ref
  if authored_on_date:
    med_req.authored_on.CopyFrom(fhir_utils.to_fhir_datetime(authored_on_date))
  elif medication.start_date:
    start_dt = datetime.datetime.combine(
        medication.start_date, datetime.time.min
    ).replace(tzinfo=datetime.timezone.utc)
    med_req.authored_on.CopyFrom(fhir_utils.to_fhir_datetime(start_dt))

  if encounter_uri_ref:
    med_req.encounter.uri.value = encounter_uri_ref

  if medication.medication_code and medication.medication_code_name:
    med_req.medication.codeable_concept.coding.add(
        system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
        code=datatypes_pb2.Code(value=medication.medication_code),
        display=datatypes_pb2.String(value=medication.medication_code_name),
    )
    med_req.medication.codeable_concept.text.value = (
        medication.medication_code_name
    )
  elif medication.name:
    med_req.medication.codeable_concept.text.value = medication.name

  if medication.dosage_instructions:
    instruction = medication.dosage_instructions
    dosage = med_req.dosage_instruction.add()
    dosage_text_parts = []
    if instruction.dose_value is not None and instruction.dose_unit:
      dosage_text_parts.append(
          f"{instruction.dose_value} {instruction.dose_unit}"
      )
    if medication.route:
      dosage_text_parts.append(medication.route)
      dosage.route.text.value = medication.route
    if instruction.frequency is not None:
      if (
          instruction.frequency_period is not None
          and instruction.frequency_period_unit
      ):
        dosage_text_parts.append(
            f"{instruction.frequency} times per"
            f" {instruction.frequency_period}"
            f" {instruction.frequency_period_unit}"
        )
      elif instruction.frequency_period_unit:
        dosage_text_parts.append(
            f"{instruction.frequency} times per"
            f" {instruction.frequency_period_unit}"
        )
    if dosage_text_parts:
      dosage.text.value = " ".join(dosage_text_parts)
    if instruction.instruction:
      dosage.patient_instruction.value = instruction.instruction
  elif medication.route:
    dosage = med_req.dosage_instruction.add()
    dosage.route.text.value = medication.route
    dosage.text.value = medication.route

  return med_req


def create_immunization(
    immunization: resources.Immunization | None,
    immunization_id: str,
    patient_uri_ref: str,
    practitioner_uri_ref: str | None = None,
    occurrence_date: datetime.date | datetime.datetime | None = None,
    encounter_uri_ref: str | None = None,
    primary_source: bool = False,
) -> immunization_pb2.Immunization:
  """Creates a FHIR Immunization resource from an Immunization object."""
  if not immunization:
    raise ValueError("Immunization cannot be None")

  imm = immunization_pb2.Immunization(
      id=datatypes_pb2.Id(value=immunization_id),
      meta=datatypes_pb2.Meta(
          profile=[
              datatypes_pb2.Canonical(
                  value=(
                      "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Immunization"
                  )
              )
          ]
      ),
      status=immunization_pb2.Immunization.StatusCode(
          value=valuesets_pb2.ImmunizationStatusCodesValueSet.COMPLETED
      ),
      primary_source=datatypes_pb2.Boolean(value=primary_source),
  )
  imm.patient.uri.value = patient_uri_ref

  if practitioner_uri_ref:
    performer = imm.performer.add()
    performer.actor.uri.value = practitioner_uri_ref
    performer.function.coding.add(
        system=datatypes_pb2.Uri(
            value="http://terminology.hl7.org/CodeSystem/v2-0443"
        ),
        code=datatypes_pb2.Code(value="AP"),
        display=datatypes_pb2.String(value="Administering Provider"),
    )

  if occurrence_date:
    if isinstance(occurrence_date, datetime.datetime):
      occ_dt = occurrence_date
      if occ_dt.tzinfo is None:
        occ_dt = occ_dt.replace(tzinfo=datetime.timezone.utc)
      imm.occurrence.date_time.CopyFrom(fhir_utils.to_fhir_datetime(occ_dt))
    else:
      dt = datetime.datetime.combine(
          occurrence_date, datetime.time.min
      ).replace(tzinfo=datetime.timezone.utc)
      imm.occurrence.date_time.CopyFrom(fhir_utils.to_fhir_datetime(dt))
  elif immunization.date:
    dt = datetime.datetime.combine(
        immunization.date, datetime.time.min
    ).replace(tzinfo=datetime.timezone.utc)
    imm.occurrence.date_time.CopyFrom(fhir_utils.to_fhir_datetime(dt))

  if encounter_uri_ref:
    imm.encounter.uri.value = encounter_uri_ref

  if immunization.vaccine_code and immunization.vaccine_code_name:
    imm.vaccine_code.coding.add(
        system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
        code=datatypes_pb2.Code(value=immunization.vaccine_code),
        display=datatypes_pb2.String(value=immunization.vaccine_code_name),
    )
    imm.vaccine_code.text.value = immunization.vaccine_code_name
  elif immunization.vaccine_name:
    imm.vaccine_code.text.value = immunization.vaccine_name

  return imm

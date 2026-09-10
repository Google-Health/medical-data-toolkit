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
"""Unit tests for ABDM FHIR resource converters.

This module tests helper functions in `abdm_fhir_resource_converter` that
convert internal medical document data models into Ayushman Bharat Digital
Mission (ABDM) compliant FHIR R4 resources.

The test cases verify proper schema translation, ABDM profile URIs, identifier
systems, coding systems (e.g. SNOMED, LOINC, RxNorm), reference linking, and
data normalization for generated FHIR resources.
"""

import datetime

from absl.testing import absltest
from absl.testing import parameterized

from google.fhir.r4.proto.core import datatypes_pb2
from google.fhir.r4.proto.core import valuesets_pb2
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_resources
from src.document_to_fhir.core.fhir.abdm import abdm_fhir_resource_converter


class AbdmFhirResourceConverterTest(parameterized.TestCase):

  def test_create_patient(self):
    patient_data = resources.Patient(
        name="John Doe",
        identifiers=resources.PatientIdentifiers(mr="MR123"),
        dob=datetime.date(1990, 1, 1),
        gender="male",
    )
    patient_fhir = abdm_fhir_resource_converter.create_patient(
        patient_data, "patient-id"
    )
    self.assertEqual(patient_fhir.id.value, "patient-id")
    self.assertEqual(
        patient_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Patient",
    )
    self.assertEqual(patient_fhir.identifier[0].value.value, "MR123")

  def test_create_patient_missing_identifier(self):
    patient_data = resources.Patient(
        name="John Doe",
        dob=datetime.date(1990, 1, 1),
        gender="male",
    )
    patient_fhir = abdm_fhir_resource_converter.create_patient(
        patient_data, "patient-id"
    )
    self.assertEqual(patient_fhir.id.value, "patient-id")
    self.assertEqual(patient_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        patient_fhir.identifier[0].system.value, "http://hospital.smarthealthit.org"
    )

  def test_create_practitioner(self):
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
        identifiers=abdm_resources.AbdmPractitionerIdentifiers(md="MD123"),
    )
    practitioner_fhir = abdm_fhir_resource_converter.create_practitioner(
        practitioner_data, "practitioner-id"
    )
    self.assertEqual(practitioner_fhir.id.value, "practitioner-id")
    self.assertEqual(
        practitioner_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Practitioner",
    )

  def test_create_practitioner_missing_identifier(self):
    practitioner_data = resources.Practitioner(
        name="Dr. Smith",
    )
    practitioner_fhir = abdm_fhir_resource_converter.create_practitioner(
        practitioner_data, "practitioner-id"
    )
    self.assertEqual(practitioner_fhir.id.value, "practitioner-id")
    self.assertEqual(practitioner_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        practitioner_fhir.identifier[0].system.value, "https://doctor.ndhm.gov.in"
    )

  def test_create_organization(self):
    org_data = resources.Organization(
        name="General Hospital",
        identifiers=abdm_resources.AbdmOrganizationIdentifiers(prn="PRNORG"),
    )
    org_fhir = abdm_fhir_resource_converter.create_organization(
        org_data, "org-id"
    )
    self.assertEqual(org_fhir.id.value, "org-id")
    self.assertEqual(
        org_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Organization",
    )
    self.assertEqual(org_fhir.identifier[0].value.value, "PRNORG")

  def test_create_organization_missing_identifier(self):
    org_data = resources.Organization(
        name="General Hospital",
    )
    org_fhir = abdm_fhir_resource_converter.create_organization(
        org_data, "org-id"
    )
    self.assertEqual(org_fhir.id.value, "org-id")
    self.assertEqual(org_fhir.identifier[0].value.value, "0")
    self.assertEqual(
        org_fhir.identifier[0].system.value, "https://facility.ndhm.gov.in"
    )

  def test_create_encounter(self):
    start_dt = datatypes_pb2.DateTime(value_us=12345)
    encounter_fhir = abdm_fhir_resource_converter.create_encounter(
        "encounter-id", "Patient/patient-id", start_dt
    )
    self.assertEqual(encounter_fhir.id.value, "encounter-id")
    self.assertEqual(encounter_fhir.subject.uri.value, "Patient/patient-id")

  def test_create_lab_observation(self):
    lab_test = resources.LabTest(
        core_analyte="Glucose",
        name="Glucose",
        result="100",
        unit="mg/dL",
        loinc_code="2345-7",
    )
    effective_dt = datatypes_pb2.DateTime(value_us=12345)
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, effective_dt
    )
    self.assertEqual(obs_fhir.id.value, "obs-id")
    self.assertEqual(
        obs_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Observation",
    )

  def test_create_diagnostic_report(self):
    effective_dt = datatypes_pb2.DateTime(value_us=12345)
    report_fhir = abdm_fhir_resource_converter.create_diagnostic_report(
        "report-id",
        "Patient/patient-id",
        "Practitioner/prac-id",
        "Organization/org-id",
        "Encounter/enc-id",
        ["Observation/obs-id"],
        ["Specimen/spec-id"],
        effective_dt,
    )
    self.assertEqual(report_fhir.id.value, "report-id")
    self.assertEqual(
        report_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportLab",
    )

  def test_create_imaging_diagnostic_report(self):
    effective_dt = datatypes_pb2.DateTime(value_us=12345)
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref="Practitioner/prac-id",
        organization_ref="Organization/org-id",
        encounter_uri_ref="Encounter/enc-id",
        effective_dt=effective_dt,
        exam_name="CT Brain",
        modality="CT",
        findings="Some findings",
        conclusion="Some conclusion",
        conclusion_code="123456",
    )
    self.assertEqual(report_fhir.id.value, "report-id")
    self.assertEqual(
        report_fhir.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportImaging",
    )
    self.assertEqual(report_fhir.code.text.value, "CT Brain")
    self.assertEqual(
        report_fhir.conclusion.value,
        "Findings: Some findings\nConclusion: Some conclusion",
    )
    self.assertEqual(report_fhir.category[0].text.value, "CT")
    self.assertEqual(
        report_fhir.conclusion_code[0].coding[0].code.value, "123456"
    )
    self.assertEqual(
        report_fhir.conclusion_code[0].coding[0].display.value,
        "Some conclusion",
    )

  def test_create_imaging_diagnostic_report_with_loinc(self):
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref=None,
        organization_ref=None,
        encounter_uri_ref=None,
        effective_dt=None,
        exam_name="CT Brain",
        modality="CT",
        modality_subtype="Angio",
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name="CT Head and Neck WO contrast",
    )
    self.assertEqual(report_fhir.code.text.value, "CT Brain")
    self.assertLen(report_fhir.code.coding, 1)
    self.assertEqual(
        report_fhir.code.coding[0].system.value, "http://loinc.org"
    )
    self.assertEqual(report_fhir.code.coding[0].code.value, "82692-5")
    self.assertEqual(
        report_fhir.code.coding[0].display.value,
        "CT Head and Neck WO contrast",
    )
    self.assertEqual(report_fhir.category[0].text.value, "CT Angio")

  def test_create_imaging_diagnostic_report_with_loinc_no_exam_name(self):
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref=None,
        organization_ref=None,
        encounter_uri_ref=None,
        effective_dt=None,
        exam_name="",
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name="CT Head and Neck WO contrast",
    )
    self.assertEqual(
        report_fhir.code.text.value, "CT Head and Neck WO contrast"
    )
    self.assertLen(report_fhir.code.coding, 1)
    self.assertEqual(
        report_fhir.code.coding[0].system.value, "http://loinc.org"
    )
    self.assertEqual(report_fhir.code.coding[0].code.value, "82692-5")
    self.assertEqual(
        report_fhir.code.coding[0].display.value,
        "CT Head and Neck WO contrast",
    )

  def test_create_imaging_diagnostic_report_with_loinc_no_common_name(self):
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref=None,
        organization_ref=None,
        encounter_uri_ref=None,
        effective_dt=None,
        exam_name="CT Brain",
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name=None,
    )
    self.assertEqual(report_fhir.code.text.value, "CT Brain")
    self.assertLen(report_fhir.code.coding, 1)
    self.assertEqual(
        report_fhir.code.coding[0].system.value, "http://loinc.org"
    )
    self.assertEqual(report_fhir.code.coding[0].code.value, "82692-5")
    self.assertEqual(
        report_fhir.code.coding[0].display.value,
        "CT Brain",
    )

  def test_create_imaging_diagnostic_report_with_loinc_only_code(self):
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref=None,
        organization_ref=None,
        encounter_uri_ref=None,
        effective_dt=None,
        exam_name="",
        procedure_loinc_code="82692-5",
        procedure_loinc_common_name=None,
    )
    self.assertFalse(report_fhir.code.HasField("text"))
    self.assertLen(report_fhir.code.coding, 1)
    self.assertEqual(
        report_fhir.code.coding[0].system.value, "http://loinc.org"
    )
    self.assertEqual(report_fhir.code.coding[0].code.value, "82692-5")
    self.assertFalse(report_fhir.code.coding[0].HasField("display"))

  def test_create_imaging_diagnostic_report_no_modality(self):
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref="Practitioner/prac-id",
        organization_ref="Organization/org-id",
        encounter_uri_ref="Encounter/enc-id",
        effective_dt=None,
        exam_name="CT Brain",
        modality=None,
    )
    self.assertLen(report_fhir.category, 1)
    self.assertEqual(report_fhir.category[0].coding[0].code.value, "721981007")
    self.assertFalse(report_fhir.category[0].HasField("text"))

  @parameterized.named_parameters(
      (
          "microsecond",
          datatypes_pb2.DateTime.Precision.MICROSECOND,
          datatypes_pb2.Instant.Precision.MICROSECOND,
      ),
      (
          "millisecond",
          datatypes_pb2.DateTime.Precision.MILLISECOND,
          datatypes_pb2.Instant.Precision.MILLISECOND,
      ),
      (
          "second",
          datatypes_pb2.DateTime.Precision.SECOND,
          datatypes_pb2.Instant.Precision.SECOND,
      ),
  )
  def test_create_imaging_diagnostic_report_precision(
      self, dt_precision, expected_instant_precision
  ):
    effective_dt = datatypes_pb2.DateTime(
        value_us=12345, precision=dt_precision
    )
    report_fhir = abdm_fhir_resource_converter.create_imaging_diagnostic_report(
        report_id="report-id",
        patient_uri_ref="Patient/patient-id",
        practitioner_ref=None,
        organization_ref=None,
        encounter_uri_ref=None,
        effective_dt=effective_dt,
        exam_name="CT Brain",
    )
    self.assertEqual(report_fhir.issued.precision, expected_instant_precision)

  @parameterized.named_parameters(
      (
          "zero_prefix_normalization",
          "Eosinophils",
          "Absolute Eosinophil Count",
          "00",
          "cells/mcL",
          [resources.ReferenceRange(low="00", high="450", text="00-450")],
          "0",
          "0",
          "450",
      ),
      (
          "threshold_and_decimal_normalization",
          "Glucose",
          "Glucose",
          "01.0",
          "mg/dL",
          [resources.ReferenceRange(high="01", text="< 01")],
          "1.0",
          None,
          "1",
      ),
  )
  def test_create_lab_observation_normalization(
      self,
      core_analyte,
      name,
      result,
      unit,
      reference_range,
      expected_value,
      expected_low,
      expected_high,
  ):
    lab_test = resources.LabTest(
        core_analyte=core_analyte,
        name=name,
        result=result,
        unit=unit,
        reference_range=reference_range,
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertEqual(obs_fhir.value.quantity.value.value, expected_value)
    self.assertLen(obs_fhir.reference_range, 1)
    if expected_low is not None:
      self.assertEqual(
          obs_fhir.reference_range[0].low.value.value, expected_low
      )
    else:
      self.assertFalse(obs_fhir.reference_range[0].low.HasField("value"))

    if expected_high is not None:
      self.assertEqual(
          obs_fhir.reference_range[0].high.value.value, expected_high
      )
    else:
      self.assertFalse(obs_fhir.reference_range[0].high.HasField("value"))

  @parameterized.named_parameters(
      ("unparseable_text", "M: 1 - 7 / F: 3 - 12", "M: 1 - 7 / F: 3 - 12"),
      ("invalid_decimals", "1.2.3-4.5.6", "1.2.3-4.5.6"),
  )
  def test_create_lab_observation_invalid_range_fallback(
      self, reference_range_str, expected_text
  ):
    lab_test = resources.LabTest(
        core_analyte="ESR",
        name="ESR",
        result="23",
        unit="mm",
        reference_range=[resources.ReferenceRange(text=reference_range_str)],
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertLen(obs_fhir.reference_range, 1)
    self.assertFalse(obs_fhir.reference_range[0].low.HasField("value"))
    self.assertFalse(obs_fhir.reference_range[0].high.HasField("value"))
    self.assertEqual(obs_fhir.reference_range[0].text.value, expected_text)

  @parameterized.named_parameters(
      ("empty_string", ""),
      ("whitespace_only", "   "),
  )
  def test_create_lab_observation_empty_result(self, result):
    lab_test = resources.LabTest(
        core_analyte="WBC",
        name="White Blood Cell Count",
        result=result,
        unit="Thousand/uL",
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertFalse(obs_fhir.HasField("value"))

  def test_create_lab_observation_reference_range_fields(self):
    lab_test = resources.LabTest(
        core_analyte="HbA1c",
        name="HbA1c",
        result="5.5",
        unit="%",
        reference_range=[
            resources.ReferenceRange(
                low="4.5",
                high="6.0",
                text="Male 18+ 4.5-6.0",
            )
        ],
    )
    obs_fhir = abdm_fhir_resource_converter.create_lab_observation(
        lab_test, "obs-id", "Patient/patient-id", None, None
    )
    self.assertLen(obs_fhir.reference_range, 1)
    rr = obs_fhir.reference_range[0]
    self.assertEqual(rr.low.value.value, "4.5")
    self.assertEqual(rr.high.value.value, "6.0")
    self.assertEqual(rr.text.value, "Male 18+ 4.5-6.0")

  def test_create_medication_request(self):
    medication = resources.Medication(
        name="Azithromycin 250 mg Tablet",
        brand_name="Azithromycin",
        strength="250 mg",
        medicine_form="Tablet",
        medication_code="1145423002",
        medication_code_name="Azithromycin 250 mg oral tablet",
        dosage_instructions=resources.Dosage(
            instruction="One tablet at once with or after food",
            dose_value=1.0,
            dose_unit="TAB",
            frequency=1,
            frequency_period=1.0,
            frequency_period_unit="d",
        ),
        route="Oral",
    )
    authored_date = datetime.datetime(2026, 1, 1, 10, 0, 0)
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication,
        "med-req-id",
        "urn:uuid:patient-id",
        "urn:uuid:practitioner-id",
        authored_date,
        "urn:uuid:encounter-id",
    )
    self.assertEqual(med_req.id.value, "med-req-id")
    self.assertEqual(
        med_req.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/MedicationRequest",
    )
    self.assertEqual(med_req.subject.uri.value, "urn:uuid:patient-id")
    self.assertEqual(med_req.requester.uri.value, "urn:uuid:practitioner-id")
    self.assertEqual(med_req.encounter.uri.value, "urn:uuid:encounter-id")
    self.assertEqual(
        med_req.medication.codeable_concept.coding[0].system.value,
        "http://snomed.info/sct",
    )
    self.assertEqual(
        med_req.medication.codeable_concept.coding[0].code.value, "1145423002"
    )
    self.assertEqual(
        med_req.medication.codeable_concept.text.value,
        "Azithromycin 250 mg oral tablet",
    )
    self.assertLen(med_req.dosage_instruction, 1)
    self.assertEqual(
        med_req.dosage_instruction[0].patient_instruction.value,
        "One tablet at once with or after food",
    )
    self.assertEqual(
        med_req.dosage_instruction[0].text.value,
        "1.0 TAB Oral 1 times per 1.0 d",
    )
    self.assertEqual(
        med_req.dosage_instruction[0].route.text.value,
        "Oral",
    )

  def test_create_medication_request_missing_medication(self):
    with self.assertRaises(ValueError):
      abdm_fhir_resource_converter.create_medication_request(
          None, "med-req-id", "urn:uuid:patient-id"
      )

  def test_create_medication_request_start_date_fallback(self):
    medication = resources.Medication(
        name="Amoxicillin 500 MG",
        start_date=datetime.date(2026, 2, 1),
    )
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication,
        "med-req-id",
        "urn:uuid:patient-id",
        authored_on_date=None,
    )
    self.assertTrue(med_req.HasField("authored_on"))
    expected_us = int(
        datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc).timestamp()
        * 1000000
    )
    self.assertEqual(med_req.authored_on.value_us, expected_us)
    self.assertEqual(
        med_req.authored_on.precision,
        datatypes_pb2.DateTime.Precision.SECOND,
    )

  def test_create_medication_request_dosage_frequency_with_period(self):
    medication = resources.Medication(
        name="Paracetamol 500 MG",
        dosage_instructions=resources.Dosage(
            instruction="Take every 8 hours",
            dose_value=500.0,
            dose_unit="mg",
            frequency=1,
            frequency_period=8.0,
            frequency_period_unit="h",
        ),
    )
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertLen(med_req.dosage_instruction, 1)
    self.assertEqual(
        med_req.dosage_instruction[0].text.value,
        "500.0 mg 1 times per 8.0 h",
    )
    self.assertEqual(
        med_req.dosage_instruction[0].patient_instruction.value,
        "Take every 8 hours",
    )

  def test_create_medication_request_dosage_frequency_period_unit_only(self):
    medication = resources.Medication(
        name="Paracetamol 500 MG",
        dosage_instructions=resources.Dosage(
            instruction="Take 2 times daily",
            frequency=2,
            frequency_period_unit="d",
        ),
    )
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertLen(med_req.dosage_instruction, 1)
    self.assertEqual(med_req.dosage_instruction[0].text.value, "2 times per d")

  def test_create_medication_request_dosage_dose_only(self):
    medication = resources.Medication(
        name="Syrup",
        dosage_instructions=resources.Dosage(
            dose_value=5.0,
            dose_unit="mL",
        ),
    )
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertLen(med_req.dosage_instruction, 1)
    self.assertEqual(med_req.dosage_instruction[0].text.value, "5.0 mL")

  def test_create_medication_request_route_only_no_dosage_instructions(self):
    medication = resources.Medication(
        name="Ibuprofen 400 MG",
        route="Oral",
    )
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertLen(med_req.dosage_instruction, 1)
    self.assertEqual(med_req.dosage_instruction[0].route.text.value, "Oral")
    self.assertEqual(med_req.dosage_instruction[0].text.value, "Oral")

  def test_create_medication_request_name_only_no_code(self):
    medication = resources.Medication(name="Custom Syrup")
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertEqual(
        med_req.medication.codeable_concept.text.value, "Custom Syrup"
    )
    self.assertFalse(med_req.medication.codeable_concept.coding)

  def test_create_medication_request_code_only_no_name(self):
    medication = resources.Medication(medication_code="1145423002")
    med_req = abdm_fhir_resource_converter.create_medication_request(
        medication, "med-req-id", "urn:uuid:patient-id"
    )
    self.assertEmpty(med_req.medication.codeable_concept.coding)
    self.assertFalse(med_req.medication.codeable_concept.HasField("text"))

  def test_create_immunization_full_fields(self):
    immunization = resources.Immunization(
        vaccine_name="Covaxin",
        date=datetime.date(2026, 1, 15),
        vaccine_code="1119349007",
    )
    imm = abdm_fhir_resource_converter.create_immunization(
        immunization=immunization,
        immunization_id="imm-id",
        patient_uri_ref="urn:uuid:patient-id",
        practitioner_uri_ref="urn:uuid:practitioner-id",
        encounter_uri_ref="urn:uuid:encounter-id",
        primary_source=True,
    )
    self.assertEqual(imm.id.value, "imm-id")
    self.assertEqual(
        imm.meta.profile[0].value,
        "https://nrces.in/ndhm/fhir/r4/StructureDefinition/Immunization",
    )
    self.assertEqual(
        imm.status.value,
        valuesets_pb2.ImmunizationStatusCodesValueSet.COMPLETED,
    )
    self.assertTrue(imm.primary_source.value)
    self.assertEqual(imm.patient.uri.value, "urn:uuid:patient-id")
    self.assertEqual(imm.encounter.uri.value, "urn:uuid:encounter-id")
    self.assertLen(imm.performer, 1)
    self.assertEqual(
        imm.performer[0].actor.uri.value, "urn:uuid:practitioner-id"
    )
    self.assertEqual(imm.performer[0].function.coding[0].code.value, "AP")
    self.assertEqual(
        imm.performer[0].function.coding[0].display.value,
        "Administering Provider",
    )
    self.assertLen(imm.vaccine_code.coding, 1)
    self.assertEqual(
        imm.vaccine_code.coding[0].system.value,
        "http://snomed.info/sct",
    )
    self.assertEqual(
        imm.vaccine_code.coding[0].code.value,
        "1119349007",
    )
    self.assertEqual(
        imm.vaccine_code.coding[0].display.value,
        "Covaxin",
    )
    self.assertEqual(
        imm.vaccine_code.text.value,
        "Covaxin",
    )
    self.assertTrue(imm.occurrence.HasField("date_time"))
    expected_us = int(
        datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc).timestamp()
        * 1000000
    )
    self.assertEqual(imm.occurrence.date_time.value_us, expected_us)

  def test_create_immunization_name_only_no_code(self):
    immunization = resources.Immunization(vaccine_name="BCG Vaccine")
    imm = abdm_fhir_resource_converter.create_immunization(
        immunization=immunization,
        immunization_id="imm-id",
        patient_uri_ref="urn:uuid:patient-id",
    )
    self.assertEqual(imm.vaccine_code.text.value, "BCG Vaccine")
    self.assertEmpty(imm.vaccine_code.coding)
    self.assertFalse(imm.primary_source.value)
    self.assertEmpty(imm.performer)
    self.assertFalse(imm.HasField("encounter"))

  def test_create_immunization_missing_immunization(self):
    with self.assertRaises(ValueError):
      abdm_fhir_resource_converter.create_immunization(
          None, "imm-id", "urn:uuid:patient-id"
      )

  def test_create_immunization_occurrence_date_datetime_override(self):
    immunization = resources.Immunization(
        vaccine_name="Covishield",
        date=datetime.date(2026, 1, 1),
    )
    custom_dt = datetime.datetime(
        2026, 2, 1, 10, 30, tzinfo=datetime.timezone.utc
    )
    imm = abdm_fhir_resource_converter.create_immunization(
        immunization=immunization,
        immunization_id="imm-id",
        patient_uri_ref="urn:uuid:patient-id",
        occurrence_date=custom_dt,
    )
    expected_us = int(custom_dt.timestamp() * 1000000)
    self.assertEqual(imm.occurrence.date_time.value_us, expected_us)

  def test_create_immunization_occurrence_date_date_override(self):
    immunization = resources.Immunization(
        vaccine_name="Covishield",
    )
    custom_date = datetime.date(2026, 3, 15)
    imm = abdm_fhir_resource_converter.create_immunization(
        immunization=immunization,
        immunization_id="imm-id",
        patient_uri_ref="urn:uuid:patient-id",
        occurrence_date=custom_date,
    )
    expected_us = int(
        datetime.datetime(2026, 3, 15, tzinfo=datetime.timezone.utc).timestamp()
        * 1000000
    )
    self.assertEqual(imm.occurrence.date_time.value_us, expected_us)


if __name__ == "__main__":
  absltest.main()

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
"""Converts LabReport medical documents to ABDM FHIR resources."""

import collections
from collections.abc import Sequence
import datetime
import logging
from typing import Any
import uuid

from google.fhir.r4.proto.core import codes_pb2
from google.fhir.r4.proto.core import datatypes_pb2
from google.fhir.r4.proto.core.resources import bundle_and_contained_resource_pb2 as fhir_pb2
from google.fhir.r4.proto.core.resources import composition_pb2
from google.fhir.r4.proto.core.resources import diagnostic_report_pb2
from google.fhir.r4.proto.core.resources import encounter_pb2
from google.fhir.r4.proto.core.resources import observation_pb2
from google.fhir.r4.proto.core.resources import organization_pb2
from google.fhir.r4.proto.core.resources import patient_pb2
from google.fhir.r4.proto.core.resources import practitioner_pb2
from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.common.schema import resources
from src.document_to_fhir.common.schema.abdm import abdm_medical_documents
from src.document_to_fhir.core.fhir import fhir_generator
from src.document_to_fhir.core.fhir import fhir_utils
from src.document_to_fhir.core.fhir.abdm import abdm_fhir_resource_converter as resource_converter


class AbdmLabReportFhirGenerator(fhir_generator.IFhirGenerator):
  """Generates FHIR resources from a LabReport medical document.

  This class converts a structured LabReport object into a FHIR Bundle
  adhering to the ABDM DiagnosticReportRecord profile.
  """

  _RESOURCE_TYPE_MAP = {
      patient_pb2.Patient: "patient",
      practitioner_pb2.Practitioner: "practitioner",
      organization_pb2.Organization: "organization",
      encounter_pb2.Encounter: "encounter",
      observation_pb2.Observation: "observation",
      diagnostic_report_pb2.DiagnosticReport: "diagnostic_report",
      composition_pb2.Composition: "composition",
  }

  def generate_fhir(
      self, medical_document: medical_documents.MedicalDocument
  ) -> fhir_pb2.Bundle:
    if not isinstance(medical_document, abdm_medical_documents.AbdmLabReport):
      raise TypeError(
          "AbdmLabReportFhirGenerator only processes AbdmLabReport documents."
      )
    lab_report: abdm_medical_documents.AbdmLabReport = medical_document
    if not lab_report.patient:
      raise ValueError("LabReport must have a patient.")

    patient_id = str(uuid.uuid4())
    practitioner_id = str(uuid.uuid4())
    organization_id = str(uuid.uuid4())
    encounter_id = str(uuid.uuid4())
    composition_id = str(uuid.uuid4())

    patient = resource_converter.create_patient(lab_report.patient, patient_id)
    patient_ref = f"urn:uuid:{patient.id.value}"

    practitioner = None
    practitioner_ref = None
    if lab_report.practitioner:
      practitioner = resource_converter.create_practitioner(
          lab_report.practitioner,
          practitioner_id=practitioner_id,
      )
      practitioner_ref = f"urn:uuid:{practitioner.id.value}"

    organization = None
    organization_ref = None
    if lab_report.service_provider:
      organization = resource_converter.create_organization(
          lab_report.service_provider, organization_id
      )
      organization_ref = f"urn:uuid:{organization.id.value}"

    collection_dt = None
    if lab_report.sample_collection_time:
      collection_dt = fhir_utils.to_fhir_datetime(
          lab_report.sample_collection_time
      )

    encounter = resource_converter.create_encounter(
        encounter_id=encounter_id,
        patient_uri_ref=patient_ref,
        period_start_dt=collection_dt,
        practitioner_uri_ref=practitioner_ref,
        organization_uri_ref=organization_ref,
    )
    encounter_ref = f"urn:uuid:{encounter.id.value}"

    # List to hold resources, Composition MUST be first in the bundle for a
    # DOCUMENT.
    resources_to_bundle = []

    # Create Master Observations and Test Observations first to get their IDs for
    # Composition
    panels, unpanelled_tests = _group_tests_by_panel(lab_report.lab_tests)
    diagnostic_report_result_refs = []

    for panel_name, panel_data in panels.items():
      member_refs = []
      for test in panel_data["tests"]:
        obs_id = str(uuid.uuid4())
        member_refs.append(f"urn:uuid:{obs_id}")

        observation = resource_converter.create_lab_observation(
            test,
            obs_id,
            patient_ref,
            None,
            collection_dt,
            organization_ref,
        )
        resources_to_bundle.append(observation)

      master_obs_id = str(uuid.uuid4())
      diagnostic_report_result_refs.append(f"urn:uuid:{master_obs_id}")

      master_observation = resource_converter.create_panel_observation(
          master_obs_id,
          patient_ref,
          collection_dt,
          panel_name,
          panel_data["loinc"],
          member_refs,
      )
      resources_to_bundle.append(master_observation)

    for test in unpanelled_tests:
      obs_id = str(uuid.uuid4())
      diagnostic_report_result_refs.append(f"urn:uuid:{obs_id}")

      observation = resource_converter.create_lab_observation(
          test,
          obs_id,
          patient_ref,
          None,
          collection_dt,
          organization_ref,
      )
      resources_to_bundle.append(observation)

    # Create single Diagnostic Report
    report_id = str(uuid.uuid4())
    report = resource_converter.create_diagnostic_report(
        report_id,
        patient_ref,
        practitioner_ref,
        organization_ref,
        encounter_ref,
        diagnostic_report_result_refs,
        [],
        collection_dt,
    )
    resources_to_bundle.append(report)

    # Create Composition
    composition = composition_pb2.Composition(
        id=datatypes_pb2.Id(value=composition_id),
        meta=datatypes_pb2.Meta(
            profile=[
                datatypes_pb2.Canonical(
                    value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportRecord"
                )
            ]
        ),
        status=composition_pb2.Composition.StatusCode(
            value=codes_pb2.CompositionStatusCode.FINAL
        ),
        type=datatypes_pb2.CodeableConcept(
            coding=[
                datatypes_pb2.Coding(
                    system=datatypes_pb2.Uri(value="http://snomed.info/sct"),
                    code=datatypes_pb2.Code(value="721981007"),
                    display=datatypes_pb2.String(
                        value="Diagnostic studies report"
                    ),
                )
            ]
        ),
        subject=datatypes_pb2.Reference(
            uri=datatypes_pb2.String(value=patient_ref)
        ),
        date=collection_dt
        or fhir_utils.to_fhir_datetime(
            datetime.datetime.now(datetime.timezone.utc)
        ),
        title=datatypes_pb2.String(value="Diagnostic Report Record"),
    )
    if practitioner_ref:
      composition.author.add(uri=datatypes_pb2.String(value=practitioner_ref))
    elif organization_ref:
      composition.author.add(uri=datatypes_pb2.String(value=organization_ref))

    # Add section to Composition
    section = composition.section.add()
    section.title.value = "Diagnostic Report"
    entry = section.entry.add()
    entry.uri.value = f"urn:uuid:{report.id.value}"
    entry.type.value = "DiagnosticReport"

    # Now add all resources to the final list in the correct order
    # 1. Composition
    # 2. Patient
    # 3. Practitioner
    # 4. Organization
    # 5. Encounter
    # 6. Observations (already in resources_to_bundle)
    # 7. DiagnosticReports (already in resources_to_bundle)

    ordered_resources = [
        composition,
        patient,
        encounter,
    ]
    if practitioner:
      ordered_resources.insert(2, practitioner)
    if organization:
      ordered_resources.insert(3, organization)
    ordered_resources.extend(resources_to_bundle)

    return self._create_bundle(ordered_resources)

  def _create_bundle(
      self,
      fhir_resources: Sequence[Any],
      bundle_id: str | None = None,
  ) -> fhir_pb2.Bundle:
    """Creates a FHIR Document Bundle containing the provided resources."""
    if not bundle_id:
      bundle_id = str(uuid.uuid4())
    bundle = fhir_pb2.Bundle(
        id={"value": bundle_id},
        identifier=datatypes_pb2.Identifier(
            system=datatypes_pb2.Uri(value="http://hip.in"),
            value=datatypes_pb2.String(value=bundle_id),
        ),
        type={"value": codes_pb2.BundleTypeCode.DOCUMENT},
        timestamp=datatypes_pb2.Instant(
            value_us=int(
                datetime.datetime.now(datetime.timezone.utc).timestamp()
            )
            * 1000000,
            precision=datatypes_pb2.Instant.Precision.SECOND,
            timezone="UTC",
        ),
        meta=datatypes_pb2.Meta(
            version_id=datatypes_pb2.Id(value="1"),
            profile=[
                datatypes_pb2.Canonical(
                    value="https://nrces.in/ndhm/fhir/r4/StructureDefinition/DocumentBundle"
                )
            ],
        ),
    )
    for resource in fhir_resources:
      resource_field_name = self._RESOURCE_TYPE_MAP.get(type(resource))
      if resource_field_name:
        full_url_value = f"urn:uuid:{resource.id.value}"
        entry = bundle.entry.add(full_url={"value": full_url_value})
        getattr(entry.resource, resource_field_name).CopyFrom(resource)
    return bundle


def _group_tests_by_panel(
    tests: Sequence[resources.LabTest],
) -> tuple[dict[str, dict[str, Any]], list[resources.LabTest]]:
  """Groups tests by panel name."""
  panels = collections.defaultdict(
      lambda: {
          "loinc": None,
          "name": None,
          "tests": [],
      }
  )
  unpanelled_tests: list[resources.LabTest] = []

  for test in tests:
    if test.panel_name:
      panel_name = test.panel_name
      panel_data = panels[panel_name]

      if panel_data["name"] is None:
        panel_data["name"] = panel_name
        panel_data["loinc"] = test.panel_loinc_code
      elif (
          test.panel_loinc_code and panel_data["loinc"] != test.panel_loinc_code
      ):
        logging.warning(
            "Conflicting panel_loinc_code found for panel '%s'.", panel_name
        )

      panel_data["tests"].append(test)
    else:
      unpanelled_tests.append(test)
  return dict(panels), unpanelled_tests

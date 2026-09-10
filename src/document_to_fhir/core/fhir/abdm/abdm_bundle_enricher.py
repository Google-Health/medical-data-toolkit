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
"""Helper for enriching FHIR bundles with additional resources."""

import dataclasses
import datetime
import json
from typing import Any
import uuid

from google.fhir.r4.proto.core.resources import bundle_and_contained_resource_pb2
from google.fhir.r4 import json_format
from src.document_to_fhir.core.fhir.abdm import abdm_fhir_resource_converter as resource_converter


@dataclasses.dataclass(frozen=True)
class _DocumentReferenceMetadata:
  """Metadata for creating a DocumentReference for a supported profile."""

  type_code: str
  type_display: str


_SUPPORTED_COMPOSITION_PROFILES: dict[str, _DocumentReferenceMetadata] = {
    "https://nrces.in/ndhm/fhir/r4/StructureDefinition/DiagnosticReportRecord": (
        _DocumentReferenceMetadata(
            type_code="721981007",
            type_display="Diagnostic studies report",
        )
    ),
    "https://nrces.in/ndhm/fhir/r4/StructureDefinition/PrescriptionRecord": (
        _DocumentReferenceMetadata(
            type_code="440545006",
            type_display="Prescription record",
        )
    ),
    "https://nrces.in/ndhm/fhir/r4/StructureDefinition/ImmunizationRecord": (
        _DocumentReferenceMetadata(
            type_code="41000179103",
            type_display="Immunization record",
        )
    ),
}


def add_document_reference_to_bundle(
    fhir_bundle: dict[str, Any] | None, encoded_data: str, mime_type: str
) -> tuple[dict[str, Any] | None, bool]:
  """Converts fhir_bundle to proto, adds DocumentReference if it's a supported ABDM document record.

  Args:
    fhir_bundle: The FHIR bundle as a dictionary.
    encoded_data: The base64 encoded source document data.
    mime_type: The mime type of the source document.

  Returns:
    A tuple containing the (possibly updated) fhir_bundle and a boolean
    indicating if it was modified.
  """
  if not fhir_bundle:
    return fhir_bundle, False

  # 1. Convert Dict -> JSON String -> Proto
  bundle_json_str = json.dumps(fhir_bundle)
  bundle_proto = json_format.json_fhir_string_to_proto(
      bundle_json_str, bundle_and_contained_resource_pb2.Bundle
  )

  # 2. Check if it is a supported document profile from ABDM generator
  if not (
      bundle_proto.entry
      and bundle_proto.entry[0].resource.HasField("composition")
  ):
    return fhir_bundle, False

  comp = bundle_proto.entry[0].resource.composition
  composition_profiles = [p.value for p in comp.meta.profile]

  matched_profile_meta = None
  for p in composition_profiles:
    if p in _SUPPORTED_COMPOSITION_PROFILES:
      matched_profile_meta = _SUPPORTED_COMPOSITION_PROFILES[p]
      break

  if not matched_profile_meta:
    return fhir_bundle, False

  # 3. Find patient reference
  patient_ref = None
  for entry in bundle_proto.entry:
    if entry.resource.HasField("patient"):
      patient_ref = entry.full_url.value
      break

  if not patient_ref:
    return fhir_bundle, False

  doc_ref_id = str(uuid.uuid4())
  doc_ref_proto = resource_converter.create_document_reference(
      doc_ref_id=doc_ref_id,
      patient_ref=patient_ref,
      mime_type=mime_type,
      encoded_data=encoded_data,
      creation_time=datetime.datetime.now(datetime.timezone.utc),
      type_code=matched_profile_meta.type_code,
      type_display=matched_profile_meta.type_display,
      type_system="http://snomed.info/sct",
  )

  # 4. Add DocumentReference to bundle proto
  entry = bundle_proto.entry.add()
  entry.full_url.value = f"urn:uuid:{doc_ref_id}"
  entry.resource.document_reference.CopyFrom(doc_ref_proto)

  # 5. Link in Composition if exists
  if comp.section:
    section = comp.section[0]
    new_entry = section.entry.add()
    new_entry.uri.value = f"urn:uuid:{doc_ref_id}"
    new_entry.type.value = "DocumentReference"

  # 6. Convert back to dict
  return json.loads(json_format.print_fhir_to_json_string(bundle_proto)), True

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
"""Document types for the MDDAS system."""

import enum


class MedicalDocumentType(str, enum.Enum):
  LABORATORY_REPORT = "LABORATORY_REPORT"
  PRESCRIPTION = "PRESCRIPTION"
  DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
  IMMUNIZATION = "IMMUNIZATION"
  OP_CONSULTATION = "OP_CONSULTATION"
  HEALTH_DOCUMENT_RECORD = "HEALTH_DOCUMENT_RECORD"
  WELLNESS_RECORD = "WELLNESS_RECORD"
  INVOICE = "INVOICE"
  SMART_REPORT = "SMART_REPORT"
  NON_MEDICAL = "NON_MEDICAL"


class DocumentStandardizationPolicy(str, enum.Enum):
  """Policy for standardizing documents with unsupported types.

  Attributes:
    ACCEPT_ALL: Accepts the document regardless of unsupported types.
    ALLOW_UNSUPPORTED_NON_MEDICAL: Process only supported segments if
      unsupported types are only NON_MEDICAL.
    ALLOW_ONLY_SUPPORTED: Discard the document if any unsupported type is
      present.
  """

  ACCEPT_ALL = "ACCEPT_ALL"
  ALLOW_UNSUPPORTED_NON_MEDICAL = "ALLOW_UNSUPPORTED_NON_MEDICAL"
  ALLOW_ONLY_SUPPORTED = "ALLOW_ONLY_SUPPORTED"

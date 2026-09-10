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
"""Configuration constants and schema mappings for RSNA / LOINC Radiology Playbook."""

# Official LOINC/RSNA Playbook CSV Column Headers
COL_LOINC_NUMBER = "LoincNumber"
COL_LONG_COMMON_NAME = "LongCommonName"
COL_PART_NUMBER = "PartNumber"
COL_PART_TYPE_NAME = "PartTypeName"
COL_PART_NAME = "PartName"
COL_PART_SEQUENCE_ORDER = "PartSequenceOrder"
COL_RID = "RID"
COL_PREFERRED_NAME = "PreferredName"
COL_RPID = "RPID"
COL_LONG_NAME = "LongName"

# Official RSNA Part Type Names from LoincRsnaRadiologyPlaybook.csv
PART_TYPE_MODALITY_TYPE = "Rad.Modality.Modality Type"
PART_TYPE_MODALITY_SUBTYPE = "Rad.Modality.Modality Subtype"
PART_TYPE_REGION_IMAGED = "Rad.Anatomic Location.Region Imaged"
PART_TYPE_IMAGING_FOCUS = "Rad.Anatomic Location.Imaging Focus"
PART_TYPE_SUBSTANCE_GIVEN = "Rad.Pharmaceutical.Substance Given"
PART_TYPE_ROUTE = "Rad.Pharmaceutical.Route"
PART_TYPE_VIEW_TYPE = "Rad.View.View Type"
PART_TYPE_VIEW_AGGREGATION = "Rad.View.Aggregation"
PART_TYPE_LATERALITY = "Rad.Anatomic Location.Laterality"
PART_TYPE_LATERALITY_PRESENCE = "Rad.Anatomic Location.Laterality.Presence"
PART_TYPE_MANEUVER = "Rad.Maneuver.Maneuver Type"
PART_TYPE_GUIDANCE_ACTION = "Rad.Guidance for.Action"
PART_TYPE_GUIDANCE_APPROACH = "Rad.Guidance for.Approach"
PART_TYPE_GUIDANCE_OBJECT = "Rad.Guidance for.Object"
PART_TYPE_GUIDANCE_PRESENCE = "Rad.Guidance for.Presence"
PART_TYPE_REASON_FOR_EXAM = "Rad.Reason for Exam"
PART_TYPE_SUBJECT = "Rad.Subject"
PART_TYPE_TIMING = "Rad.Timing"

# Normalized Target Index Column Headers
FIELD_LOINC_NUM = "LOINC_NUM"
FIELD_LONG_COMMON_NAME = "LONG_COMMON_NAME"
FIELD_MODALITY_TYPE = "MODALITY_TYPE"
FIELD_MODALITY_SUBTYPE = "MODALITY_SUBTYPE"
FIELD_REGION_IMAGED = "REGION_IMAGED"
FIELD_CONTRAST_USED = "CONTRAST_USED"
FIELD_VIEW_TYPE = "VIEW_TYPE"
FIELD_LATERALITY = "LATERALITY"
FIELD_COMMON_TEST_RANK = "COMMON_TEST_RANK"
FIELD_RPID = "RPID"
FIELD_SEARCH_TEXT = "SEARCH_TEXT"

INDEX_FIELDNAMES = (
    FIELD_LOINC_NUM,
    FIELD_LONG_COMMON_NAME,
    FIELD_MODALITY_TYPE,
    FIELD_MODALITY_SUBTYPE,
    FIELD_REGION_IMAGED,
    FIELD_CONTRAST_USED,
    FIELD_VIEW_TYPE,
    FIELD_LATERALITY,
    FIELD_COMMON_TEST_RANK,
    FIELD_RPID,
    FIELD_SEARCH_TEXT,
)

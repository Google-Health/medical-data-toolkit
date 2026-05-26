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
"""Defines common type schemas and Pydantic models used in LOINC mapping."""

from typing import Optional
import pydantic


class LoincRow(pydantic.BaseModel):
  """Represents a candidate LOINC record returned by query engines."""

  loinc_num: str = pydantic.Field(alias="LOINC_NUM")
  core_analyte: Optional[str] = pydantic.Field(
      default=None, alias="core_analyte"
  )
  long_common_name: Optional[str] = pydantic.Field(
      default=None, alias="LONG_COMMON_NAME"
  )
  system: Optional[str] = pydantic.Field(default=None, alias="SYSTEM")
  property: Optional[str] = pydantic.Field(default=None, alias="PROPERTY")
  scale_typ: Optional[str] = pydantic.Field(default=None, alias="SCALE_TYPE")
  common_test_rank: Optional[int] = pydantic.Field(
      default=None, alias="COMMON_TEST_RANK"
  )

  model_config = pydantic.ConfigDict(populate_by_name=True)

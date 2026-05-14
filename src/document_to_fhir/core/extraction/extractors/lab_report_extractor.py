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
"""Lab report extractor."""

import os
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import medical_documents
from src.document_to_fhir.core.extraction import medical_extractor


class LabReportExtractor(medical_extractor.MedicalExtractor):
  """Extractor for Lab Reports."""

  def __init__(
      self,
      client: model_client.LLMClient,
      schema: type[medical_documents.LabReport],
      prompt: str | None = None,
  ):
    if prompt is None:
      # The default prompt is a suggestion. Users are advised to update the
      # prompt based on their specific use case and performance expectations.
      prompt_path = os.path.join(
          os.path.dirname(__file__),
          "..",
          "suggested_prompts",
          "lab_report.jinja2",
      )
      prompt = medical_extractor.read_prompt(prompt_path)
    super().__init__(client, prompt, schema)

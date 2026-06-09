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
"""Extractor for medical documents."""

import abc
import logging

from google.genai import types

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import medical_documents


def read_prompt(prompt_path: str) -> str:
  """Reads the prompt from the given path."""
  with open(prompt_path, "rt") as f:
    return f.read().strip()


class MedicalExtractor(abc.ABC):
  """Extracts structured medical data from documents using an LLM."""

  def __init__(
      self,
      client: model_client.LLMClient,
      prompt: str,
      schema: type[medical_documents.MedicalDocument],
  ):
    self.client = client
    self.prompt = prompt
    self.schema = schema

  def extract(
      self, images: list[bytes], mime_type: str = "image/png"
  ) -> medical_documents.MedicalDocument:
    """Extracts structured data from document images.

    Args:
      images: A list of image bytes to extract data from.
      mime_type: The MIME type of the images.

    Returns:
      The parsed object matching the target schema.
    """
    # Construct request contents (Prompt + Media).
    request_contents = [
        types.Part.from_bytes(data=img, mime_type=mime_type) for img in images
    ]
    request_contents.append(self.prompt)

    def post_process_extractor(parsed_json):
      if isinstance(parsed_json, list) and len(parsed_json) == 1:
        logging.warning(
            "Post-processing Unwrapping response from list of length 1"
        )
        return parsed_json[0]
      return parsed_json

    response = self.client.generate_content(
        contents=request_contents,
        schema=self.schema,
        config={
            "response_mime_type": "application/json",
        },
        post_process=post_process_extractor,
    )
    if response.parsed:
      return response.parsed

    # Fallback for manual parsing
    try:
      clean_text = llm_util.extract_json_from_llm_response(response.text)
      return self.schema.model_validate_json(clean_text)
    except Exception as e:
      logging.debug("Failed response text: %s", response.text)
      logging.exception("Failed to parse response")
      raise ValueError("Failed to parse LLM response.") from e

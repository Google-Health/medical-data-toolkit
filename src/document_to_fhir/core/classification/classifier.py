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
"""Medical document classification library."""

import abc
import logging
import os
from typing import Any, Union

from google.genai import types

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common import pdf_util
from src.document_to_fhir.common.schema import standardized_composite_medical_document


def read_prompt(prompt_path: str) -> str:
  """Reads the prompt from the given path."""
  with open(prompt_path, "rt") as f:
    return f.read().strip()


# Standard DPI for PDF to Image conversion
TARGET_PDF_TO_IMAGE_DPI = 300

# A large chunk size to process the entire document at once.
LARGE_CHUNK_SIZE = 999999


def sort_document_segments(
    doc: standardized_composite_medical_document.CompositeDocument,
) -> standardized_composite_medical_document.CompositeDocument:
  """Sorts the document segments in ascending order by page range.

  Sorts primarily by start_page, and uses end_page as a secondary tie-breaker.

  Args:
      doc: The CompositeDocument containing the list of segments.

  Returns:
      The CompositeDocument with its segments sorted.
  """
  sorted_segments = sorted(
      doc.segments, key=lambda seg: (seg.start_page, seg.end_page)
  )
  return standardized_composite_medical_document.CompositeDocument(
      segments=sorted_segments
  )


class DocumentClassifierBase(abc.ABC):
  """Base class for document classifiers."""

  def __init__(self, client: model_client.LLMClient):
    self.client = client

  @abc.abstractmethod
  def classify(
      self,
      data: Union[bytes, list[bytes]],
      **kwargs,
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Classifies a medical document."""
    pass

  def _chunk_pdf_to_parts(
      self, pdf_bytes: bytes, chunk_size: int, overlap: int
  ) -> list[list[types.Part]]:
    """Splits a PDF into smaller chunks of pages represented as lists of Parts."""
    if overlap >= chunk_size:
      raise ValueError("Chunk size needs to be greater than overlap.")

    images = pdf_util.convert_pdf_pages_to_png_images(
        pdf_bytes, TARGET_PDF_TO_IMAGE_DPI
    )
    total_pages = len(images)

    all_chunks_parts = []
    for start in range(0, total_pages, chunk_size - overlap):
      end = min(start + chunk_size, total_pages)

      current_chunk_parts = [
          types.Part.from_text(text="==Start of Document==\n")
      ]

      for i in range(start, end):
        current_chunk_parts.append(
            types.Part.from_text(text=f"==Screenshot for page {i + 1}==\n")
        )
        current_chunk_parts.append(
            types.Part.from_bytes(data=images[i], mime_type="image/png")
        )
      current_chunk_parts.append(
          types.Part.from_text(text="\n==End of Document==\n\n")
      )

      all_chunks_parts.append(current_chunk_parts)

    return all_chunks_parts

  def _prepare_request_contents(
      self,
      data: Union[bytes, list[bytes]],
      prompt: str,
      mime_type: str = "application/pdf",
      chunk_size: int = 15,
      overlap: int = 1,
  ) -> list[list[Any]]:
    """Prepares the request contents for the LLM client.

    If the client supports PDF and the mime_type is application/pdf, the PDF
    bytes are sent directly. Otherwise, the PDF is converted to PNG images
    (if applicable), and the images are sent.

    Args:
      data: The document data, either as bytes (for PDF or a single image) or a
        list of bytes (for multiple images).
      prompt: The prompt string to be included in the request.
      mime_type: The MIME type of the data. Can be "application/pdf" or
        "image/png".
      chunk_size: The number of pages in each chunk (only for PDF).
      overlap: The number of pages to overlap between consecutive chunks (only
        for PDF).

    Returns:
      A list of lists, where each inner list contains `types.Part` objects and
      the prompt string for a chunk.
    """
    if mime_type == "application/pdf":
      if self.client.supports_pdf:
        return [[
            types.Part.from_bytes(data=data, mime_type=mime_type),
            types.Part.from_text(text=prompt),
        ]]

      pdf_chunks = self._chunk_pdf_to_parts(data, chunk_size, overlap)
      return [
          chunk_parts + [types.Part.from_text(text=prompt)]
          for chunk_parts in pdf_chunks
      ]
    else:  # Images (single or multiple)
      images = [data] if isinstance(data, bytes) else data
      actual_mime_type = (
          mime_type if mime_type.startswith("image/") else "image/png"
      )

      request_contents = [
          types.Part.from_text(text="==Start of Document==\n")
      ]
      for i, img in enumerate(images):
        request_contents.append(
            types.Part.from_text(text=f"==Screenshot for page {i + 1}==\n")
        )
        request_contents.append(
            types.Part.from_bytes(data=img, mime_type=actual_mime_type)
        )
      request_contents.append(
          types.Part.from_text(text="\n==End of Document==\n\n")
      )
      request_contents.append(types.Part.from_text(text=prompt))
      return [request_contents]


class MultiDocumentClassifier(DocumentClassifierBase):
  """Refactored DocumentClassifier for composite documents (multi-page PDFs)."""

  def _get_classification_response(
      self, contents: list[Any], temperature: float
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Gets a classification response from the LLM client.

    Args:
      contents: The document parts and prompt.
      temperature: The temperature for the LLM.

    Returns:
      A CompositeDocument parsed from the LLM's response.
    """

    def post_process_classification(parsed_json):
      if isinstance(parsed_json, list):
        logging.warning(
            "Post-processing: Wrapping list in CompositeDocument segments."
        )
        return {"segments": parsed_json}
      return parsed_json

    response = self.client.generate_content(
        contents=contents,
        schema=standardized_composite_medical_document.CompositeDocument,
        config={
            "temperature": temperature,
        },
        post_process=post_process_classification,
    )

    if response.parsed:
      return response.parsed

    # Fallback for manual parsing
    try:
      clean_text = llm_util.extract_json_from_llm_response(response.text)
      return standardized_composite_medical_document.CompositeDocument.model_validate_json(
          clean_text
      )
    except Exception as e:
      logging.debug("Failed response text: %s", response.text)
      logging.exception("Failed to parse response")
      raise ValueError("Failed to parse LLM response.") from e

  def _merge_outputs(
      self,
      classification_outputs: list[
          standardized_composite_medical_document.CompositeDocument
      ],
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Merges classification outputs from multiple PDF chunks.

    This method takes a list of CompositeDocument objects, each representing the
    classification of a chunk of a larger PDF. It merges these outputs,
    handling overlaps between chunks to produce a single CompositeDocument
    for the entire original PDF.

    Args:
      classification_outputs: A list of CompositeDocument objects, where each
        object contains the classification segments for a PDF chunk.

    Returns:
      A single CompositeDocument containing the merged and de-duplicated
      classification segments from all chunks.
    """
    if not classification_outputs:
      logging.warning("No classification outputs to merge.")
      return standardized_composite_medical_document.CompositeDocument()

    merged_classified_output = classification_outputs[0].segments[:]
    for index in range(1, len(classification_outputs)):
      overlap_segment_left = classification_outputs[index - 1].segments[-1]
      overlap_segment_right = classification_outputs[index].segments[0]

      # Resolve overlaps between adjacent chunks
      if overlap_segment_left.start_page == overlap_segment_right.start_page:
        # Case 1: Segments start on the same page; replace left with right.
        del merged_classified_output[-1]
        merged_classified_output.extend(classification_outputs[index].segments)
      elif overlap_segment_left.end_page == overlap_segment_right.end_page:
        # Case 2: Segment overlaps at the end; skip the first segment of right.
        merged_classified_output.extend(
            classification_outputs[index].segments[1:]
        )
      else:
        # Case 3: Bridge the page boundary gap between segments
        merged_classified_output[-1].end_page = classification_outputs[
            index
        ].segments[0].end_page
        merged_classified_output.extend(
            classification_outputs[index].segments[1:]
        )

    return standardized_composite_medical_document.CompositeDocument(
        segments=merged_classified_output
    )

  def classify(
      self,
      data: Union[bytes, list[bytes]],
      prompt: str | None = None,
      temperature: float = 0.0,
      split_into_chunks: bool = True,
      chunk_size: int = 15,
      mime_type: str = "application/pdf",
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Classifies a multi-page medical document into segments.

    Args:
      data: The document data, either as bytes (for PDF or a single image) or a
        list of bytes (for multiple images).
      prompt: The prompt to use for the classification.
      temperature: The temperature for the LLM.
      split_into_chunks: Whether to split the PDF into chunks before sending to
        the LLM. This is useful for very large PDFs.
      chunk_size: The number of pages in each chunk if `split_into_chunks` is
        True.
      mime_type: The MIME type of the data. Can be "application/pdf" or an image
        mime type (e.g. "image/png").

    Returns:
      A CompositeDocument containing the classified segments of the document.
    """
    if prompt is None:
      # The default prompt is a suggestion. Users are advised to update the
      # prompt based on their specific use case and performance expectations.
      prompt_path = os.path.join(
          os.path.dirname(__file__),
          "suggested_prompts",
          "composite_document_classification.jinja2",
      )
      prompt = read_prompt(prompt_path)

    overlap = 1 if split_into_chunks else 0
    actual_chunk_size = chunk_size if split_into_chunks else LARGE_CHUNK_SIZE

    chunks_of_contents = self._prepare_request_contents(
        data,
        prompt,
        mime_type=mime_type,
        chunk_size=actual_chunk_size,
        overlap=overlap,
    )

    if len(chunks_of_contents) == 1:
      return self._get_classification_response(
          chunks_of_contents[0], temperature
      )

    document_classification_outputs = []
    for index, contents in enumerate(chunks_of_contents):
      logging.warning(
          "Processing chunk %d/%d", index + 1, len(chunks_of_contents)
      )
      composite_document_part = self._get_classification_response(
          contents, temperature
      )

      if composite_document_part is None:
        logging.error("Chunk %d returned None, skipping.", index + 1)
        continue
      document_classification_outputs.append(
          sort_document_segments(composite_document_part)
      )

    merged_output = self._merge_outputs(document_classification_outputs)
    return merged_output

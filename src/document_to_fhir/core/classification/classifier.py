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
import concurrent.futures
import contextvars
import logging
import os
import time
from typing import Any, Optional, Union

from google.genai import types

from src.document_to_fhir.common import llm_util
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import standardized_composite_medical_document


# Context variable to store individual classification call latencies.
classification_latencies_var: contextvars.ContextVar[Optional[list[float]]] = (
    contextvars.ContextVar("classification_latencies", default=None)
)


def read_prompt(prompt_path: str) -> str:
  """Reads the prompt from the given path."""
  with open(prompt_path, "rt") as f:
    return f.read().strip()


# Standard DPI for PDF to Image conversion
TARGET_PDF_TO_IMAGE_DPI = 300

# A large chunk size to process the entire document at once.
LARGE_CHUNK_SIZE = 999999

# Maximum number of parallel workers for classification chunks
MAX_PARALLEL_CLASSIFICATION_WORKERS = 3


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

  @abc.abstractmethod
  def process_handwritten_medical_pages(
      self,
      composite_doc: standardized_composite_medical_document.CompositeDocument,
      **kwargs,
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Marks segments as handwritten if they meet the threshold."""
    pass

  def _chunk_images_to_parts(
      self, images: list[bytes], chunk_size: int, overlap: int, mime_type: str
  ) -> list[list[types.Part]]:
    """Splits a list of images (pages) into smaller chunks represented as lists of Parts."""
    if overlap >= chunk_size:
      raise ValueError("Chunk size needs to be greater than overlap.")

    total_pages = len(images)

    all_chunks_parts = []
    prev_end = 0
    for chunk_id, start in enumerate(
        range(0, total_pages, chunk_size - overlap)
    ):
      end = min(start + chunk_size, total_pages)

      # This is to avoid processing the last chunk if it only contains the
      # overlap of the previous chunk.
      if chunk_id > 0 and end <= prev_end:
        break

      prev_end = end

      current_chunk_parts = [
          types.Part.from_text(text="==Start of Document==\n")
      ]

      for i in range(start, end):
        current_chunk_parts.append(
            types.Part.from_text(text=f"==Screenshot for page {i + 1}==\n")
        )
        current_chunk_parts.append(
            types.Part.from_bytes(data=images[i], mime_type=mime_type)
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
      mime_type: str = "image/png",
      chunk_size: int = 15,
      overlap: int = 1,
  ) -> list[list[Any]]:
    """Prepares the request contents for the LLM client.

    Args:
      data: The document data, either as bytes (for single image) or a list of
        bytes (for multiple images).
      prompt: The prompt string to be included in the request.
      mime_type: The MIME type of the data. Can be "image/png".
      chunk_size: The number of pages in each chunk.
      overlap: The number of pages to overlap between consecutive chunks.

    Returns:
      A list of lists, where each inner list contains `types.Part` objects and
      the prompt string for a chunk.
    """
    images = [data] if isinstance(data, bytes) else data
    actual_mime_type = (
        mime_type if mime_type.startswith("image/") else "image/png"
    )

    image_chunks = self._chunk_images_to_parts(
        images, chunk_size, overlap, actual_mime_type
    )
    return [
        chunk_parts + [types.Part.from_text(text=prompt)]
        for chunk_parts in image_chunks
    ]


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
    """Merges classification outputs from multiple chunks of a document.

    This method takes a list of CompositeDocument objects, each representing the
    classification of a chunk of a larger document. It merges these outputs,
    handling overlaps between chunks to produce a single CompositeDocument
    for the entire original document.

    Args:
      classification_outputs: A list of CompositeDocument objects, where each
        object contains the classification segments for a chunk of images from
        the original document.

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
        merged_classified_output[-1].end_page = (
            classification_outputs[index].segments[0].end_page
        )
        merged_classified_output.extend(
            classification_outputs[index].segments[1:]
        )

    return standardized_composite_medical_document.CompositeDocument(
        segments=merged_classified_output
    )

  def process_handwritten_medical_pages(
      self,
      composite_doc: standardized_composite_medical_document.CompositeDocument,
      handwritten_percent_threshold: int = 33,
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Marks document segments as handwritten if they meet the threshold.

    If a segment's handwritten content percentage is above the threshold and
    it is a medical document, its type is set to HANDWRITTEN.

    Args:
      composite_doc: The composite document containing segments to process.
      handwritten_percent_threshold: Percent threshold for handwritten content.

    Returns:
      The processed composite document with updated segment document types.
    """
    for segment in composite_doc.segments:
      if (
          segment.handwritten_content_percent > handwritten_percent_threshold
          and segment.document_type
          != document_types.MedicalDocumentType.NON_MEDICAL
      ):
        segment.document_type = document_types.MedicalDocumentType.HANDWRITTEN
    return composite_doc

  def _process_chunk(
      self,
      index: int,
      contents: list[Any],
      total_chunks: int,
      temperature: float,
  ) -> tuple[
      Optional[standardized_composite_medical_document.CompositeDocument],
      list[model_client.LLMUsage],
      float,
  ]:
    """Processes a single chunk of document for classification.

    This method is intended to be run in a separate thread. It sets up its own
    local token usage tracking and measures latency.

    Args:
      index: The index of the chunk (0-indexed).
      contents: The contents of the chunk to be sent to the LLM.
      total_chunks: The total number of chunks.
      temperature: The temperature for the LLM.

    Returns:
      A tuple containing:
        - The parsed CompositeDocument (or None if failed).
        - A list of LLMUsage objects collected during this chunk's processing.
        - The latency of the classification call in milliseconds.
    """
    chunk_num = index + 1
    logging.info(
        "Processing classification chunk %d/%d...", chunk_num, total_chunks
    )

    local_usages = []
    token = model_client.token_usage_var.set(local_usages)

    start_time = time.perf_counter()
    try:
      composite_document_part = self._get_classification_response(
          contents, temperature
      )
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Classification chunk %d/%d FAILED with exception",
          chunk_num,
          total_chunks,
      )
      return None, [], 0.0
    finally:
      model_client.token_usage_var.reset(token)

    latency = (time.perf_counter() - start_time) * 1000.0

    usage_str = "N/A"
    if local_usages:
      u = local_usages[-1]
      usage_str = (
          f"[Prompt: {u.prompt_tokens}, Completion: {u.completion_tokens},"
          f" Total: {u.total_tokens}]"
      )

    segments_details = []
    if composite_document_part:
      for i, seg in enumerate(composite_document_part.segments):
        segments_details.append(
            f"  - Segment {i+1}: {seg.document_type} (Pages:"
            f" {seg.start_page}-{seg.end_page})"
        )
    segments_str = (
        "\n".join(segments_details)
        if segments_details
        else "  - No segments identified"
    )

    logging.info(
        "Classification chunk %d/%d completed | "
        f"Tokens: {usage_str} | "
        f"Latency: {latency:.2f}ms\n"
        f"{segments_str}",
        chunk_num,
        total_chunks,
    )

    return composite_document_part, local_usages, latency

  def classify(
      self,
      data: Union[bytes, list[bytes]],
      prompt: str | None = None,
      temperature: float = 0.0,
      split_into_chunks: bool = True,
      chunk_size: int = 15,
      mime_type: str = "image/png",
  ) -> standardized_composite_medical_document.CompositeDocument:
    """Classifies a multi-page medical document into segments.

    Args:
      data: The document data, either as bytes (for single image) or a list of
        list of bytes (for multiple images).
      prompt: The prompt to use for the classification.
      temperature: The temperature for the LLM.
      split_into_chunks: Whether to split the list of images into chunks before
        sending to the LLM. This is useful for very large documents.
      chunk_size: The number of images in each chunk if `split_into_chunks` is
        True.
      mime_type: The MIME type of the data. Can be "image/png" or "image/jpeg".

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

    document_classification_outputs = []
    total_chunks = len(chunks_of_contents)
    if total_chunks == 0:
      return standardized_composite_medical_document.CompositeDocument(
          segments=[]
      )

    parent_latencies_list = classification_latencies_var.get()
    parent_token_usages = model_client.token_usage_var.get()

    futures = []
    # Create an inline thread pool with max_workers capped at the constant
    max_workers = min(total_chunks, MAX_PARALLEL_CLASSIFICATION_WORKERS)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      for index, contents in enumerate(chunks_of_contents):
        ctx = contextvars.copy_context()
        future = executor.submit(
            ctx.run,
            self._process_chunk,
            index,
            contents,
            total_chunks,
            temperature,
        )
        futures.append(future)

      for future in futures:
        try:
          composite_document_part, local_usages, latency = future.result()
        except Exception:  # pylint: disable=broad-except
          logging.exception("Failed to get result from future")
          continue

        if composite_document_part is not None:
          document_classification_outputs.append(
              sort_document_segments(composite_document_part)
          )

        # Merge the thread's results back into the parent's context-based lists
        if parent_latencies_list is not None and latency > 0.0:
          parent_latencies_list.append(latency)
        if parent_token_usages is not None and local_usages:
          parent_token_usages.extend(local_usages)

    if not document_classification_outputs:
      return standardized_composite_medical_document.CompositeDocument(
          segments=[]
      )

    merged_output = self._merge_outputs(document_classification_outputs)
    return merged_output

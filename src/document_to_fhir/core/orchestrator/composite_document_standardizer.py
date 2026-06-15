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
"""Standardizer for composite medical documents (pdfs)."""

import base64
from collections.abc import Mapping
import concurrent.futures
import json
import logging
import time

from google.fhir.r4 import json_format
from src.document_to_fhir.common import pdf_util
from src.document_to_fhir.common.model_client import token_usage_var
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import standardized_composite_medical_document
from src.document_to_fhir.core.classification import classifier as classifier_lib
from src.document_to_fhir.core.fhir.abdm import abdm_bundle_enricher
from src.document_to_fhir.core.orchestrator import medical_document_standardizer


# Standard DPI for PDF to Image conversion
TARGET_PDF_TO_IMAGE_DPI = 300
_MS_PER_SECOND = 1000


class CompositeDocumentStandardizer:
  """High-level orchestrator for composite medical documents (pdfs)."""

  def __init__(
      self,
      classifier: classifier_lib.DocumentClassifierBase,
      standardizers: Mapping[
          document_types.MedicalDocumentType,
          medical_document_standardizer.MedicalDocumentStandardizer,
      ],
      document_standardization_policy: document_types.DocumentStandardizationPolicy = document_types.DocumentStandardizationPolicy.ACCEPT_ALL,
      attach_document_to_bundle: bool = False,
      return_metadata: bool = False,
  ):
    self.classifier = classifier
    self.document_standardizers = standardizers
    self.document_standardization_policy = document_standardization_policy
    self.attach_document_to_bundle = attach_document_to_bundle
    self.return_metadata = return_metadata

  def _process_segment(self, segment_images, mime_type, segment):
    """Processes a single document segment.

    This method takes a segment identified by the classifier, and standardizes
    the segment using the appropriate document standardizer based on its type.

    Args:
      segment_images: A list of images for the pages of this segment.
      mime_type: The mime type of the images being processed.
      segment: The document segment object containing type and page range.

    Returns:
      A tuple containing:
        - StandardizedMedicalDocumentWithContext or None if skipped.
        - Latency in milliseconds.
        - List of LLMUsage objects.
    """
    start_time = time.perf_counter()
    segment_usages = []
    token = token_usage_var.set(segment_usages)
    try:
      document_type = segment.document_type

      standardizer = self.document_standardizers.get(document_type)

      medical_document, fhir_bundle = None, None

      if standardizer:
        medical_document, fhir_bundle = standardizer.standardize(
            segment_images, mime_type=mime_type
        )
      else:
        logging.warning("No standardizer for %s. Skipping.", document_type)

      # Wrap in our Pydantic segment model
      fhir_json = (
          json.loads(json_format.print_fhir_to_json_string(fhir_bundle))
          if fhir_bundle
          else None
      )
      standardized_document = standardized_composite_medical_document.StandardizedMedicalDocumentWithContext(
          document_type=document_type,
          start_page=segment.start_page,
          end_page=segment.end_page,
          medical_document=medical_document,
          fhir_bundle=fhir_json,
      )
      latency_ms = (time.perf_counter() - start_time) * _MS_PER_SECOND
      return standardized_document, latency_ms, segment_usages
    finally:
      token_usage_var.reset(token)

  def _partition_segments(
      self,
      segments: list[standardized_composite_medical_document.DocumentSegment],
  ) -> tuple[
      list[standardized_composite_medical_document.DocumentSegment],
      list[standardized_composite_medical_document.DocumentSegment],
  ]:
    """Partitions segments into those to process and those to pass through.

    Args:
      segments: A list of DocumentSegment objects to be partitioned.

    Returns:
      A tuple (segments_to_process, segments_to_pass_through).
      If segments_to_process is empty, the document should be discarded.
    """
    unsupported_segments = [
        seg
        for seg in segments
        if seg.document_type not in self.document_standardizers
    ]
    has_unsupported_types = len(unsupported_segments) > 0

    is_only_unsupported_type_non_medical = all(
        seg.document_type == document_types.MedicalDocumentType.NON_MEDICAL
        for seg in unsupported_segments
    )

    # Check policy
    if (
        self.document_standardization_policy
        == document_types.DocumentStandardizationPolicy.ALLOW_ONLY_SUPPORTED
        and has_unsupported_types
    ):
      return [], segments

    if (
        self.document_standardization_policy
        == document_types.DocumentStandardizationPolicy.ALLOW_UNSUPPORTED_NON_MEDICAL
        and not is_only_unsupported_type_non_medical
    ):
      return [], segments

    # Proceeding
    segments_to_process = [
        seg
        for seg in segments
        if seg.document_type in self.document_standardizers
    ]
    segments_to_pass_through = [
        seg
        for seg in segments
        if seg.document_type not in self.document_standardizers
    ]

    return segments_to_process, segments_to_pass_through

  def standardize(
      self,
      data: bytes,
      mime_type: str = "application/pdf",
  ) -> (
      standardized_composite_medical_document.StandardizedCompositeMedicalDocumentWithContext
  ):
    """Standardizes a composite medical document from input bytes.

    This method first classifies the document to identify segments of different
    document types. It then standardizes each segment in parallel.

    Args:
      data: The raw bytes of the document (PDF or image).
      mime_type: The format of the data. Defaults to "application/pdf".

    Returns:
      A StandardizedCompositeMedicalDocumentWithContext containing the
      standardized medical documents for each segment, or an empty composite
      document if classification fails.
    """
    start_total = time.perf_counter()
    latencies = {}
    token_usages = {}

    if not (mime_type == "application/pdf" or mime_type.startswith("image/")):
      raise ValueError(
          f"Unsupported mime_type: {mime_type}. "
          "Supported formats are 'application/pdf' and 'image/*'."
      )

    # 1. Classification
    start_classify = time.perf_counter()
    usages_classify = []
    token = token_usage_var.set(usages_classify)
    try:
      composite_doc = self.classifier.classify(
          data, temperature=0, mime_type=mime_type
      )
      composite_doc = self.classifier.process_handwritten_medical_pages(
          composite_doc
      )
    finally:
      token_usage_var.reset(token)

    latencies["classification"] = (
        time.perf_counter() - start_classify
    ) * _MS_PER_SECOND

    # Extract classification tokens
    classify_prompt_tokens = sum(u.prompt_tokens for u in usages_classify)
    classify_completion_tokens = sum(
        u.completion_tokens for u in usages_classify
    )
    classify_total_tokens = sum(u.total_tokens for u in usages_classify)

    token_usages["classification"] = {
        "prompt_tokens": classify_prompt_tokens,
        "completion_tokens": classify_completion_tokens,
        "total_tokens": classify_total_tokens,
    }

    segments_to_process, segments_to_pass_through = self._partition_segments(
        composite_doc.segments
    )

    # Create a standardized composite document with pass through segments
    standardized_composite_document = (
        standardized_composite_medical_document.StandardizedCompositeMedicalDocumentWithContext()
    )
    for segment in segments_to_pass_through:
      standardized_composite_document.add_standardized_document(
          standardized_composite_medical_document.StandardizedMedicalDocumentWithContext(
              document_type=segment.document_type,
              start_page=segment.start_page,
              end_page=segment.end_page,
          )
      )

    if not segments_to_process:
      logging.warning(
          "Unsupported types violate policy. Discarding document. "
          "Passing through segments: %s",
          segments_to_pass_through,
      )
      latencies["total"] = (time.perf_counter() - start_total) * _MS_PER_SECOND
      if self.return_metadata:
        standardized_composite_document.metadata = (
            standardized_composite_medical_document.PipelineMetadata(
                latency_ms=latencies,
                token_usage=token_usages,
            )
        )
      return standardized_composite_document

    # 2. Input Pre-processing
    start_preprocess = time.perf_counter()
    if mime_type == "application/pdf":
      all_pages = pdf_util.convert_pdf_pages_to_png_images(
          data, TARGET_PDF_TO_IMAGE_DPI
      )
      working_mime_type = "image/png"
    else:
      # Assume single image content
      all_pages = [data]
      working_mime_type = mime_type

    total_pages = len(all_pages)
    latencies["pre_processing"] = (
        time.perf_counter() - start_preprocess
    ) * _MS_PER_SECOND

    # 3. Segment Loop (Parallelized)
    start_standardize = time.perf_counter()
    segment_latencies = []
    usages_standardize = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(segments_to_process)
    ) as executor:
      futures = []
      for segment in segments_to_process:
        # Range Validation
        if not (1 <= segment.start_page <= segment.end_page <= total_pages):
          logging.warning(
              "Skipping %s: Invalid page range.", segment.document_type
          )
          continue

        segment_images = all_pages[segment.start_page - 1 : segment.end_page]
        futures.append(
            executor.submit(
                self._process_segment,
                segment_images,
                working_mime_type,
                segment,
            )
        )

      for future in concurrent.futures.as_completed(futures):
        standardized_document, segment_latency, segment_usages = future.result()
        if standardized_document:
          standardized_composite_document.add_standardized_document(
              standardized_document
          )
          segment_latencies.append({
              "document_type": standardized_document.document_type,
              "latency_ms": segment_latency,
              "start_page": standardized_document.start_page,
              "end_page": standardized_document.end_page,
          })
          usages_standardize.extend(segment_usages)

    latencies["standardization"] = {
        "total": (time.perf_counter() - start_standardize) * _MS_PER_SECOND,
        "segments": segment_latencies,
    }

    # Extract standardization tokens
    standardize_prompt_tokens = sum(u.prompt_tokens for u in usages_standardize)
    standardize_completion_tokens = sum(
        u.completion_tokens for u in usages_standardize
    )
    standardize_total_tokens = sum(u.total_tokens for u in usages_standardize)

    token_usages["standardization"] = {
        "prompt_tokens": standardize_prompt_tokens,
        "completion_tokens": standardize_completion_tokens,
        "total_tokens": standardize_total_tokens,
    }

    standardized_composite_document.sort_documents_by_page_number()

    # Attach original file base64 encoded to the first available fhir_bundle
    if self.attach_document_to_bundle:
      encoded_data = base64.b64encode(data).decode("utf-8")

      for doc in standardized_composite_document.standardized_medical_documents:
        if (
            doc.document_type
            == document_types.MedicalDocumentType.LABORATORY_REPORT
        ):
          if doc.fhir_bundle:
            updated_bundle, modified = (
                abdm_bundle_enricher.add_document_reference_to_lab_report(
                    doc.fhir_bundle, encoded_data, mime_type
                )
            )
            if modified:
              doc.fhir_bundle = updated_bundle
              break

    # Calculate total latency
    latencies["total"] = (time.perf_counter() - start_total) * _MS_PER_SECOND

    # Calculate total tokens
    total_prompt_tokens = classify_prompt_tokens + standardize_prompt_tokens
    total_completion_tokens = (
        classify_completion_tokens + standardize_completion_tokens
    )
    total_total_tokens = classify_total_tokens + standardize_total_tokens

    token_usages["total"] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_total_tokens,
    }

    if self.return_metadata:
      standardized_composite_document.metadata = (
          standardized_composite_medical_document.PipelineMetadata(
              latency_ms=latencies,
              token_usage=token_usages,
          )
      )

    # Always log metrics for production visibility
    logging.info(
        (
            "MDDAS Pipeline Metrics: Total Latency: %.2fms (Classify: %.2fms,"
            " Standardize: %.2fms). Tokens: Total %d (Classify: %d,"
            " Standardize: %d)."
        ),
        latencies["total"],
        latencies["classification"],
        latencies["standardization"]["total"],
        total_total_tokens,
        classify_total_tokens,
        standardize_total_tokens,
    )

    return standardized_composite_document

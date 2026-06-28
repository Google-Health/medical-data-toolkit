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
from typing import Any

from google.fhir.r4 import json_format
from src.document_to_fhir.common import model_client
from src.document_to_fhir.common import pdf_util
from src.document_to_fhir.common.schema import document_types
from src.document_to_fhir.common.schema import standardized_composite_medical_document
from src.document_to_fhir.core.classification import classifier as classifier_lib
from src.document_to_fhir.core.fhir.abdm import abdm_bundle_enricher
from src.document_to_fhir.core.orchestrator import medical_document_standardizer


token_usage_var = model_client.token_usage_var


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
      log_metrics: bool = True,
  ):
    self.classifier = classifier
    self.document_standardizers = standardizers
    self.document_standardization_policy = document_standardization_policy
    self.attach_document_to_bundle = attach_document_to_bundle
    self.return_metadata = return_metadata
    self.log_metrics = log_metrics

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
      step_latencies = {}

      if standardizer:
        medical_document, fhir_bundle, step_latencies = (
            standardizer.standardize(segment_images, mime_type=mime_type)
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
      return standardized_document, latency_ms, segment_usages, step_latencies
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
        seg.document_type
        in (
            document_types.MedicalDocumentType.NON_MEDICAL,
            document_types.MedicalDocumentType.SMART_REPORT,
        )
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

  def _log_pipeline_metrics(
      self,
      total_pages: int,
      latencies: dict[str, Any],
      token_usages: dict[str, Any],
      composite_doc: (
          standardized_composite_medical_document.CompositeDocument | None
      ) = None,
  ):
    """Logs detailed pipeline metrics including latencies and token usages."""
    classify_calls = token_usages.get("classification", [])
    classify_lats = latencies.get("classification_calls", [])
    classify_calls_log_parts = []
    for i, u in enumerate(classify_calls):
      lat_str = (
          f" | Latency: {classify_lats[i]:.2f}ms"
          if i < len(classify_lats)
          else ""
      )
      classify_calls_log_parts.append(
          f"  - Call {i+1}: [Prompt Tokens: {u['prompt_tokens']}, Completion"
          f" Tokens: {u['completion_tokens']}, Total Tokens:"
          f" {u['total_tokens']}]"
          f"{lat_str}"
      )

    segments_log_parts = []
    if composite_doc and composite_doc.segments:
      segments_log_parts.append("  Segments:")
      for i, seg in enumerate(composite_doc.segments):
        segments_log_parts.append(
            f"    - Segment {i+1}: {seg.document_type} (Pages:"
            f" {seg.start_page}-{seg.end_page})"
        )
    segments_str = "\n".join(segments_log_parts)

    classify_log = "\n".join(classify_calls_log_parts)
    if segments_str:
      classify_log += f"\n{segments_str}"

    standardize_calls_log_parts = []
    standardize_token_usages = token_usages.get("standardization", [])
    for seg in standardize_token_usages:
      calls_str_list = []
      for i, c in enumerate(seg["calls"]):
        calls_str_list.append(
            f"Call {i+1}: [Prompt Tokens: {c['prompt_tokens']}, Completion"
            f" Tokens: {c['completion_tokens']}, Total Tokens:"
            f" {c['total_tokens']}]"
        )
      calls_str = ", ".join(calls_str_list)
      seg_latency = seg.get("latency_ms", 0.0)

      step_lats = seg.get("step_latencies", {})
      step_lats_str = ""
      if step_lats:
        step_lats_str = (
            f"\n      - Extraction: {step_lats.get('extraction', 0.0):.2f}ms\n "
            "     - Terminology Mapping:"
            f" {step_lats.get('terminology_mapping', 0.0):.2f}ms\n      - FHIR"
            f" Generation: {step_lats.get('fhir_generation', 0.0):.2f}ms"
        )

      standardize_calls_log_parts.append(
          f"  - {seg['document_type']} (Pages:"
          f" {seg['start_page']}-{seg['end_page']}): {calls_str} (Latency:"
          f" {seg_latency:.2f}ms){step_lats_str}"
      )
    standardize_calls_log = "\n".join(standardize_calls_log_parts)

    preprocess_latency = latencies.get("pre_processing", 0.0)
    log_msg = (
        "MDDAS Pipeline Metrics: "
        f"Pages: {total_pages} | "
        f"Total Latency: {latencies['total']:.2f}ms\n"
        f"--- Pre-processing (Latency: {preprocess_latency:.2f}ms) ---\n"
        "--- Classification (Total Latency:"
        f" {latencies['classification']:.2f}ms) ---\n"
        f"{classify_log}\n"
        "--- Standardization (Total Latency:"
        f" {latencies['standardization']['total']:.2f}ms) ---\n"
        f"{standardize_calls_log}"
    )
    logging.info(log_msg)

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
    composite_doc = None

    if mime_type not in ("application/pdf", "image/png", "image/jpeg"):
      raise ValueError(
          f"Unsupported mime_type: {mime_type}. Supported formats are:"
          " 'application/pdf', 'image/png', 'image/jpeg'."
      )

    # 1. Input Pre-processing
    start_preprocess = time.perf_counter()
    if mime_type == "application/pdf":
      all_pages_as_images = pdf_util.convert_pdf_pages_to_png_images(
          data, TARGET_PDF_TO_IMAGE_DPI
      )
      working_mime_type = "image/png"
    else:
      # Assume single image content
      all_pages_as_images = [data]
      working_mime_type = mime_type

    total_pages = len(all_pages_as_images)
    latencies["pre_processing"] = (
        time.perf_counter() - start_preprocess
    ) * _MS_PER_SECOND

    logging.info(
        "Input pre-processing completed | Pages: %d | Latency: %.2fms",
        total_pages,
        latencies["pre_processing"],
    )

    # 2. Classification
    start_classify = time.perf_counter()
    usages_classify = []
    classify_latencies = []
    token = token_usage_var.set(usages_classify)
    token_lats = classifier_lib.classification_latencies_var.set(
        classify_latencies
    )
    try:
      composite_doc = self.classifier.classify(
          all_pages_as_images, temperature=0, mime_type=working_mime_type
      )
      composite_doc = self.classifier.process_handwritten_medical_pages(
          composite_doc
      )
    finally:
      token_usage_var.reset(token)
      classifier_lib.classification_latencies_var.reset(token_lats)
      latencies["classification_calls"] = classify_latencies

    latencies["classification"] = (
        time.perf_counter() - start_classify
    ) * _MS_PER_SECOND

    token_usages["classification"] = [
        {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        for u in usages_classify
    ]

    classify_tokens_str = "N/A"
    if token_usages["classification"]:
      total_prompt = sum(
          u["prompt_tokens"] for u in token_usages["classification"]
      )
      total_completion = sum(
          u["completion_tokens"] for u in token_usages["classification"]
      )
      classify_tokens_str = (
          f"[Prompt: {total_prompt}, Completion: {total_completion}]"
      )

    segments_details = []
    for i, seg in enumerate(composite_doc.segments):
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
        (
            "Document classification completed | Segments: %d | Tokens: %s |"
            " Latency: %.2fms\n%s"
        ),
        len(composite_doc.segments),
        classify_tokens_str,
        latencies["classification"],
        segments_str,
    )

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

    # 3. Segment Loop (Parallelized)
    start_standardize = time.perf_counter()
    segment_latencies = []
    segment_token_usages = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(3, len(segments_to_process))
    ) as executor:
      future_to_segment = {}
      for segment in segments_to_process:
        # Range Validation
        if not (1 <= segment.start_page <= segment.end_page <= total_pages):
          logging.warning(
              "Skipping %s: Invalid page range.", segment.document_type
          )
          continue

        segment_images = all_pages_as_images[
            segment.start_page - 1 : segment.end_page
        ]
        future = executor.submit(
            self._process_segment,
            segment_images,
            working_mime_type,
            segment,
        )
        future_to_segment[future] = segment

      for future in concurrent.futures.as_completed(future_to_segment):
        segment = future_to_segment[future]
        try:
          (
              standardized_document,
              segment_latency,
              segment_usages,
              step_latencies,
          ) = future.result()

          if standardized_document:
            calls_str_list = []
            for i, c in enumerate(segment_usages):
              calls_str_list.append(
                  f"Call {i+1}: [Prompt: {c.prompt_tokens}, Completion:"
                  f" {c.completion_tokens}]"
              )
            calls_str = ", ".join(calls_str_list) if calls_str_list else "N/A"

            step_lats_str = ""
            if step_latencies:
              step_lats_str = (
                  " | Steps: [Extraction:"
                  f" {step_latencies.get('extraction', 0.0):.2f}ms,"
                  " Terminology:"
                  f" {step_latencies.get('terminology_mapping', 0.0):.2f}ms,"
                  " FHIR Gen:"
                  f" {step_latencies.get('fhir_generation', 0.0):.2f}ms]"
              )

            logging.info(
                (
                    "Segment standardization completed | Type: %s | Pages:"
                    " %d-%d | Tokens: %s | Total Segment Latency:"
                    " %.2fms%s"
                ),
                standardized_document.document_type,
                standardized_document.start_page,
                standardized_document.end_page,
                calls_str,
                segment_latency,
                step_lats_str,
            )

            standardized_composite_document.add_standardized_document(
                standardized_document
            )
            segment_latencies.append({
                "document_type": standardized_document.document_type,
                "latency_ms": segment_latency,
                "start_page": standardized_document.start_page,
                "end_page": standardized_document.end_page,
            })
            segment_token_usages.append({
                "document_type": standardized_document.document_type.value,
                "start_page": standardized_document.start_page,
                "end_page": standardized_document.end_page,
                "latency_ms": segment_latency,
                "step_latencies": step_latencies,
                "calls": [
                    {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    }
                    for u in segment_usages
                ],
            })
        except Exception as e:
          logging.error(
              (
                  "Segment standardization FAILED | Type: %s | Pages:"
                  " %d-%d | Error: %s"
              ),
              segment.document_type,
              segment.start_page,
              segment.end_page,
              e,
          )
          raise

    latencies["standardization"] = {
        "total": (time.perf_counter() - start_standardize) * _MS_PER_SECOND,
        "segments": segment_latencies,
    }

    token_usages["standardization"] = segment_token_usages

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

    if self.return_metadata:
      standardized_composite_document.metadata = (
          standardized_composite_medical_document.PipelineMetadata(
              latency_ms=latencies,
              token_usage=token_usages,
          )
      )

    # Log metrics for production visibility if enabled
    if self.log_metrics:
      self._log_pipeline_metrics(
          total_pages, latencies, token_usages, composite_doc
      )

    return standardized_composite_document

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
"""Utility functions for PDF processing and image conversion."""

import concurrent.futures
import io
import threading

from PIL import Image
import pypdfium2 as pdfium

# Module-level lock for PDFium rendering, as it's not thread-safe.
PDFIUM_RENDERER_LOCK = threading.Lock()

# The default DPI of PDFium is 72.
PDFIUM_DEFAULT_DPI = 72

# Maximum number of workers for parallel PDF conversion.
# Limited to a constant to avoid resource exhaustion in multi-core/shared
# environments.
_MAX_PDF_CONVERSION_WORKERS = 8


def extract_pil_images_from_pdf(
    file_content: bytes, target_pdf_to_image_dpi: int
) -> list[Image.Image]:
  """Extracts each page of a PDF and renders it to a PIL Image.

  Args:
    file_content: Raw byte content of the PDF file.
    target_pdf_to_image_dpi: The target DPI for rendering PDF pages.

  Returns:
    A list of PIL Image objects, one for each page.
  """
  pil_images = []
  with PDFIUM_RENDERER_LOCK:
    pdf_file = pdfium.PdfDocument(file_content)
    for page in pdf_file:
      try:
        pil_images.append(
            page.render(
                scale=target_pdf_to_image_dpi / PDFIUM_DEFAULT_DPI
            ).to_pil()
        )
      finally:
        page.close()
    pdf_file.close()
  return pil_images


def _convert_to_png_bytes(pil_image: Image.Image) -> bytes:
  img_byte_arr = io.BytesIO()
  pil_image.save(img_byte_arr, format="PNG", compress_level=1, optimize=False)
  return img_byte_arr.getvalue()


def convert_pdf_pages_to_png_images(
    file_content: bytes,
    target_pdf_to_image_dpi: int,
    max_workers: int | None = None,
) -> list[bytes]:
  """Converts each page of a PDF to a binary PNG image.

  Args:
    file_content: Raw byte content of the PDF file.
    target_pdf_to_image_dpi: The target DPI for rendering PDF pages to images.
    max_workers: The maximum number of workers for parallel PDF conversion.

  Returns:
    A list of byte arrays, each representing a PNG image of a page.
  """
  # 1. Get the list of PIL images
  pil_images = extract_pil_images_from_pdf(
      file_content, target_pdf_to_image_dpi
  )

  if not pil_images:
    return []

  if max_workers is None:
    max_workers = _MAX_PDF_CONVERSION_WORKERS

  num_workers = min(max_workers, len(pil_images))

  with concurrent.futures.ThreadPoolExecutor(
      max_workers=num_workers
  ) as executor:
    # map preserves the order of pages
    png_images = list(executor.map(_convert_to_png_bytes, pil_images))

  return png_images

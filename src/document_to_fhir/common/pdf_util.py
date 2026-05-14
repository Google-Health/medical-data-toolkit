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

import io
import threading

import pypdfium2 as pdfium

# Module-level lock for PDFium rendering, as it's not thread-safe.
PDFIUM_RENDERER_LOCK = threading.Lock()

# The default DPI of PDFium is 72.
PDFIUM_DEFAULT_DPI = 72


# TODO: b/375482381 - Before we use this in production, Ask the security
# team if we can use this library given our current set up and use case.
# Convert each page to a binary PNG image.
def convert_pdf_pages_to_png_images(
    file_content: bytes,
    target_pdf_to_image_dpi: int,
) -> list[bytes]:
  """Converts each page of a PDF to a binary PNG image.

  Args:
    file_content: Raw byte content of the PDF file.
    target_pdf_to_image_dpi: The target DPI for rendering PDF pages to images.

  Returns:
    A list of byte arrays, each representing a PNG image of a page.
  """

  pages_as_image_bytes = []
  # pypdfium2 sometimes goes to bad state.
  # If there are many errors, restart the session.
  with PDFIUM_RENDERER_LOCK:
    pdf_file = pdfium.PdfDocument(file_content)
    for page in pdf_file:
      try:
        pdf_bitmap: pdfium.PdfBitmap = page.render(
            scale=target_pdf_to_image_dpi / PDFIUM_DEFAULT_DPI
        )
        pil_image = pdf_bitmap.to_pil()
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="PNG")
        pages_as_image_bytes.append(img_byte_arr.getvalue())
      finally:
        page.close()
    pdf_file.close()
  return pages_as_image_bytes

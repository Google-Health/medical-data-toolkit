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
"""Tests for pdf_util."""

from unittest import mock

from absl.testing import absltest
from PIL import Image

from src.document_to_fhir.common import pdf_util

# A minimal 1-page PDF document.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
    b"xref\n"
    b"0 4\n"
    b"0000000000 65535 f\n"
    b"0000000009 00000 n\n"
    b"0000000052 00000 n\n"
    b"0000000101 00000 n\n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n"
    b"178\n"
    b"%%EOF\n"
)

# A minimal 2-page PDF document.
MINIMAL_2PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 2/Kids[3 0 R 4 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
    b"4 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R>>endobj\n"
    b"xref\n"
    b"0 5\n"
    b"0000000000 65535 f\n"
    b"0000000009 00000 n\n"
    b"0000000052 00000 n\n"
    b"0000000107 00000 n\n"
    b"0000000168 00000 n\n"
    b"trailer<</Size 5/Root 1 0 R>>\n"
    b"startxref\n"
    b"229\n"
    b"%%EOF\n"
)


class PdfUtilTest(absltest.TestCase):

  def test_convert_pdf_pages_to_png_images_success(self):
    images = pdf_util.convert_pdf_pages_to_png_images(MINIMAL_PDF, 72)
    self.assertLen(images, 1)
    # Check if it looks like a PNG (starts with PNG signature)
    self.assertTrue(images[0].startswith(b"\x89PNG\r\n\x1a\n"))

  def test_convert_pdf_pages_to_png_images_multipage_success(self):
    images = pdf_util.convert_pdf_pages_to_png_images(MINIMAL_2PAGE_PDF, 72)
    self.assertLen(images, 2)
    self.assertTrue(images[0].startswith(b"\x89PNG\r\n\x1a\n"))
    self.assertTrue(images[1].startswith(b"\x89PNG\r\n\x1a\n"))

  @mock.patch.object(pdf_util.concurrent.futures, "ThreadPoolExecutor")
  def test_convert_pdf_pages_to_png_images_multipage_uses_parallel(
      self, mock_executor_class
  ):
    mock_executor = mock_executor_class.return_value.__enter__.return_value
    mock_executor.map.return_value = [b"fake_png_1", b"fake_png_2"]

    images = pdf_util.convert_pdf_pages_to_png_images(MINIMAL_2PAGE_PDF, 72)

    self.assertEqual(images, [b"fake_png_1", b"fake_png_2"])
    mock_executor_class.assert_called_once_with(max_workers=2)

  @mock.patch.object(pdf_util.concurrent.futures, "ThreadPoolExecutor")
  def test_convert_pdf_pages_to_png_images_with_explicit_max_workers(
      self, mock_executor_class
  ):
    mock_executor = mock_executor_class.return_value.__enter__.return_value
    mock_executor.map.return_value = [b"fake_png_1", b"fake_png_2"]

    images = pdf_util.convert_pdf_pages_to_png_images(
        MINIMAL_2PAGE_PDF, 72, max_workers=1
    )

    self.assertEqual(images, [b"fake_png_1", b"fake_png_2"])
    mock_executor_class.assert_called_once_with(max_workers=1)

  @mock.patch.object(Image.Image, "save", autospec=True)
  def test_convert_pdf_pages_to_png_images_save_parameters(self, mock_save):
    # Runs the conversion process
    _ = pdf_util.convert_pdf_pages_to_png_images(MINIMAL_PDF, 72)

    mock_save.assert_called_once()
    _, kwargs = mock_save.call_args
    self.assertEqual(kwargs.get("format"), "PNG")
    self.assertEqual(kwargs.get("compress_level"), 1)
    self.assertEqual(kwargs.get("optimize"), False)


if __name__ == "__main__":
  absltest.main()

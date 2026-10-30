"""
PDF parser based on PyMuPDF.
"""

import os
from typing import Dict, Any, List

import fitz

from .base_parser import BaseParser
from utils.image_filters import repeated_image_paths, should_keep_image


class PDFParser(BaseParser):
    def __init__(self, logger=None, output_images_dir: str = os.path.join("extraction", "output", "default", "images")):
        super().__init__(logger)
        self.current_doc = None
        self.output_images_dir = output_images_dir

    def parse(self, file_path: str) -> Dict[str, Any]:
        if not self.validate_file(file_path):
            self.log_error(f"File invalid: {file_path}")
            return {"success": False, "error": "invalid file"}

        try:
            self.log_info(f"Start parsing PDF: {file_path}")
            doc = fitz.open(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]

            pages = []
            for page_num in range(len(doc)):
                self.log_info(f"Parsing page {page_num + 1}/{len(doc)}")
                pages.append(self.extract_page_content(file_path, page_num))

            self._mark_repeated_images(pages)

            metadata = self.get_metadata(file_path)
            metadata.update({"total_pages": len(doc), "pdf_metadata": doc.metadata})
            doc.close()

            result = {
                "success": True,
                "filename": filename,
                "pages": pages,
                "metadata": metadata,
            }
            self.log_info(f"PDF parsed: {filename}")
            return result
        except Exception as e:
            self.log_error(f"PDF parse failed: {e}")
            return {"success": False, "error": str(e)}

    def _mark_repeated_images(self, pages: List[Dict[str, Any]], min_count: int = 3):
        """Repetition is a review signal, not evidence of irrelevant content.

        The same circuit or diagram may be reused on several teaching pages.
        Keep each occurrence and its file so page evidence remains available.
        """
        all_images = []
        for page in pages:
            all_images.extend(page.get("images", []))

        repeated = repeated_image_paths(all_images, min_count=min_count)
        if not repeated:
            return

        for page in pages:
            candidates = []
            for image_path in page.get("images", []):
                if image_path in repeated:
                    candidates.append({
                        "image_path": image_path,
                        "signature": repeated[image_path],
                        "reason": "repeated_visual_requires_review",
                        "review_status": "pending",
                    })
                    self.log_info(
                        f"Retain repeated PDF visual for review: {os.path.basename(image_path)}"
                    )
            if candidates:
                page["repeated_image_candidates"] = candidates

    def get_page_count(self, file_path: str) -> int:
        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            self.log_error(f"Get page count failed: {e}")
            return 0

    def extract_page_content(self, file_path: str, page_num: int) -> Dict[str, Any]:
        doc = None
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            text = page.get_text()

            images: List[str] = []
            image_list = page.get_images(full=True)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = self.output_images_dir
            os.makedirs(output_dir, exist_ok=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue

                img_path = os.path.join(
                    output_dir,
                    f"{filename}_p{page_num + 1}_img{img_index + 1}.{base_image.get('ext', 'png')}",
                )
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                keep, reason, stats = should_keep_image(img_path)
                if not keep:
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                    self.log_info(
                        f"Skip low-quality PDF image: {os.path.basename(img_path)} | reason={reason} | stats={stats}"
                    )
                    continue

                images.append(img_path)

            result = {
                "page_num": page_num,
                "text": text,
                "images": images,
                # PyMuPDF Page objects become orphaned when their document is
                # closed. Consumers must reopen this source page when needed.
                "raw_page": None,
                "source_page_index": page_num,
                "page_size": {"width": page.rect.width, "height": page.rect.height},
            }
            return result
        except Exception as e:
            self.log_error(f"Extract page content failed (page {page_num}): {e}")
            return {
                "page_num": page_num,
                "text": "",
                "images": [],
                "raw_page": None,
                "source_page_index": page_num,
                "error": str(e),
            }
        finally:
            if doc is not None:
                doc.close()

    def extract_page_images_only(
        self,
        file_path: str,
        page_num: int,
        output_dir: str = os.path.join("extraction", "output", "default", "images"),
    ) -> List[str]:
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            images: List[str] = []
            filename = os.path.splitext(os.path.basename(file_path))[0]
            os.makedirs(output_dir, exist_ok=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if not image_bytes:
                    continue

                img_path = os.path.join(
                    output_dir,
                    f"{filename}_p{page_num + 1}_img{img_index + 1}.{base_image.get('ext', 'png')}",
                )
                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                keep, _, _ = should_keep_image(img_path)
                if not keep:
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                    continue

                images.append(img_path)

            doc.close()
            return images
        except Exception as e:
            self.log_error(f"Extract images failed (page {page_num}): {e}")
            return []

    def extract_page_text_only(self, file_path: str, page_num: int) -> str:
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            text = page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.log_error(f"Extract text failed (page {page_num}): {e}")
            return ""


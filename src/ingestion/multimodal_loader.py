import os
import glob
from typing import List
from pypdf import PdfReader
import logging
from src.ingestion.models import IngestedDocument
from pdf2image import convert_from_path
import pytesseract

logger = logging.getLogger(__name__)

class MultimodalLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def _extract_text_ocr(self, pdf_path: str) -> str:
        """Runs the document through Tesseract OCR using images."""
        logger.info(f"Initiating OCR for Scanned PDF/Image: {pdf_path}")
        text_content = ""
        try:
            # Convert PDF pages to PIL images
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                logger.info(f"Running OCR on page {i+1} of {len(images)}...")
                page_text = pytesseract.image_to_string(image)
                text_content += page_text + "\n"
        except Exception as e:
            logger.error(f"OCR Failed for {pdf_path}: {e}")
        return text_content

    def load_single_document(self, file_path: str) -> IngestedDocument | None:
        """Processes a single PDF file (useful for streaming)."""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
            
        try:
            reader = PdfReader(file_path)
            text_content = ""
            
            # Check for standard text
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_content += extracted + "\n"
            
            source_type = "digital_pdf"
            
            # If text_content is extremely short but there are many pages, it's a scanned PDF
            if len(text_content.strip()) < 50 and len(reader.pages) > 0:
                source_type = "scanned_pdf"
                logger.warning(f"No textual content detected in {file_path}. Applying Image OCR fallback.")
                text_content = self._extract_text_ocr(file_path)
            
            doc = IngestedDocument(
                filename=os.path.basename(file_path),
                content=text_content,
                metadata={
                    "source": file_path, 
                    "total_pages": len(reader.pages),
                    "source_type": source_type
                }
            )
            logger.info(f"Successfully ingested [{source_type}]: {doc.filename} ({len(text_content)} chars).")
            return doc
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            return None

    def load_documents(self) -> List[IngestedDocument]:
        """Scans the directory for PDFs. Uses PyPDF initially, falling back to OCR if scanned."""
        documents = []
        pattern = os.path.join(self.directory_path, "*.pdf")
        files = glob.glob(pattern)
        
        logger.info(f"Found {len(files)} local files to ingest in {self.directory_path}")

        for file_path in files:
            doc = self.load_single_document(file_path)
            if doc:
                documents.append(doc)
        
        return documents

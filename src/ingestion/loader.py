import os
import glob
from typing import List
from pypdf import PdfReader
import logging
from src.ingestion.models import IngestedDocument

logger = logging.getLogger(__name__)

class PDFLoader:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

    def load_documents(self) -> List[IngestedDocument]:
        """Scans the directory for PDFs and converts them to IngestedDocument objects."""
        documents = []
        pattern = os.path.join(self.directory_path, "*.pdf")
        files = glob.glob(pattern)
        
        logger.info(f"Found {len(files)} PDF files in {self.directory_path}")

        for file_path in files:
            try:
                logger.info(f"Processing file: {file_path}")
                reader = PdfReader(file_path)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
                
                doc = IngestedDocument(
                    filename=os.path.basename(file_path),
                    content=text_content,
                    metadata={"source": file_path, "total_pages": len(reader.pages)}
                )
                documents.append(doc)
                logger.info(f"Successfully loaded {doc.filename} with {len(text_content)} characters.")
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
        
        return documents

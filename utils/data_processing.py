# data_processing.py
import os
import logging
import fitz
from models.embedding_model import EmbeddingModel as embedding_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataPreprocessor:
    def __init__(self, embedding_model):
        self.tokenizer = embedding_model.model.tokenizer
        
    def process_documents_from_folder(self, folder_path: str, extensions=(".txt", ".pdf")):
        
        """
        Retrieve paths to documents with specified extensions from a folder and its subdirectories.
        Args:
            folder_path (str): The root directory to search for documents.
            extensions (tuple): A tuple of file extensions to include (e.g., ".txt", ".pdf").
        Returns:
            List[str]: A list of full file paths to the matching documents.        
        """
        document_paths = []
        
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(extensions):  
                    document_paths.append(os.path.join(subdir, file))
        
        return document_paths

    def token_based_chunking(self, document_paths, chunk_size=256, overlap=128):
        
        """
        Splits text into chunks based on token count instead of characters.
        Args:
            text (str): Document text.
            chunk_size (int): Number of tokens per chunk.
            overlap (int): Number of tokens to overlap.
        Returns:
            List of text chunks.
        """
        
        chunks = []
        for path in document_paths:
            logger.info(f"Reading PDF: {path}")
            
            try:
                with fitz.open(path) as doc:
                    text = ""
                    for page in doc:
                        text += page.get_text()
                    
                    # print(text)
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")
                continue
            
            logger.info('Tokenizing document')
            tokens = self.tokenizer.encode(text)

            for i in range(0, len(tokens), chunk_size - overlap):
                logger.info('Creating chunks for embedding process')
                chunk = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk)
                chunks.append(chunk_text)

        logger.info('Chunking complete!')
        
        return chunks

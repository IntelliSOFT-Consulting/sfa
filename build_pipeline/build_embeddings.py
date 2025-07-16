# build_embeddings.py
# from ..models.embedding_model import EmbeddingModel
# from ..utils.pinecone_utils import store_in_pinecone, create_pinecone_index
# from ..utils.data_processing import DataPreprocessor
# from concurrent.futures import ThreadPoolExecutor
# from pinecone import Pinecone, ServerlessSpec
# from src_v2.config import PINECONE_API_KEY
# import logging

from models.embedding_model import EmbeddingModel
from utils.pinecone_utils import store_in_pinecone, create_pinecone_index
from utils.data_processing import DataPreprocessor
from concurrent.futures import ThreadPoolExecutor
from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pc = Pinecone(api_key=PINECONE_API_KEY)

def process_and_store_data_parallel(chunks, embedding_model, index_name):
    
    """
    Embed and store data chunks in Pinecone index in parallel using threading.
    Args:
        chunks (List[str]): List of text chunks to embed and store.
        embedding_model (EmbeddingModel): An instance of the embedding model to generate vector representations.
        index_name (str): The name of the Pinecone index where embeddings will be stored.
    Returns:
        None
    """
    
    with ThreadPoolExecutor() as executor:
        futures = []
        # for chunk in chunks:
        #     vectors = embedding_model.embed(chunk)
        #     futures.append(executor.submit(store_in_pinecone, index_name, vectors))
        
        for i, chunk in enumerate(chunks):
            vector = embedding_model.embed(chunk) 
            id = str(i)
            futures.append(executor.submit(store_in_pinecone, index_name, vector, chunk, id))
    
        logger.info(f'{len(chunks)} chunks upserted')
        
        for future in futures:
            future.result()

def process_and_store_data():
    
    """
    End-to-end pipeline to preprocess raw documents, chunk them, and store embeddings in a Pinecone index.
    Steps:
        1. Load and preprocess documents from a specified folder.
        2. Chunk the documents into manageable text units.
        3. Create or connect to a Pinecone index.
        4. Embed each chunk and store the resulting vectors in the index in parallel.
    Args:
        None
    Returns:
        None
    """
    
    embedding_model = EmbeddingModel()
    data_preprocessor = DataPreprocessor(embedding_model)
    folder_path = "data/raw"
    
    document_paths = data_preprocessor.process_documents_from_folder(folder_path)
    chunks = data_preprocessor.token_based_chunking(document_paths)
    
    index_name = "oncology-index"
    
    # pc.delete_index(index_name)
    # logger.info('Index deleted')
    
    create_pinecone_index(index_name, create_if_missing=True)
    
    # pc = create_pinecone_index(index_name, create_if_missing=True)
    # print(pc.describe_index_stats())
    # sample = pc.fetch(ids=["0"])
    # print(sample)
    
    process_and_store_data_parallel(chunks, embedding_model, index_name)
    logger.info('Building process complete')

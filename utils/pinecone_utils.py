# utils/pinecone_utils.py
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from config import PINECONE_API_KEY
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pc = Pinecone(api_key=PINECONE_API_KEY)

def store_in_pinecone(index_name, vector, text, id):
    
    """
    Store a document embedding in a Pinecone index.
    Args:
        index_name (str): The name of the Pinecone index where the embedding will be stored.
        vector (List[float]): The embedding vector representing the document.
        text (str): The original text associated with the embedding.
        id (str): A unique identifier for the document.
    Returns:
        None
    """
    
    index = pc.Index(index_name)

    # index.upsert(vectors)
    index.upsert([(id, vector, {"text": text})])



def create_pinecone_index(index_name: str, create_if_missing: bool = False):
    
    """
    Return an existing Pinecone index. Optionally create it if it doesn't exist.
    Args:
        index_name (str): The name of the index.
        create_if_missing (bool): Whether to create the index if it doesn't exist.
    Returns:
        pc.Index: The Pinecone index object.
    """

    if not pc.has_index(index_name):
        if create_if_missing:
            try:
                pc.create_index(
                    name=index_name,
                    vector_type="dense",
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                print(f"Created new index: {index_name}")
                logger.info(f'Created new index: {index_name}')
            except Exception as e:
                print(f"Failed to create index: {e}")
                logger.info(f'Failed to create index: {e}')
                raise
        else:
            raise ValueError(f"Index '{index_name}' does not exist. Set `create_if_missing=True` to create it.")

    logger.info(f"Using index: {index_name}")
    return pc.Index(index_name)
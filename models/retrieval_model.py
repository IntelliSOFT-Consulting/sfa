# retrieval_model.py
from models.embedding_model import EmbeddingModel
import logging
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RetrievalModel:
    def __init__(self, index, embedding_model: EmbeddingModel):
        self.index = index
        self.embedding_model = embedding_model

    def get_relevant_documents(self, query: str, top_k: int = 3):
        
        """
        Retrieve the top K most relevant documents from the Pinecone index based on a query.
        Args:
            query (str): The input text query to search for relevant documents.
            top_k (int): The number of top matching documents to retrieve.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the retrieved document content and relevance score.
        """
        
        query_embedding = self.embedding_model.embed(query)
        embeddings = query_embedding
        print(len(embeddings))
        
        results = self.index.query(vector=embeddings, top_k=top_k, include_metadata=True)
        
        matches = results['matches']
        print('Matches')
        print(matches)
        
        logger.info('Vector DB search complete')
        
        if not matches:
            return []
        
        return [
            {"content": match['metadata']['text'], "score": match['score']}
            for match in matches
        ]
        
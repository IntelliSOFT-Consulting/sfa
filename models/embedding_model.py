# embedding_model.py
from sentence_transformers import SentenceTransformer
from typing import Union, List
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

    def embed(self, text: Union[str, List[str]]):
        
        """
        Compute embedding for a given text using a Sentence-Transformers model.
        Args:
            text (Union[str, List[str]]): The input text or list of texts to embed.
        Returns:
            List[float]: The embedding vector for the input text.
        """
        
        # if isinstance(text, str):
        #     text = [text]
            
        # (sentence-level embedding)
        # embeddings = self.model.encode(text, convert_to_tensor=True)
        # vectors = [(str(i), embeddings[i].tolist(), {"text": text[i]}) for i in range(len(text))]

        # return vectors
        
        embedding = self.model.encode([text], convert_to_tensor=True)[0]
        # print(embedding.tolist())

        return embedding.tolist() 
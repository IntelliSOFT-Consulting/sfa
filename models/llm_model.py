# # llm_model.py
# from llama_cpp import Llama
# from cachetools import TTLCache

# class LLMModel:
#     def __init__(self, model_path, n_ctx=2048, n_gpu_layers=35, cache_ttl=600):
        
#         """
#         Initialize the LLM model with caching.
#         Args:
#             model_path (str): The path to the Llama model.
#             n_ctx (int): The context size for the model.
#             n_gpu_layers (int): The number of GPU layers.
#             cache_ttl (int): The time-to-live (TTL) for the cache in seconds. Defaults to 600 seconds (10 minutes).
#         """
        
#         self.model = Llama(
#             model_path=model_path,
#             n_ctx=n_ctx,
#             n_gpu_layers=n_gpu_layers,
#             use_mlock=True,
#             use_mmap=True
#         )
#         self.cache = TTLCache(maxsize=1000, ttl=cache_ttl)

#     def generate_response(self, prompt: str, max_tokens=512, temperature=0.7, top_p=0.95) -> str:
        
#         """
#         Generate a text response from the language model, with caching for repeated prompts.
#         Args:
#             prompt (str): The input prompt to send to the language model.
#             max_tokens (int, optional): Maximum number of tokens to generate. Defaults to 512.
#             temperature (float, optional): Sampling temperature for randomness. Defaults to 0.7.
#             top_p (float, optional): Nucleus sampling parameter; limits to tokens with cumulative probability <= top_p. Defaults to 0.95.
#         Returns:
#             str: The generated response text.
#         """
        
#         if prompt in self.cache:
#             return self.cache[prompt]
        
#         response = self.model(
#             prompt=prompt,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             stop=["</s>", "User:", "Assistant:"],
#             echo=False
#         )
        
#         # Cache the result
#         result = response["choices"][0]["text"].strip()
#         self.cache[prompt] = result
        
#         return result




from openai import OpenAI



class LLMModel:
    def __init__(self, model_name, token):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=token
        )

    def generate_response(self, prompt: str):
        
        """
        Generate a response from a chat-based language model given a user prompt.
        Args:
            prompt (str): The input text prompt to send to the language model.
        Returns:
            str: The generated response text from the model.
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
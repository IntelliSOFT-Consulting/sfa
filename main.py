from build_pipeline.build_embeddings import process_and_store_data
from query_pipeline.query_model import query_pipeline
from config import OPENAI_API_KEY
import sys
import os



if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    prompt_dir = os.path.join(project_root, 'data/prompts/use_cases_prompt.txt')
    llm_dir = os.path.join(project_root, '../llm_models/llama-2-7b-chat.Q4_K_M.gguf')
    hg_token = OPENAI_API_KEY
    
    sample_queries = [            
            "What is the prevalence of cervical cancer in Sub-Saharan Africa?",
            "What are the post-treatment care protocols for cervical cancer patients?",
            "What causes cervical cancer?",]
    use_cases = [
            "public_health_monitoring",
            "clinical decision making",
            "research",
    ]
    
    process = 'query'

    if process == 'build':
        process_and_store_data()
    else:
        query_result = query_pipeline(
            sample_queries[2],
            hg_token,
            use_cases[2],
            prompt_dir
        )
        print('\n\n\n\n\n\n')
        print(query_result)
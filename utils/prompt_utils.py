def load_custom_prompt(prompt_file_path: str) -> str:
    
    """
    Loads the custom prompt from a text file.
    Args:
        prompt_file_path (str): Path to the prompt file.
    Returns:
        str: The custom prompt.
    """
    
    with open(prompt_file_path, 'r', encoding="utf-8") as file:
        prompt = file.read()
        return prompt

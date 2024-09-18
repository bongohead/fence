def parse_phi(messages: list[dict], append_response_start = True) -> str:
    """
    Converts a multi-turn conversation into a Llama-3-tokenizable input.

    Output format:
    # <s><|system|>
    # You are a helpful AI assistant.<|end|>
    # <|user|>
    # Guess my dog's name!<|end|>
    # <|assistant|>
    """
    format = '<s>'
    
    format += '\n'.join([f"<|{m['role']}|>\n{m['content']}<|end|>" for m in messages])

    if append_response_start:
        format += "\n<|assistant|>"
    
    return format

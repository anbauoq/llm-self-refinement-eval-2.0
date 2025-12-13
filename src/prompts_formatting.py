
def gemma_prompt_formatting(prompt: str) -> str:
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model"

def phi4_prompt_formatting(prompt: str) -> str:
    return f"<|user|>{prompt}<|end|><|assistant|>"

def llama_prompt_formatting(prompt: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

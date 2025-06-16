import re

extra_patterns = [
     r'@torch\.no_grad\(\)(?:\s*\n\s*)?', # @torch.no_grad()
]

def remove_extra_code(code: str) -> str:

    for pattern in extra_patterns:
        code = re.sub(pattern, '', code)
    
    return code
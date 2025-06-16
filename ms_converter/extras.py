import re
import ast

def replace_context_blocks(source_code, target_context):
    class ContextReplacer(ast.NodeVisitor):
        def __init__(self, source_lines):
            self.source_lines = source_lines
            self.replacements = []
            
        def visit_With(self, node):
            context_expr = ast.unparse(node.items[0].context_expr).strip()
            
            if context_expr.startswith(target_context):
                start_line = node.lineno
                end_line = max(stmt.end_lineno for stmt in node.body)

                first_line = self.source_lines[start_line-1]
                base_indent = len(first_line) - len(first_line.lstrip())
                indent = ' ' * base_indent
                
                body_code = []
                for stmt in node.body:
                    code = ast.unparse(stmt)
                    body_code.extend(indent + line for line in code.splitlines())
                
                self.replacements.append({
                    'start': start_line,
                    'end': end_line,
                    'content': '\n'.join(body_code)
                })
            
            self.generic_visit(node)

    source_lines = source_code.splitlines()
    tree = ast.parse(source_code)
    replacer = ContextReplacer(source_lines)
    replacer.visit(tree)
    replacer.replacements.sort(key=lambda x: x['start'], reverse=True)
    result_lines = source_lines.copy()
    for replacement in replacer.replacements:
        result_lines[replacement['start']-1:replacement['end']] = [replacement['content']]
    
    return '\n'.join(result_lines)

extra_patterns = [
     r'@torch\.no_grad\(\)(?:\s*\n\s*)?', # @torch.no_grad():
]

def remove_extra_code(code: str, target_contexts=["torch.no_grad", "torch.is_autocast_enabled", "torch.autocast"]) -> str:

    for target_context in target_contexts:
        print(f"Removing context: {target_context}...")
        code = replace_context_blocks(code, target_context)

    for pattern in extra_patterns:
        code = re.sub(pattern, '', code)
    
    return code

if __name__ == '__main__':
    test_cases = [
'''
def some_function():
    print("Before context")
    index = 0
    with torch.no_grad():
        past_key_values = StaticCache(
            model.config,
            batch_size=batch_size,
            device=device,
            dtype=torch.float16,
            max_cache_len=seq_length + num_tokens_to_generate,
        )
        index += 1
    
    print("After context")
    
    with torch.cuda.device(0):
        x = torch.tensor([1, 2, 3])
        y = x * 2
''',
'''
with torch.autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()
    sin = emb.sin()
''',
'''
@torch.no_grad()
def generate(input_ids):
    past_key_values = StaticCache(
        model.config,
        batch_size=batch_size,
        device=device,
        dtype=torch.float16,
        max_cache_len=seq_length + num_tokens_to_generate,
    )
    return model.generate(input_ids, past_key_values=past_key_values)
'''
   ]

    for test_case in test_cases:
        print("#"*40+" Input " + "#"*40)
        print(test_case)
        print("#"*40+" Output " + "#"*40)
        code = remove_extra_code(test_case)
        print(code)
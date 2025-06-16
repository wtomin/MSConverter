import re

param_patterns = [
]

def replace_param_code(code: str) -> str:
    
    pattern = r'(\w+)\s*\.\s*requires_grad_\s*\(\s*(?:requires_grad\s*=\s*)?(True|False)\s*\)'
    replacement = r'\1.requires_grad = \2'
    code = re.sub(pattern, replacement, code)
    for pattern in param_patterns:
        code = re.sub(pattern, '', code)
    
    code = re.sub(r'\(\s*,\s*', '(', code)
    code = re.sub(r',\s*,', ',', code)
    return code

if __name__=="__main__":
    code = [
        '''
    x = torch.randn(3, 4)
    x.requires_grad_(requires_grad=True)
''',
'''
    x = torch.randn(3, 4)
    x.requires_grad_(False)
'''
    ]
    print(replace_param_code(code[0]))
    print(replace_param_code(code[1]))
import re

device_patterns = [
    r'device\s*=\s*[^,\)]+',      # device=xxx 
]

def replace_device_code(code: str) -> str:

    def replace_to_calls(match):
        # .to(device) or .to(x.device) replaced by empty string
        # .to(device, dtype) replaced by .to(dtype)
        params_str = match.group(1)
        params_str = re.sub(r'\s+', ' ', params_str.strip())
        params = []
        for p in params_str.split(','):
            p = p.strip()
            if '=' in p:
                key, value = p.split('=', 1)
                if 'device' not in key:
                    params.append(p)
            else:
                if 'device' not in p:
                    params.append(p)
        
        if params:
            return f".to({', '.join(params)})"
        return ''

    code = re.sub(r'\.to\((.*?)\)', replace_to_calls, code, flags=re.DOTALL)
    
    for pattern in device_patterns:
        code = re.sub(pattern, '', code)
    
    code = re.sub(r'\(\s*,\s*', '(', code)
    code = re.sub(r',\s*,', ',', code)
    
    return code
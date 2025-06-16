import re

device_patterns = [
    r'device[ \t]*=[ \t]*[^,\)\n]+',    # device=xxx 
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

if __name__=="__main__":
    code = '''
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
            )
    '''

    print(replace_device_code(code))
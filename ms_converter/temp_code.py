"""
Temorary code conversion rules:
1. size() -> shape; size(0) -> shape[0];
2. .expand(xxx) -> .broadcast_to((xxx))
3. mint.zeros(s1, s2, s2, **kwargs) -> mint.zeros((s1, s2, s2), **kwargs) | mint.ones
4. mint.randint(low=x, high=x, **kwargs) -> ops.randint(low=x, high=x, **kwargs)
"""
import re

def convert_mint_randint(code: str) -> str:
    pattern = r'mint\.randint\s*\(\s*([^)]*)\s*\)'
    
    def replace_randint(match):
        args = match.group(1).strip()
        if args:
            return f'ops.randint({args})'
        else:
            return 'ops.randint()'
    
    code = re.sub(pattern, replace_randint, code)
    return code

def convert_size_to_shape(code: str) -> str:
    pattern_with_args = r'(\w+)\.size\s*\(\s*([^)]+)\s*\)'
    replacement_with_args = r'\1.shape[\2]'
    code = re.sub(pattern_with_args, replacement_with_args, code)
    
    pattern_no_args = r'(\w+)\.size\s*\(\s*\)'
    replacement_no_args = r'\1.shape'
    code = re.sub(pattern_no_args, replacement_no_args, code)
    
    return code
def convert_mint_x(code: str) -> str:
    pattern = r'mint\.(ones|zeros)\s*\(\s*([^)]*)\s*\)'
    
    def replace_mint_func(match):
        func_name = match.group(1)  # ones, or zeros
        all_args = match.group(2).strip()
        
        if not all_args:
            return match.group(0) 
        
        parts = all_args.split(',')
        pos_args = []
        kw_args = []
        
        for part in parts:
            part = part.strip()
            if '=' in part:
                kw_args.append(part)
            else:
                pos_args.append(part)
        
        if pos_args and kw_args:
            pos_args_str = ', '.join(pos_args)
            kw_args_str = ', '.join(kw_args)
            return f'mint.{func_name}(({pos_args_str}), {kw_args_str})'
        elif pos_args:
            pos_args_str = ', '.join(pos_args)
            return f'mint.{func_name}(({pos_args_str}))'
        else:
            return match.group(0)
    
    code = re.sub(pattern, replace_mint_func, code)
    return code
def convert_expand_to_broadcast(code: str) -> str:
    pattern = r'(\w+)\.expand\s*\(\s*([^)]+)\s*\)'
    
    def replace_expand(match):
        var_name = match.group(1)
        args = match.group(2)
        return f'{var_name}.broadcast_to(({args}))'
    
    code = re.sub(pattern, replace_expand, code)
    return code


def replace_temporary_code(code: str) -> str:
    

    code = convert_size_to_shape(code)
    code = convert_expand_to_broadcast(code)
    code = convert_mint_x(code)
    code = convert_mint_randint(code)

    return code


if __name__ == "__main__":
    code = [
        '''
x = torch.randn(3, 4)
print(x.size())
print(x.size(0))
print(x.size(1))
print(x.size(-1))

tensor_data = torch.tensor([1, 2, 3])
batch_size = tensor_data.size(0)
seq_len = tensor_data.size()

model_output = some_model(input_data)
output_size = model_output.size()
first_dim = model_output.size(0)
second_dim = model_output.size(1)
''',

'''
x = torch.randn(3, 4)
y = x.expand(5, 3, 4)
z = x.expand(2, 3, 4)

tensor_data = torch.tensor([1, 2, 3])
expanded_tensor = tensor_data.expand(5, 3)

model_output = some_model(input_data)
broadcasted_output = model_output.expand(batch_size, seq_len, hidden_size)

result = (x + y).expand(10, 3, 4)
attention_weights = attention_scores.expand(heads, batch_size, seq_len, seq_len)
''',
'''
ones_tensor = mint.ones(3, 4, dtype=ms.float32)
ones_2d = mint.ones(5, 6, dtype=ms.float16)

zeros_tensor_alt = mint.zeros(3, 4, dtype=ms.float32)
zeros_2d_alt = mint.zeros(5, 6, dtype=ms.float16)

ones_3d = mint.ones(2, 3, 4, dtype=ms.int32)
ones_4d = mint.ones(1, 2, 3, 4, dtype=ms.float64)

zeros_3d_alt = mint.zeros(2, 3, 4, dtype=ms.int32)
zeros_4d_alt = mint.zeros(1, 2, 3, 4, dtype=ms.float64)

batch_size = 32
seq_len = 128
hidden_size = 512
zeros_complex = mint.zero(batch_size, seq_len, hidden_size, dtype=ms.float32)
ones_complex = mint.ones(batch_size, seq_len, hidden_size, dtype=ms.float32)
zeros_complex_alt = mint.zeros(batch_size, seq_len, hidden_size, dtype=ms.float32)

simple_zeros = mint.zero(5, 5)
simple_ones = mint.ones(5, 5)
simple_zeros_alt = mint.zeros(5, 5)

zeros_kwargs_only = mint.zero(dtype=ms.float32)
ones_kwargs_only = mint.ones(dtype=ms.float32)
zeros_kwargs_only_alt = mint.zeros(dtype=ms.float32)
''',
        '''

random_tensor = mint.randint(low=0, high=10, size=(3, 4))
random_2d = mint.randint(low=1, high=100, size=(5, 6))

random_with_dtype = mint.randint(low=0, high=255, size=(2, 3, 4), dtype=ms.int32)
random_float = mint.randint(low=0, high=1, size=(10, 10), dtype=ms.float32)

batch_size = 32
seq_len = 128
vocab_size = 1000
random_complex = mint.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
random_kwargs_only = mint.randint(low=0, high=10, size=(5, 5))
''',
    ]
    for i, test_case in enumerate(code):
        print(f"=== Test Case {i+1} ===")
        print("Original:")
        print(test_case)
        print("\nConverted:")
        print(replace_temporary_code(test_case))
        print("\n" + "="*50 + "\n")
    
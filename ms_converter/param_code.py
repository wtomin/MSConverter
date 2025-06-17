import re

torch2ms_mapping = {
    ".named_children()": ".name_cells().items()",
    ".parameters()": ".get_parameters()",
    "@torch.no_grad()": "",
    ".numpy()": ".asnumpy()"
}

def convert_init_code(pytorch_code: str) -> str:
    # 1. Handle special initialization with indices (including fill_ with arguments)
    # module.weight.data[idx].fill_(1) -> module.weight.data[idx] = 1
    pattern_special_index_with_args = r'(\w+)\.(weight|bias|embedding_table)\.data\[(.+?)\]\.(\w+)_\s*\((.*?)\)'
    def replace_special_index_with_args(match):
        var, param_type, idx, func, args = match.groups()
        if func == 'fill':
            return f'{var}.{param_type}[{idx}] = {args}'
        else:
            # 对于其他带参数的函数，保持原始形式
            return f'{var}.{param_type}[{idx}].{func}_({args})'
    
    result = re.sub(pattern_special_index_with_args, replace_special_index_with_args, pytorch_code)
    
    # 2. Handle special initialization with indices and no arguments
    pattern_special_index = r'(\w+)\.(weight|bias|embedding_table)\[(.+?)\]\.(\w+)_\s*\(\)'
    def replace_special_index(match):
        var, param_type, idx, func = match.groups()
        value_map = {
            'zero': '0',
            'ones': '1'
        }
        if func in value_map:
            return f'{var}.{param_type}[{idx}] = {value_map[func]}'
        else:
            return f'{func}_({var}.{param_type}[{idx}])'
    
    result = re.sub(pattern_special_index, replace_special_index, result)
    
    # 3. Handle normal initialization function calls (with arguments)
    pattern = r'(\w+)\.(weight|bias|embedding_table)\.data\.(\w+)_\s*\((.*?)\)'
    replacement = r'\3_(\1.\2, \4)'
    result = re.sub(pattern, replacement, result)
    
    # 4. Handle normal initialization function calls (without arguments)
    pattern_no_args = r'(\w+)\.(weight|bias|embedding_table)\.data\.(\w+)_\s*\(\)'
    replacement_no_args = r'\3_(\1.\2)'
    result = re.sub(pattern_no_args, replacement_no_args, result)
    
    pattern_nn_init = r'nn\.init\.(\w+)_'
    replacement_nn_init = r'\1_'
    result = re.sub(pattern_nn_init, replacement_nn_init, result)
    return result

def replace_param_code(code: str) -> str:
    
    pattern = r'(\w+)\s*\.\s*requires_grad_\s*\(\s*(?:requires_grad\s*=\s*)?(True|False)\s*\)'
    replacement = r'\1.requires_grad = \2'
    code = re.sub(pattern, replacement, code)
    code = convert_init_code(code)
    
    for p_name in torch2ms_mapping.keys():
        ms_name = torch2ms_mapping[p_name]
        code = code.replace(p_name, ms_name)
    
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
''',
'''
@torch.no_grad()
def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
''',
'''
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, AltCLIPVisionEmbeddings):
            factor = self.config.initializer_factor
            nn.init.normal_(module.class_embedding, mean=0.0, std=module.embed_dim**-0.5 * factor)
            nn.init.normal_(module.patch_embedding.weight, std=module.config.initializer_range * factor)
            nn.init.normal_(module.position_embedding.weight, std=module.config.initializer_range * factor)
        elif isinstance(module, AltCLIPAttention):
            factor = self.config.initializer_factor
            in_proj_std = (module.embed_dim**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            out_proj_std = (module.embed_dim**-0.5) * factor
            nn.init.normal_(module.q_proj.weight, std=in_proj_std)
            nn.init.normal_(module.k_proj.weight, std=in_proj_std)
            nn.init.normal_(module.v_proj.weight, std=in_proj_std)
            nn.init.normal_(module.out_proj.weight, std=out_proj_std)
        elif isinstance(module, AltCLIPMLP):
            factor = self.config.initializer_factor
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, AltCLIPModel):
            nn.init.normal_(
                module.text_projection.weight,
                std=module.text_embed_dim**-0.5 * self.config.initializer_factor,
            )
            module.text_projection._is_hf_initialized = True
            nn.init.normal_(
                module.visual_projection.weight,
                std=module.vision_embed_dim**-0.5 * self.config.initializer_factor,
            )
            module.visual_projection._is_hf_initialized = True
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


''',
    ]

    for i, test_case in enumerate(code):
        print(f"=== Test Case {i+1} ===")
        print("Original:")
        print(test_case)
        print("\nConverted:")
        print(replace_param_code(test_case))
        print("\n" + "="*50 + "\n")
    
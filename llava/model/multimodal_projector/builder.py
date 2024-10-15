import torch
import torch.nn as nn
import re

from .honeybee.projectors import CAbstractor
from .honeybee.configuration_honeybee import HoneybeeVisualProjectorConfig

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    num_input_tokens = kwargs.get('num_input_tokens', None)
    c_abs_match = re.match(r'^c_abs(?:_(\d+))?$', projector_type)
    if c_abs_match:
        # Extract the number of query tokens if specified
        num_query_tokens_str = c_abs_match.group(1)
        if num_query_tokens_str is not None:
            honeybee_num_query_tokens = int(num_query_tokens_str)
        else:
            # Use default value or value from config
            honeybee_num_query_tokens = getattr(config, 'honeybee_num_query_tokens', 256)
        print("CONFIGS SIZES:", config.mm_hidden_size, config.hidden_size, num_input_tokens, honeybee_num_query_tokens)
        # CONFIGS SIZES: 1024 5120 576 honeybee_num_query_tokens
        # projector has three inter-module configs:
        # 1) encoder_hidden_size (hidden size of vision model)
        # 2) output_hidden_size (hidden size of LLM)
        # the number of query tokens  (total num_visual_tokens = num_query_tokens + num_eos_tokens)
        proj_config = {
            "projector_type": "c-abs",
            "depth": 3,
            "mlp_depth": 2,
            "hidden_size": 1024 , #1024, #vision_hidden_size, #HoneybeeVisionConfig.from_exp_config(vision_config).hidden_size,
            "num_eos_tokens": 0,
            "pos_emb": True,
            "feature_layer_index": -1,
            "prenorm": False,
            "num_query_tokens": honeybee_num_query_tokens,
            "encoder_hidden_size": config.mm_hidden_size ,# num_input_tokens+1, #+1 to include cls token
            "output_hidden_size": config.hidden_size #5120 #lm_hidden_size, #self.text_config.hidden_size
        }
        projector_config = HoneybeeVisualProjectorConfig(**proj_config)
        return CAbstractor(projector_config, num_input_tokens)

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

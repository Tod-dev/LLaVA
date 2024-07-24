import torch
import torch.nn as nn
import re
from .projectors import CAbstractor
from .configuration_honeybee import HoneybeeVisualProjectorConfig

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


# class SimpleResBlock(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super().__init__()
#         print("ADAPTER is set to -> SimpleResBlock, input_channels: ", input_channels, "output_channels: ", output_channels)
#         self.pre_norm = nn.LayerNorm(input_channels)

#         self.proj = nn.Sequential(
#             nn.Linear(input_channels, input_channels),
#             nn.GELU(),
#             nn.Linear(input_channels, input_channels)
#         )

#         self.adjust_channels = nn.Linear(input_channels, output_channels) if input_channels != output_channels else nn.Identity()

#     def forward(self, x):
#         print("adapter SimpleResBlock forward", x.shape)  # [16, 576, 1024]
#         x = self.pre_norm(x)
#         proj_output = self.proj(x)
#         print("proj_output shape:", proj_output.shape)
#         proj_output = self.adjust_channels(proj_output)
#         print("adjusted proj_output shape:", proj_output.shape)
#         return x + proj_output

class CAbstractorProjector(nn.Module):
    def __init__(self, config, num_input_tokens):
        super().__init__()
        self.cabstractor = CAbstractor(config, num_input_tokens)

    def forward(self, x):
        return self.cabstractor(x)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    num_input_tokens = kwargs.get('num_input_tokens', None)

    print("CONFIGS SIZES:", config.mm_hidden_size, config.hidden_size, num_input_tokens)
    # CONFIGS SIZES: 1024 5120 576

    # num_input_tokens = config.mm_hidden_size

    if projector_type == 'c_abs':
        # projector has three inter-module configs:
        # 1) encoder_hidden_size (hidden size of vision model)
        # 2) output_hidden_size (hidden size of LLM)
        # the number of query tokens  (total num_visual_tokens = num_query_tokens + num_eos_tokens)
        proj_config = {
            # "projector_type": "c-abs",
            "depth": 3,
            "mlp_depth": 2,
            "hidden_size": 1024 , #1024, #vision_hidden_size, #HoneybeeVisionConfig.from_exp_config(vision_config).hidden_size,
            "num_eos_tokens": 0,
            "pos_emb": True,
            # "feature_layer_index": -1,
            "prenorm": False,
            "num_query_tokens": 144,
            "encoder_hidden_size": config.mm_hidden_size ,# num_input_tokens+1, #+1 to include cls token
            "output_hidden_size": config.hidden_size #5120 #lm_hidden_size, #self.text_config.hidden_size
        }
        projector_config = HoneybeeVisualProjectorConfig(**proj_config)
        return CAbstractorProjector(projector_config, num_input_tokens)

    # if projector_type == 'simple_resblock':
    #     return SimpleResBlock(config.mm_hidden_size, config.hidden_size)

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

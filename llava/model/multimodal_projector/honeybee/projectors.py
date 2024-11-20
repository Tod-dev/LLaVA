"""Custom Honeybee projectors based on Conv and MLP, including C-Abstractor.
"""
from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
#cabs
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
# from transformers.modeling_outputs import BaseModelOutput
from .configuration_honeybee import HoneybeeVisualProjectorConfig
from llava.utils import rank0_print #, process_video_with_pyav, process_video_with_decord

# def build_pos_embeds(
#     config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
# ):
#     rank0_print("build_pos_embeds",config, num_input_tokens, vision_hidden_size)
#     # pos emb
#     if config.pos_emb:
#         pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size)) #seq_len, hidden_size
#         nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
#     else:
#         pos_emb = None
#     #rank0_print("pos_emb shape:", pos_emb.shape)
#     return pos_emb

def build_pos_embeds(
    config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    rank0_print("Initializing positional embeddings...")
    
    # Only initialize pos_emb if the config requires it
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.empty(1, num_input_tokens, vision_hidden_size))
        # Use a truncated normal distribution for initialization
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
        rank0_print("pos_emb initialized with shape:", pos_emb.shape)
    else:
        pos_emb = None
        rank0_print("No positional embedding initialized (config.pos_emb is False).")
    
    return pos_emb

def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if getattr(config, "prenorm", False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int,
    ):
        super().__init__()
        rank0_print("init")
        self.config = config
        self.num_input_tokens = num_input_tokens

        # think tokens
        self.eos_tokens = build_eos_tokens(config, config.output_hidden_size)

        # pos emb
        #self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)
        # Check initialization in the model's __init__ or in _load_from_state_dict
        #rank0_print("Initial pos_emb values:", self.pos_emb)
        self.prenorm = build_prenorm(config)

        self.build_net()
        # rank0_print(f"init  pos_emb : {self.pos_emb}")

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        #rank0_print("projector forward of x ", x.shape)
        #projector forward of x  torch.Size([8, 576, 1024])
        if self.prenorm is not None:
            x = self.prenorm(x)

        # rank0_print(" forward x shape:", x.shape)
        # rank0_print(" forward pos_emb:", self.pos_emb.shape)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        output = x #BaseModelOutput(last_hidden_state=x)
        return output
    
    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        if "pos_emb" in state_dict:
            rank0_print("`pos_emb` is currently uninitialized, initializing it now.")
            self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)
        super()._load_from_state_dict(state_dict, *args, **kwargs)

#    def _load_from_state_dict(self, state_dict, *args, **kwargs):
#            pos_emb_keys = ["model.mm_projector.pos_emb", "pos_emb"]
#
#           # Identify the key for pos_emb if present in state_dict
#            pos_emb_key = next((k for k in pos_emb_keys if k in state_dict), None)
#
#            if pos_emb_key and self.config.pos_emb:
#                rank0_print(f"Loading `pos_emb` from checkpoint with key: {pos_emb_key}")
#                
#                if self.pos_emb.numel() == 0:
#                    rank0_print("`pos_emb` is currently uninitialized, initializing it now.")
#                    self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)
#
#                rank0_print("Checkpoint pos_emb shape:",  state_dict[pos_emb_key].shape)
#                rank0_print("Model pos_emb shape:", self.pos_emb.shape)
#
#                # Load the modified state dict and check shapes
#               super()._load_from_state_dict(state_dict, *args, **kwargs)
#            else:
#                if pos_emb_key:
#                    rank0_print(f"Removing {pos_emb_key} from state_dict due to mismatch or uninitialized `pos_emb` in config.")
#                    state_dict.pop(pos_emb_key, None)
#                super()._load_from_state_dict(state_dict, *args, **kwargs)

    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #         pos_emb_keys = ["model.mm_projector.pos_emb", "pos_emb"]

    #         # Identify the key for pos_emb if present in state_dict
    #         pos_emb_key = next((k for k in pos_emb_keys if k in state_dict), None)

    #         if pos_emb_key and self.config.pos_emb:
    #             rank0_print(f"Loading `pos_emb` from checkpoint with key: {pos_emb_key}")
                
    #             # Load pos_emb from checkpoint and compare shapes
    #             pos_emb_checkpoint = state_dict[pos_emb_key]
                
    #             if self.pos_emb.numel() == 0:
    #                 rank0_print("`pos_emb` is currently uninitialized, initializing it now.")
    #                 self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)

    #             rank0_print("Checkpoint pos_emb shape:", pos_emb_checkpoint.shape)
    #             rank0_print("Model pos_emb shape:", self.pos_emb.shape)

    #             if pos_emb_checkpoint.size(1) == self.pos_emb.size(1) + 1:
    #                 rank0_print("Removing extra position embedding token from checkpoint to match model.")
    #                 pos_emb_checkpoint = pos_emb_checkpoint[:, 1:]

    #             # Load the modified state dict and check shapes
    #             state_dict[pos_emb_key] = pos_emb_checkpoint
    #             rank0_print("Adjusted `pos_emb` shape:", state_dict[pos_emb_key].shape)
    #             super()._load_from_state_dict(state_dict, *args, **kwargs)
    #         else:
    #             if pos_emb_key:
    #                 rank0_print(f"Removing {pos_emb_key} from state_dict due to mismatch or uninitialized `pos_emb` in config.")
    #                 state_dict.pop(pos_emb_key, None)
    #             super()._load_from_state_dict(state_dict, *args, **kwargs)

    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     key = None
    #     keys = ["model.mm_projector.pos_emb","pos_emb"]
    #     #pos_emb -> caricamento pesi pretrain
    #     # model.mm_projector.pos_emb -> ckp
    #     for k in keys:
    #         if k in state_dict.keys():
    #             key = k
    #             break
    #     rank0_print("key", key)
    #     if self.config.pos_emb and key:            
    #         rank0_print("Projector _load_from_state_dict", state_dict.keys())
    #         rank0_print("load_state_dict pos_emb pre: ",self.pos_emb)
    #         # update old ckpt compatible with current code
    #         pos_emb = state_dict[key]
    #         rank0_print("self.pos_emb numel", self.pos_emb.numel())
    #         if self.pos_emb.numel() == 0:
    #             self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)
    #             rank0_print("load_state_dict pos_emb post: ",self.pos_emb)
    #         rank0_print("pos_emb sizes",  pos_emb.size(), self.pos_emb.size())
    #         if pos_emb.size(1) == self.pos_emb.size(1) + 1:
    #             # remove obsolete first pos emb (for cls token originally)
    #             state_dict[key] = pos_emb[:, 1:]
    #         super()._load_from_state_dict(state_dict, *args, **kwargs)
    #     else:
    #         state_dict.pop(key)
    #         super()._load_from_state_dict(state_dict, *args, **kwargs)
    
    # def _load_from_state_dict(self, state_dict, *args, **kwargs):
    #     rank0_print("load from state dict")
    #     rank0_print(vars(self))
    #     rank0_print(self.pos_emb)
    #     # update old ckpt compatible with current code
    #     rank0_print("Projector _load_from_state_dict", state_dict.keys())
    #     key = "model.mm_projector.cabstractor.pos_emb" #"abstractor.pos_emb"
    #     #rank0_print("Projector _load_from_state_dict", state_dict.keys())
    #     #key = "model.mm_projector.cabstractor.pos_emb" #"abstractor.pos_emb"
    #     key = "cabstractor.pos_emb" #"abstractor.pos_emb"
    #     pos_emb = state_dict[key]
    #     rank0_print("pos_emb shape:", pos_emb.shape, self.pos_emb.shape)
    #     self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)
    #     rank0_print(f"Initialized pos_emb shape: {self.pos_emb.shape}")
    #     if pos_emb.size(1) == self.pos_emb.size(1) + 1:
    #         # remove obsolete first pos emb (for cls token originally)
    #         state_dict[key] = pos_emb[:, 1:]
    #     super()._load_from_state_dict(state_dict, *args, **kwargs)


class MLPProjector(Projector):
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth

        self.net = build_mlp(depth, encoder_hidden_size, output_hidden_size)

    def _forward(self, x):
        return self.net(x)


class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x


class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_query_tokens
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)
        self.pos_emb = build_pos_embeds(self.config, self.num_input_tokens, self.config.encoder_hidden_size)
        rank0_print(f"init  pos_emb : {self.pos_emb.shape}")

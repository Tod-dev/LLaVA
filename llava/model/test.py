import torch
# from builder import SimpleResBlock
# x = torch.randn(16, 576, 1024)  # Ensure this is the correct shape
# block = SimpleResBlock(1024, 5012)
# output = block.forward(x)
# print("Final output shape:", output.shape)  # This should match your expectations


from multimodal_projector.projectors import CAbstractor

from multimodal_encoder.builder import build_vision_tower
from multimodal_projector.configuration_honeybee import HoneybeeConfig, HoneybeeVisionConfig,HoneybeeVisualProjectorConfig



class VisionTowerConfig:
    def __init__(self, vision_tower=None, delay_load=None, mm_vision_select_layer=None):
        self.vision_tower = vision_tower
        self.delay_load = delay_load
        self.mm_vision_select_layer = mm_vision_select_layer

# Define the dictionary
# mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
vision_tower_cfg = {'vision_tower': 'openai/clip-vit-base-patch16', 'delay_load': True,
'mm_vision_select_layer': -1 }

# Convert the dictionary into a class object
vision_tower_config_obj = VisionTowerConfig(**vision_tower_cfg)

# Access the attributes of the class object
print("Vision Tower:", vision_tower_config_obj.vision_tower)
print("Delay Load:", vision_tower_config_obj.delay_load)

# Build the vision model using the configuration
vision_model = build_vision_tower(vision_tower_config_obj)

# Get the number of input tokens from the vision model
num_input_tokens = vision_model.get_num_tokens()


# #HoneyBeeConfig 
# honeybee_config = HoneybeeConfig() vision_config: dict,
#         projector_config: dict,
#         lm_config: dict,)


# # Vision config
# vision_config = HoneybeeVisionConfig.from_exp_config(vision_config)

# # LM config (from exp config)
# self.lm_config = HoneybeeLanguageConfig(**lm_config)
# lm_local_files_only, lm_file_name = check_local_file(
#     self.lm_config.pretrained_lm_name_or_path
# )
# self.text_config = AutoConfig.from_pretrained(
#     lm_file_name,
#     local_files_only=lm_local_files_only,
# )


# projector_config = HoneybeeVisualProjectorConfig.from_exp_config(
#     proj_config,
#     vision_hidden_size=vision_config.hidden_size, #diventa encoder_hidden_size
#     lm_hidden_size=text_config.hidden_size, #diventa output_hidden_size
# )

# Define projector configuration
proj_config = {
    "projector_type": "c-abs",
    "depth": 3,
    "mlp_depth": 2,
    "hidden_size": 1024,
    "num_eos_tokens": 0,
    "pos_emb": True,
    "feature_layer_index": -1,
    "prenorm": False,
    "num_query_tokens": 144,
    "encoder_hidden_size": 1024, #vision_hidden_size, #HoneybeeVisionConfig.from_exp_config(vision_config).hidden_size
    "output_hidden_size": 5120 #lm_hidden_size, #self.text_config.hidden_size
}

projector_config = HoneybeeVisualProjectorConfig(**proj_config)

# Initialize the CAbstractor with the projector configuration and number of input tokens
c_abstractor = CAbstractor(projector_config, num_input_tokens=num_input_tokens)

# Example input tensor
dim1 = 196 # should be 576 in llava
x = torch.randn(16, dim1, 1024)  # Ensure this is the correct shape based on your model requirements

# Forward pass through the CAbstractor
output = c_abstractor.forward(x)

# Print the final output shape
print("Final output shape:", output)  # This should match your expectations

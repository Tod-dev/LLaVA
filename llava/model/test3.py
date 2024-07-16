import torch
from multimodal_projector.projectors import CAbstractor
from multimodal_encoder.builder import build_vision_tower
from multimodal_projector.configuration_honeybee import HoneybeeConfig, HoneybeeVisionConfig, HoneybeeVisualProjectorConfig

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
}

class VisionTowerConfig:
    def __init__(self, vision_tower=None, delay_load=None, mm_vision_select_layer=None):
        self.vision_tower = vision_tower
        self.delay_load = delay_load
        self.mm_vision_select_layer = mm_vision_select_layer

# Define the vision tower configuration
vision_tower_cfg = {
    'vision_tower': 'openai/clip-vit-base-patch16',
    'delay_load': True,
    'mm_vision_select_layer': -1
}

# Instantiate HoneybeeVisionConfig and HoneybeeVisualProjectorConfig
vision_config = HoneybeeVisionConfig(
    encoder_type="openai.clip",
    pretrained_vision_name_or_path='openai/clip-vit-base-patch16',
)

# Convert the dictionary into a class object
vision_tower_config_obj = VisionTowerConfig(**vision_tower_cfg)

# Access the attributes of the class object
print("Vision Tower:", vision_tower_config_obj.vision_tower)
print("Delay Load:", vision_tower_config_obj.delay_load)

# Build the vision model using the configuration
vision_model = build_vision_tower(vision_tower_config_obj)

# Get the number of input tokens from the vision model
num_input_tokens = vision_model.get_num_tokens()

# Define the language model configuration
lm_config = {
    # "pretrained_lm_name_or_path": "lmsys/vicuna-7b-v1.5",
    # "pretrained_tokenizer_name_or_path": "./hf_home/hub/tokenizers--llama2"
}



# Normally, the text_config would be created from a pre-trained model's configuration
# Here, it's simplified for demonstration purposes.
text_config = {"hidden_size": 4096}

projector_config = HoneybeeVisualProjectorConfig.from_exp_config(
    proj_config,
    vision_hidden_size=vision_config.hidden_size,  # Should match vision config hidden size
    lm_hidden_size=text_config['hidden_size'],     # Should match text config hidden size
)

# Initialize the HoneybeeConfig
honeybee_config = HoneybeeConfig(
    vision_config=vision_config,
    projector_config=projector_config,
    lm_config=lm_config
)

# Initialize the CAbstractor with the projector configuration and number of input tokens
c_abstractor = CAbstractor(projector_config, num_input_tokens=num_input_tokens)

# Example input tensor
x = torch.randn(16, 576, 1024)  # Ensure this is the correct shape based on your model requirements

# Forward pass through the CAbstractor
output = c_abstractor.forward(x)

# Print the final output shape
print("Final output shape:", output.shape)  # This should match your expectations

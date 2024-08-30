import torch
import torch.nn as nn
import re

from honeybee.projectors import CAbstractor
from honeybee.configuration_honeybee import HoneybeeVisualProjectorConfig
if __name__ == "__main__":
    # Set up the configuration and parameters based on your provided instantiation
    num_input_tokens = 576  # Replace with the appropriate number of input tokens

    # Define the projector configuration
    proj_config = {
        "projector_type": "c-abs",
        "depth": 3,
        "mlp_depth": 2,
        "hidden_size": 1024,  # Vision hidden size
        "num_eos_tokens": 0,
        "pos_emb": True,
        "feature_layer_index": -1,
        "prenorm": False,
        "num_query_tokens": 144,
        "encoder_hidden_size": 1024,  # mm_hidden_size from config
        "output_hidden_size": 5120  # hidden_size from config
    }

    # Instantiate the HoneybeeVisualProjectorConfig and CAbstractor model
    projector_config = HoneybeeVisualProjectorConfig(**proj_config)
    projector_model = CAbstractor(projector_config, num_input_tokens)

    # Path to the saved weights
    weights_path = "/leonardo_scratch/fast/IscrB_F4VL/mtodaro/checkpoints/llava-v1.5-vicuna-13b-v1.5-pretrain-2/mm_projector.bin"

    # Load the saved weights
    saved_weights = torch.load(weights_path)
    
    # Remove the "model.mm_projector." prefix from all keys
    new_state_dict = {}
    for k, v in saved_weights.items():
        new_key = k.replace("model.mm_projector.", "")
        new_state_dict[new_key] = v

    # Load the weights into the projector model
    projector_model.load_state_dict(new_state_dict, strict=True)

    # Verify the specific parameters, e.g., pos_emb
    print("pos_emb", projector_model.pos_emb)  # shape Should output torch.Size([1, 576, 1024])
    # Access the first convolutional layer within the first RegStage module
    first_conv_layer = projector_model.net[0].b1.conv1.conv
    print("First conv layer weight shape:", first_conv_layer.weight.shape)  # This should give you the shape of the conv layer's weights

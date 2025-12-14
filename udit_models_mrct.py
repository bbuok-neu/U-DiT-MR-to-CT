# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
U-DiT model modified for MR-to-CT synthesis.
- 8 channel input (4 noisy CT + 4 MR condition)
- 8 channel output
- Support for loading pretrained weights with MR channels initialized to zeros
"""

import torch
import torch.nn as nn

from udit_models import (
    U_DiT,
    OverlapPatchEmbed,
    TimestepEmbedder,
    LabelEmbedder,
    U_DiTBlock,
    Downsample,
    Upsample,
    FinalLayer,
    trunc_normal_,
)


class U_DiT_MRCT(U_DiT):
    """
    U-DiT model for MR-to-CT synthesis.
    Takes concatenated [noisy_ct, mr] as input (8 channels) and outputs CT prediction (8 channels).
    """
    def __init__(
        self,
        input_size=32,
        down_factor=2,
        in_channels=4,  # Original in_channels (CT latent channels)
        hidden_size=1152,
        depth=[2, 5, 8, 5, 2],
        num_heads=16,
        mlp_ratio=4,
        class_dropout_prob=0.0,  # No class dropout for image-conditioned training
        num_classes=1,  # Dummy class (not used in image-conditioned training)
        learn_sigma=True,
        rep=1,
        ffn_type='rep',
        **kwargs
    ):
        # Store original CT latent channels for output
        self.original_in_channels = in_channels
        
        # Initialize with 8 input channels (4 CT + 4 MR)
        super().__init__(
            input_size=input_size,
            down_factor=down_factor,
            in_channels=in_channels * 2,  # 8 channels: 4 noisy CT + 4 MR
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=learn_sigma,
            rep=rep,
            ffn_type=ffn_type,
            **kwargs
        )
        
        # Override out_channels: in_channels*2 for noise+variance (learn_sigma=True), 
        # or in_channels for noise only (learn_sigma=False).
        # For CT latent with 4 channels: 8 output when learn_sigma=True, 4 otherwise.
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        
        # Recreate final layer with correct output channels
        self.final_layer = FinalLayer(hidden_size * 2, self.out_channels)
        # Initialize final layer weights (same as parent)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def forward(self, x, t, y=None):
        """
        Forward pass of U-DiT-MRCT.
        x: (N, 8, H, W) tensor of spatial inputs (concatenated noisy_ct and mr latents)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (optional, not used in image-conditioned training)
        """
        # For image-conditioned training, y can be None or dummy
        if y is None:
            y = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return super().forward(x, t, y)

    def forward_with_mr(self, noisy_ct, mr, t, y=None):
        """
        Convenience method to forward with separate noisy CT and MR inputs.
        noisy_ct: (N, 4, H, W) tensor of noisy CT latents
        mr: (N, 4, H, W) tensor of MR latents
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (optional)
        """
        x = torch.cat([noisy_ct, mr], dim=1)  # (N, 8, H, W)
        return self.forward(x, t, y)


def load_pretrained_weights(model, pretrained_path, device='cpu'):
    """
    Load pretrained U-DiT weights into U-DiT-MRCT model.
    For the input embedding layer, the first 4 channels (noisy CT) are loaded from pretrained,
    and the last 4 channels (MR condition) are initialized to zeros.
    
    Args:
        model: U_DiT_MRCT model instance
        pretrained_path: Path to pretrained U-DiT checkpoint
        device: Device to load weights to
    
    Returns:
        model: Model with loaded weights
    """
    # Load pretrained checkpoint
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Get state dict (handle both 'model' key and direct state dict)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'ema' in checkpoint:
        pretrained_state_dict = checkpoint['ema']
    else:
        pretrained_state_dict = checkpoint
    
    # Get current model state dict
    model_state_dict = model.state_dict()
    
    # Load weights
    loaded_keys = []
    skipped_keys = []
    
    for key in model_state_dict.keys():
        if key in pretrained_state_dict:
            pretrained_weight = pretrained_state_dict[key]
            model_weight = model_state_dict[key]
            
            # Handle x_embedder.proj.weight specially (input embedding layer)
            if key == 'x_embedder.proj.weight':
                # pretrained_weight shape: (hidden_size, 4, 3, 3)
                # model_weight shape: (hidden_size, 8, 3, 3)
                if pretrained_weight.shape != model_weight.shape:
                    # Initialize model weight to zeros
                    new_weight = torch.zeros_like(model_weight)
                    # Copy pretrained weights to first 4 channels (noisy CT)
                    in_channels_pretrained = pretrained_weight.shape[1]
                    new_weight[:, :in_channels_pretrained, :, :] = pretrained_weight
                    # Last 4 channels (MR) remain zeros
                    model_state_dict[key] = new_weight
                    loaded_keys.append(f"{key} (partial: first {in_channels_pretrained} channels)")
                else:
                    model_state_dict[key] = pretrained_weight
                    loaded_keys.append(key)
            elif pretrained_weight.shape == model_weight.shape:
                model_state_dict[key] = pretrained_weight
                loaded_keys.append(key)
            else:
                skipped_keys.append(f"{key} (shape mismatch: {pretrained_weight.shape} vs {model_weight.shape})")
        else:
            skipped_keys.append(f"{key} (not in pretrained)")
    
    # Load the modified state dict
    model.load_state_dict(model_state_dict)
    
    print(f"Loaded {len(loaded_keys)} keys from pretrained checkpoint")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} keys:")
        for key in skipped_keys[:10]:  # Print first 10 skipped keys
            print(f"  - {key}")
        if len(skipped_keys) > 10:
            print(f"  ... and {len(skipped_keys) - 10} more")
    
    return model


#################################################################################
#                           U-DiT-MRCT Configs                                  #
#################################################################################
def U_DiT_MRCT_custom(**kwargs):
    return U_DiT_MRCT(**kwargs)


def U_DiT_MRCT_S(**kwargs):
    return U_DiT_MRCT(
        down_factor=2,
        hidden_size=96,
        num_heads=4,
        depth=[2, 5, 8, 5, 2],
        ffn_type='rep',
        rep=1,
        mlp_ratio=2,
        attn_type='v2',
        posemb_type='rope2d',
        downsampler='dwconv5',
        down_shortcut=1,
        **kwargs
    )


def U_DiT_MRCT_B(**kwargs):
    return U_DiT_MRCT(
        down_factor=2,
        hidden_size=192,
        num_heads=8,
        depth=[2, 5, 8, 5, 2],
        ffn_type='rep',
        rep=1,
        mlp_ratio=2,
        attn_type='v2',
        posemb_type='rope2d',
        downsampler='dwconv5',
        down_shortcut=1,
        **kwargs
    )


def U_DiT_MRCT_L(**kwargs):
    return U_DiT_MRCT(
        down_factor=2,
        hidden_size=384,
        num_heads=16,
        depth=[2, 5, 8, 5, 2],
        ffn_type='rep',
        rep=1,
        mlp_ratio=2,
        attn_type='v2',
        posemb_type='rope2d',
        downsampler='dwconv5',
        down_shortcut=1,
        **kwargs
    )


DiT_MRCT_models = {
    'U-DiT-MRCT-custom': U_DiT_MRCT_custom,
    'U-DiT-MRCT-S': U_DiT_MRCT_S,
    'U-DiT-MRCT-B': U_DiT_MRCT_B,
    'U-DiT-MRCT-L': U_DiT_MRCT_L,
}


if __name__ == "__main__":
    # Test model creation
    model = U_DiT_MRCT_S(input_size=32)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    noisy_ct = torch.randn(batch_size, 4, 32, 32)
    mr = torch.randn(batch_size, 4, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    # Forward with concatenated input
    x = torch.cat([noisy_ct, mr], dim=1)
    output = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Forward with separate inputs
    output2 = model.forward_with_mr(noisy_ct, mr, t)
    print(f"Output shape (forward_with_mr): {output2.shape}")

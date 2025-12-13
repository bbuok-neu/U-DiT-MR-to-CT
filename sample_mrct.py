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
Sampling/inference script for MR-to-CT synthesis using U-DiT.
Generates CT images from MR images using a trained model.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import os

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from udit_models_mrct import DiT_MRCT_models
from dataset_mrct import MRCTDataset, denormalize, tensor_to_image


def encode_images(vae, images, scale_factor=0.18215):
    """
    Encode images to latent space using VAE.
    
    Args:
        vae: VAE model
        images: (N, 1, H, W) grayscale images in [-1, 1]
        scale_factor: Latent scaling factor
    
    Returns:
        Latent tensors (N, 4, H/8, W/8)
    """
    # Repeat grayscale to 3 channels for VAE
    images_3ch = images.repeat(1, 3, 1, 1)  # (N, 3, H, W)
    latents = vae.encode(images_3ch).latent_dist.sample()
    latents = latents * scale_factor
    return latents


def decode_latents(vae, latents, scale_factor=0.18215):
    """
    Decode latents to images using VAE.
    
    Args:
        vae: VAE model
        latents: (N, 4, H/8, W/8) latent tensors
        scale_factor: Latent scaling factor
    
    Returns:
        Images (N, 3, H, W) in [-1, 1]
    """
    latents = latents / scale_factor
    images = vae.decode(latents).sample
    return images


def save_comparison_image(mr_img, gt_ct_img, pred_ct_img, save_path):
    """
    Save MR, GT CT, and predicted CT side by side.
    
    Args:
        mr_img: MR PIL Image
        gt_ct_img: Ground truth CT PIL Image  
        pred_ct_img: Predicted CT PIL Image
        save_path: Path to save combined image
    """
    w, h = mr_img.size
    combined = Image.new('L', (w * 3, h))
    combined.paste(mr_img, (0, 0))
    combined.paste(gt_ct_img, (w, 0))
    combined.paste(pred_ct_img, (w * 2, 0))
    combined.save(save_path)


@torch.no_grad()
def main(args):
    """
    Run MR-to-CT synthesis on test set.
    """
    assert torch.cuda.is_available(), "Sampling requires GPU."
    device = torch.device("cuda")
    
    print(f"Loading model from {args.ckpt}")
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    
    # Determine model configuration
    latent_size = args.image_size // 8
    
    # Create model
    model = DiT_MRCT_models[args.model](
        input_size=latent_size,
        in_channels=4,
        learn_sigma=True,
    ).to(device)
    
    # Load weights (prefer EMA weights)
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("Loaded EMA weights")
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
        print("Loaded model weights")
    else:
        model.load_state_dict(ckpt)
        print("Loaded state dict directly")
    
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup diffusion
    diffusion = create_diffusion(str(args.num_sampling_steps))
    print(f"Using {args.num_sampling_steps} sampling steps")
    
    # Setup VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    
    # Setup data
    test_dataset = MRCTDataset(args.data_path, split='test', image_size=args.image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Test dataset contains {len(test_dataset)} images")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, 'predictions')
    comparison_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Run inference
    total_mse = 0.0
    num_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating samples")):
        mr = batch['mr'].to(device)  # (N, 1, H, W)
        ct = batch['ct'].to(device)  # (N, 1, H, W)
        mr_paths = batch['mr_path']
        
        batch_size = mr.shape[0]
        
        # Encode to latent space
        mr_latent = encode_images(vae, mr)  # (N, 4, H/8, W/8)
        ct_latent = encode_images(vae, ct)  # (N, 4, H/8, W/8)
        
        latent_h, latent_w = mr_latent.shape[2], mr_latent.shape[3]
        
        # Sample CT from noise conditioned on MR
        noise = torch.randn(batch_size, 4, latent_h, latent_w, device=device)
        
        # Create model forward function that includes MR condition
        def model_fn(x, t, **kwargs):
            x_input = torch.cat([x, mr_latent], dim=1)  # (N, 8, H/8, W/8)
            return model(x_input, t)
        
        # Sample using DDPM or DDIM (both methods available in gaussian_diffusion.py)
        if args.use_ddim:
            sampled_latent = diffusion.ddim_sample_loop(
                model_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs={},
                progress=False,
                device=device,
                eta=args.ddim_eta
            )
        else:
            sampled_latent = diffusion.p_sample_loop(
                model_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs={},
                progress=False,
                device=device
            )
        
        # Compute MSE in latent space
        mse = torch.mean((sampled_latent - ct_latent) ** 2)
        total_mse += mse.item() * batch_size
        num_samples += batch_size
        
        # Decode to image space
        sampled_images = decode_latents(vae, sampled_latent)  # (N, 3, H, W)
        gt_images = decode_latents(vae, ct_latent)  # (N, 3, H, W)
        
        # Save images
        for i in range(batch_size):
            sample_idx = batch_idx * args.batch_size + i
            
            # Convert to grayscale by averaging channels
            pred_gray = sampled_images[i].mean(dim=0)  # (H, W)
            gt_gray = gt_images[i].mean(dim=0)  # (H, W)
            mr_gray = mr[i].squeeze(0)  # (H, W)
            
            # Denormalize and convert to PIL
            pred_img = tensor_to_image(pred_gray, denorm=True)
            gt_img = tensor_to_image(gt_gray, denorm=True)
            mr_img = tensor_to_image(mr_gray, denorm=True)
            
            # Save prediction only
            pred_path = os.path.join(pred_dir, f"{sample_idx:06d}.png")
            pred_img.save(pred_path)
            
            # Save comparison (MR | GT | Pred)
            comparison_path = os.path.join(comparison_dir, f"{sample_idx:06d}.png")
            save_comparison_image(mr_img, gt_img, pred_img, comparison_path)
    
    # Report final metrics
    avg_mse = total_mse / num_samples
    print(f"\nResults saved to {args.output_dir}")
    print(f"Total samples: {num_samples}")
    print(f"Average MSE (latent space): {avg_mse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="samples_mrct", help="Output directory for samples")
    parser.add_argument("--image-size", type=int, default=256)
    
    # Model arguments
    parser.add_argument("--model", type=str, default="U-DiT-MRCT-S",
                        choices=["U-DiT-MRCT-S", "U-DiT-MRCT-B", "U-DiT-MRCT-L", "U-DiT-MRCT-custom"])
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    
    # Sampling arguments
    parser.add_argument("--num-sampling-steps", type=int, default=250, help="Number of diffusion sampling steps")
    parser.add_argument("--use-ddim", action="store_true", help="Use DDIM sampling instead of DDPM")
    parser.add_argument("--ddim-eta", type=float, default=0.0, help="DDIM eta parameter")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)

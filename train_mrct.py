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
Training script for MR-to-CT synthesis using U-DiT.
Supports:
- Paired MR-CT dataset training
- Resume training from checkpoint
- Online validation on test set
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from accelerate import Accelerator
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from udit_models_mrct import DiT_MRCT_models, load_pretrained_weights
from dataset_mrct import MRCTDataset, denormalize, tensor_to_image


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


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


@torch.no_grad()
def validate(model, vae, diffusion, val_loader, device, num_samples=4, num_sampling_steps=50, save_dir=None, step=0):
    """
    Run validation on test set.
    
    Args:
        model: Trained model (EMA model recommended)
        vae: VAE for encoding/decoding
        diffusion: Diffusion model
        val_loader: Validation data loader
        device: Device
        num_samples: Number of samples to generate
        num_sampling_steps: Number of diffusion sampling steps
        save_dir: Directory to save validation images
        step: Current training step
    
    Returns:
        Average MSE loss on validation samples
    """
    model.eval()
    
    val_diffusion = create_diffusion(str(num_sampling_steps))
    
    total_mse = 0.0
    num_validated = 0
    
    # Get samples from validation set
    val_iter = iter(val_loader)
    
    for i in range(min(num_samples, len(val_loader))):
        try:
            batch = next(val_iter)
        except StopIteration:
            break
        
        mr = batch['mr'].to(device)  # (N, 1, H, W)
        ct = batch['ct'].to(device)  # (N, 1, H, W)
        
        # Encode to latent space
        with torch.no_grad():
            mr_latent = encode_images(vae, mr)  # (N, 4, H/8, W/8)
            ct_latent = encode_images(vae, ct)  # (N, 4, H/8, W/8)
        
        batch_size = mr.shape[0]
        latent_size = mr_latent.shape[-1]
        
        # Sample CT from noise conditioned on MR
        noise = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        
        # Create model forward function that includes MR condition
        def model_fn(x, t, **kwargs):
            # x is the noisy CT latent, concatenate with MR latent
            x_input = torch.cat([x, mr_latent], dim=1)  # (N, 8, H/8, W/8)
            return model(x_input, t)
        
        # Sample
        sampled_latent = val_diffusion.p_sample_loop(
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
        total_mse += mse.item()
        num_validated += 1
        
        # Save sample images
        if save_dir is not None and i == 0:
            # Decode latents to images
            with torch.no_grad():
                sampled_images = decode_latents(vae, sampled_latent)
                gt_images = decode_latents(vae, ct_latent)
            
            # Save first sample from batch
            for j in range(min(batch_size, 4)):
                # Convert to grayscale by averaging channels
                pred_gray = sampled_images[j].mean(dim=0, keepdim=True)  # (1, H, W)
                gt_gray = gt_images[j].mean(dim=0, keepdim=True)  # (1, H, W)
                mr_gray = mr[j]  # (1, H, W)
                
                # Save images
                save_path = os.path.join(save_dir, f"val_step{step:07d}_sample{j}.png")
                
                # Denormalize and save
                pred_img = tensor_to_image(pred_gray.squeeze(0), denorm=True)
                gt_img = tensor_to_image(gt_gray.squeeze(0), denorm=True)
                mr_img = tensor_to_image(mr_gray.squeeze(0), denorm=True)
                
                # Create side-by-side image: MR | GT CT | Pred CT
                w, h = pred_img.size  # PIL returns (width, height)
                combined = Image.new('L', (w * 3, h))
                combined.paste(mr_img, (0, 0))
                combined.paste(gt_img, (w, 0))
                combined.paste(pred_img, (w * 2, 0))
                combined.save(save_path)
    
    model.train()
    
    avg_mse = total_mse / max(num_validated, 1)
    return avg_mse


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains U-DiT for MR-to-CT synthesis.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f'----> Current Device: {device}')
    
    # Setup experiment folder
    if accelerator.is_main_process:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-MRCT"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        sample_dir = f"{experiment_dir}/samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        experiment_dir = None
        checkpoint_dir = None
        sample_dir = None
    
    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    
    model = DiT_MRCT_models[args.model](
        input_size=latent_size,
        in_channels=4,  # Will be doubled internally for concat input
        learn_sigma=True,
    )
    
    # Load pretrained weights if provided
    if args.pretrained_ckpt is not None:
        if accelerator.is_main_process:
            print(f"Loading pretrained weights from {args.pretrained_ckpt}")
        model = load_pretrained_weights(model, args.pretrained_ckpt, device='cpu')
    
    # Load checkpoint for resume training
    if args.resume_ckpt is not None:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model'])
    
    model = model.to(device)
    
    if accelerator.is_main_process:
        print(model)
        logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create EMA model
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    # Load EMA from checkpoint if resuming
    if args.resume_ckpt is not None:
        ema.load_state_dict(ckpt['ema'])
    
    # Setup diffusion
    diffusion = create_diffusion(timestep_respacing="")
    
    # Setup VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    requires_grad(vae, False)
    
    # Setup optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    
    # Load optimizer state if resuming
    if args.resume_ckpt is not None:
        opt.load_state_dict(ckpt['opt'])
        del ckpt
        torch.cuda.empty_cache()
    
    # Setup data
    train_dataset = MRCTDataset(args.data_path, split='train', image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // accelerator.num_processes,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Setup validation data
    if args.validate:
        val_dataset = MRCTDataset(args.data_path, split='test', image_size=args.image_size)
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(4, len(val_dataset)),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        if accelerator.is_main_process:
            logger.info(f"Validation dataset contains {len(val_dataset):,} images")
    
    if accelerator.is_main_process:
        logger.info(f"Training dataset contains {len(train_dataset):,} images")
    
    # Prepare for training
    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()
    model, opt, train_loader = accelerator.prepare(model, opt, train_loader)
    
    # Training variables
    train_steps = args.resume_step if args.resume_step is not None else 0
    start_epoch = args.start_epoch
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs starting from epoch {start_epoch}...")
    
    for epoch in range(start_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        
        for batch in train_loader:
            mr = batch['mr'].to(device)  # (N, 1, H, W)
            ct = batch['ct'].to(device)  # (N, 1, H, W)
            
            # Encode to latent space
            with torch.no_grad():
                mr_latent = encode_images(vae, mr)  # (N, 4, H/8, W/8)
                ct_latent = encode_images(vae, ct)  # (N, 4, H/8, W/8)
            
            # Sample timesteps
            t = torch.randint(0, diffusion.num_timesteps, (ct_latent.shape[0],), device=device)
            
            # Add noise to CT latent
            noise = torch.randn_like(ct_latent)
            noisy_ct = diffusion.q_sample(ct_latent, t, noise=noise)
            
            # Concatenate noisy CT with MR condition
            model_input = torch.cat([noisy_ct, mr_latent], dim=1)  # (N, 8, H/8, W/8)
            
            # Forward pass
            model_output = model(model_input, t)
            
            # Compute loss (predict noise)
            # model_output channels: in_channels*2 when learn_sigma=True (noise + variance),
            # or in_channels when learn_sigma=False (noise only). With in_channels=4: 8 or 4.
            noise_pred = model_output[:, :4]
            
            # MSE loss on noise prediction
            loss = torch.mean((noise - noise_pred) ** 2)
            
            # Backward pass
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)
            
            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                
                if accelerator.is_main_process:
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Validation
            if args.validate and train_steps % args.val_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    val_mse = validate(
                        ema, vae, diffusion, val_loader, device,
                        num_samples=args.val_samples,
                        num_sampling_steps=args.val_sampling_steps,
                        save_dir=sample_dir,
                        step=train_steps
                    )
                    logger.info(f"(step={train_steps:07d}) Validation MSE: {val_mse:.6f}")
            
            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args,
                        "epoch": epoch,
                        "step": train_steps,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    model.eval()
    if accelerator.is_main_process:
        logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--data-path", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--results-dir", type=str, default="results_mrct")
    parser.add_argument("--image-size", type=int, default=256)
    
    # Model arguments
    parser.add_argument("--model", type=str, default="U-DiT-MRCT-S", 
                        choices=["U-DiT-MRCT-S", "U-DiT-MRCT-B", "U-DiT-MRCT-L", "U-DiT-MRCT-custom"])
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    
    # Checkpoint arguments
    parser.add_argument("--pretrained-ckpt", type=str, default=None, 
                        help="Path to pretrained U-DiT checkpoint for weight initialization")
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--resume-step", type=int, default=None,
                        help="Step to resume from (if not specified, read from checkpoint)")
    parser.add_argument("--start-epoch", type=int, default=0)
    
    # Validation arguments
    parser.add_argument("--validate", action="store_true", help="Enable online validation")
    parser.add_argument("--val-every", type=int, default=1000, help="Validation frequency")
    parser.add_argument("--val-samples", type=int, default=4, help="Number of validation samples")
    parser.add_argument("--val-sampling-steps", type=int, default=50, help="Diffusion sampling steps for validation")
    
    args = parser.parse_args()
    main(args)

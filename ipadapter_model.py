"""
IP-Adapter Model Interface

This module provides utilities for working with IP-Adapter models, including:
- Loading Stable Diffusion pipelines with IP-Adapter
- Extracting CLIP embeddings from images
- Generating images from CLIP embeddings
- Utility functions for image processing
"""

from typing import List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL

from device_utils import get_device, is_cuda_available, DEVICE

# Fix for torch 2.5.0 compatibility (only needed for CUDA)
if is_cuda_available():
    torch.backends.cuda.enable_cudnn_sdp(False)

from ip_adapter import IPAdapterPlus, IPAdapterPlusXL


def get_dtype_for_device(device: str = None) -> torch.dtype:
    """
    Get the appropriate dtype for the device.
    GPU (CUDA/MPS) can use float16, but CPU requires float32.
    """
    if device is None:
        device = get_device()
    if device == "cuda":
        return torch.float16
    else:
        # CPU and MPS under Rosetta need float32
        return torch.float32


def _resolve_ipadapter_paths(version: str) -> Tuple[str, str]:
    """
    Resolve local paths for IP-Adapter assets. If missing, fall back to HF repo IDs
    and download the IP-Adapter checkpoint.
    """
    base_dir = Path("./downloads")
    image_encoder_path = base_dir / "models" / "image_encoder"

    if version == "sd15":
        ip_ckpt = base_dir / "models" / "ip-adapter-plus_sd15.bin"
        ip_ckpt_repo = "h94/IP-Adapter"
        ip_ckpt_filename = "models/ip-adapter-plus_sd15.bin"
    else:
        ip_ckpt = base_dir / "sdxl_models" / "ip-adapter-plus_sdxl_vit-h.bin"
        ip_ckpt_repo = "h94/IP-Adapter"
        ip_ckpt_filename = "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"

    def _has_hf_model_files(path: Path) -> bool:
        return (
            (path / "config.json").is_file()
            and (
                (path / "model.safetensors").is_file()
                or (path / "pytorch_model.bin").is_file()
            )
        )

    # If image encoder folder isn't present or incomplete, use repo id directly
    if not image_encoder_path.exists() or not _has_hf_model_files(image_encoder_path):
        image_encoder_path_resolved = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    else:
        image_encoder_path_resolved = str(image_encoder_path)

    # Ensure IP-Adapter checkpoint exists; download if missing
    if not ip_ckpt.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to download IP-Adapter checkpoints. "
                "Install or re-run `pip install -r requirements.txt`."
            ) from e
        ip_ckpt_path = hf_hub_download(
            repo_id=ip_ckpt_repo,
            filename=ip_ckpt_filename,
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
        )
        ip_ckpt_resolved = ip_ckpt_path
    else:
        ip_ckpt_resolved = str(ip_ckpt)

    return image_encoder_path_resolved, ip_ckpt_resolved


# ===== Image Utility Functions =====

def create_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    # Get dimensions from first image (assumes all images are same size)
    width, height = images[0].size
    
    # Create empty grid canvas
    grid = Image.new('RGB', size=(cols * width, rows * height))
    
    # Paste each image into the grid
    for i, img in enumerate(images):
        x_pos = (i % cols) * width
        y_pos = (i // cols) * height
        grid.paste(img, box=(x_pos, y_pos))
    
    return grid


# ===== CLIP Embedding Extraction Functions =====

@torch.inference_mode()
def extract_clip_embeddings_from_pil(pil_image: Union[Image.Image, List[Image.Image]], 
                                    ip_model) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: CLIP embeddings of shape (batch_size, seq_len, embed_dim)
    """
    if isinstance(pil_image, Image.Image):
        pil_image = [pil_image]
    
    # Process images through CLIP processor
    processed_images = ip_model.clip_image_processor(
        images=pil_image, return_tensors="pt"
    ).pixel_values
    
    # Move to model device with appropriate dtype
    dtype = get_dtype_for_device(ip_model.device if hasattr(ip_model, 'device') else None)
    processed_images = processed_images.to(ip_model.device, dtype=dtype)
    
    # Extract embeddings from penultimate layer (better for downstream tasks)
    clip_embeddings = ip_model.image_encoder(
        processed_images, output_hidden_states=True
    ).hidden_states[-2]
    
    # Convert to float32 for better numerical stability
    return clip_embeddings.float()


@torch.inference_mode()
def extract_clip_embeddings_from_pil_batch(pil_images: List[Image.Image], 
                                          ip_model) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: Concatenated CLIP embeddings of shape (batch, seq_len, embed_dim)
    """
    embeddings_batch = []
    
    for image in pil_images:
        embeddings = extract_clip_embeddings_from_pil(image, ip_model)
        embeddings_batch.append(embeddings)
    
    return torch.cat(embeddings_batch, dim=0)


@torch.inference_mode()
def extract_clip_embeddings_from_tensor(tensor_image: torch.Tensor, 
                                       ip_model, 
                                       resize: bool = True) -> torch.Tensor:
    """
    Returns:
        torch.Tensor: CLIP embeddings of shape (batch_size, seq_len, embed_dim)
    """
    # Move tensor to model device with appropriate dtype
    dtype = get_dtype_for_device(ip_model.device if hasattr(ip_model, 'device') else None)
    tensor_image = tensor_image.to(ip_model.device, dtype=dtype)
    
    # Resize to CLIP input resolution if requested
    if resize:
        tensor_image = torch.nn.functional.interpolate(
            tensor_image, 
            size=(224, 224), 
            mode="bilinear", 
            align_corners=False
        )
    
    # Extract embeddings with positional encoding interpolation
    clip_embeddings = ip_model.image_encoder(
        tensor_image, 
        output_hidden_states=True, 
        interpolate_pos_encoding=True
    ).hidden_states[-2]
    
    # Convert to float32 for numerical stability
    return clip_embeddings.float()


# ===== IP-Adapter Helper Functions =====

@torch.inference_mode()
def _enhanced_get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    """
    Enhanced version of IP-Adapter's get_image_embeds method.
    
    This method processes either PIL images or pre-computed CLIP embeddings
    and returns both conditional and unconditional embeddings for generation.
    
    Args:
        pil_image: PIL Image(s) to process (optional)
        clip_image_embeds: Pre-computed CLIP embeddings (optional)
        
    Returns:
        Tuple of (conditional_embeds, unconditional_embeds)
    """
    dtype = self.dtype if hasattr(self, 'dtype') else get_dtype_for_device(self.device)
    
    # Process PIL images if provided
    if pil_image is not None:
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        
        # Convert PIL to tensor and extract CLIP embeddings
        processed_images = self.clip_image_processor(
            images=pil_image, return_tensors="pt"
        ).pixel_values
        processed_images = processed_images.to(self.device, dtype=dtype)
        
        clip_image_embeds = self.image_encoder(
            processed_images, output_hidden_states=True
        ).hidden_states[-2]
    
    # Ensure clip_image_embeds has the right dtype
    clip_image_embeds = clip_image_embeds.to(dtype=dtype)
    
    # Project CLIP embeddings to IP-Adapter space
    conditional_embeds = self.image_proj_model(clip_image_embeds)
    
    # Generate unconditional embeddings (for classifier-free guidance)
    zero_tensor = torch.zeros(1, 3, 224, 224).to(self.device, dtype=dtype)
    uncond_clip_embeds = self.image_encoder(
        zero_tensor, output_hidden_states=True
    ).hidden_states[-2]
    unconditional_embeds = self.image_proj_model(uncond_clip_embeds)
    
    return conditional_embeds, unconditional_embeds


# ===== Model Loading Functions =====

@torch.inference_mode()
def load_stable_diffusion_pipeline(device: str = None) -> StableDiffusionPipeline:
    if device is None:
        device = get_device()
    
    dtype = get_dtype_for_device(device)
    
    # Model paths
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"

    # Configure DDIM scheduler for high-quality sampling
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Load VAE separately for better quality
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=dtype)
    
    # Create Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,  # Disable safety checker for faster inference
        safety_checker=None,
    )
    
    return pipeline


@torch.inference_mode()
def load_ip_adapter_model(device: str = None, sd_only: bool = False) -> IPAdapterPlus:
    if device is None:
        device = get_device()
    
    dtype = get_dtype_for_device(device)
    
    # Model and checkpoint paths
    base_model_path = "SG161222/Realistic_Vision_V4.0_noVAE"
    vae_model_path = "stabilityai/sd-vae-ft-mse"
    image_encoder_path, ip_checkpoint_path = _resolve_ipadapter_paths("sd15")

    # Configure DDIM scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    
    # Load high-quality VAE
    vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=dtype)
    
    # Create base Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
    )
    
    if sd_only:
        return pipeline
    
    # Initialize IP-Adapter with 16 tokens for better image conditioning
    ip_model = IPAdapterPlus(
        pipeline, 
        image_encoder_path, 
        ip_checkpoint_path, 
        device, 
        num_tokens=16
    )

    # Enhance the model with our improved get_image_embeds method
    setattr(ip_model.__class__, "get_image_embeds", _enhanced_get_image_embeds)
    
    return ip_model


def load_ip_adapter_xl_model(device: str = None) -> IPAdapterPlusXL:
    if device is None:
        device = get_device()
    
    dtype = get_dtype_for_device(device)
    
    base_model_path = "SG161222/RealVisXL_V1.0"
    image_encoder_path, ip_ckpt = _resolve_ipadapter_paths("sdxl")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        add_watermarker=False,
    )
    ip_model = IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)

    return ip_model

def load_ipadapter(version: str = "sd15", device: str = None) -> IPAdapterPlus | IPAdapterPlusXL:
    if device is None:
        device = get_device()
    if version == "sd15":
        return load_ip_adapter_model(device)
    elif version == "sdxl":
        return load_ip_adapter_xl_model(device)
    else:
        raise ValueError(f"Invalid version: {version}")


# ===== Image Generation Functions =====

@torch.inference_mode()
def generate_images_from_clip_embeddings(ip_model : IPAdapterPlus,
                                       clip_embeddings: torch.Tensor,
                                       num_samples: int = 4, 
                                       num_inference_steps: int = 50, 
                                       seed: Optional[int] = 42) -> List[Image.Image]:
    """Generate images from CLIP embeddings using IP-Adapter.
    clip_embeddings is (batch, seq_len, embed_dim)
    """
    # Ensure embeddings have correct shape and dtype
    if clip_embeddings.ndim == 2:
        clip_embeddings = clip_embeddings.unsqueeze(0)
    
    if clip_embeddings.ndim != 3:
        raise ValueError(f"Expected 3D embeddings (batch, seq, dim), got {clip_embeddings.shape}")
    
    # Move to appropriate device and dtype (use model's dtype)
    dtype = ip_model.dtype if hasattr(ip_model, 'dtype') else get_dtype_for_device(ip_model.device)
    clip_embeddings = clip_embeddings.to(device=ip_model.device, dtype=dtype)
    
    # Generate images using IP-Adapter
    negative_prompt = "nsfw, lowres, (bad), text, error, fewer, extra, missing, worst quality, jpeg artifacts, low quality, watermark, unfinished, displeasing, oldest, early, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]"
    generated_images = ip_model.generate(
        clip_image_embeds=clip_embeddings,
        negative_prompt=negative_prompt,
        pil_image=None,
        num_samples=num_samples,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    return generated_images


# ===== Legacy Function Aliases =====

# Maintain backward compatibility with existing code
image_grid = create_image_grid
extract_clip_embedding_pil = extract_clip_embeddings_from_pil
extract_clip_embedding_pil_batch = extract_clip_embeddings_from_pil_batch
extract_clip_embedding_tensor = extract_clip_embeddings_from_tensor
load_sdxl = load_stable_diffusion_pipeline
generate = generate_images_from_clip_embeddings

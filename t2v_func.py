import os
import sys
from datetime import datetime
from typing import Optional
import asyncio

import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

def parse_size(size_str):
    """Parse size string like '1280*720' into (width, height)."""
    width, height = map(int, size_str.split('*'))
    return width, height


def get_flow_shift(width, height, sample_shift=None):
    """Determine flow shift based on resolution."""
    if sample_shift is not None:
        return sample_shift
    
    # Auto-determine based on resolution
    if height == 480 or width == 480:
        return 3.0
    else:
        return 5.0


def generate_output_filename(prompt, task):
    """Generate output filename based on prompt and timestamp."""
    # Clean prompt for filename
    clean_prompt = prompt[:50].replace(" ", "_").replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task}_{clean_prompt}_{timestamp}.mp4"


def generate_wan_video(
    prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    task="t2v-1.3B",
    model_id=None,
    size="832*480",
    num_frames=81,
    sample_guide_scale=6.0,
    sample_shift=None,
    sample_steps=50,
    fps=16,
    offload_model=True,
    t5_cpu=True,
    output_path=None
):
    """
    Generate video using Wan2.1 model with diffusers integration.
    
    Args:
        prompt (str): Text prompt for video generation
        negative_prompt (str): Negative prompt to avoid certain content
        task (str): Model variant - 't2v-1.3B' or 't2v-14B'
        model_id (str): HuggingFace model ID (overrides task)
        size (str): Video resolution in format 'width*height'
        num_frames (int): Number of frames to generate
        sample_guide_scale (float): Guidance scale for sampling
        sample_shift (float): Flow shift parameter (auto-determined if None)
        sample_steps (int): Number of sampling steps
        fps (int): Output video frame rate
        offload_model (bool): Enable CPU offloading to reduce GPU memory
        t5_cpu (bool): Keep T5 text encoder on CPU
        output_path (str): Output video file path (auto-generated if None)
    
    Returns:
        tuple: (video_frames, output_path) - The generated video frames and the path where it was saved
    """
    
    # Parse video dimensions
    width, height = parse_size(size)

    # Determine flow shift
    flow_shift = get_flow_shift(width, height, sample_shift)
    
    print(f"Loading model: {model_id}")
    print(f"Video size: {width}x{height}")
    print(f"Flow shift: {flow_shift}")
    print(f"Model offloading: {offload_model}")
    print(f"T5 CPU: {t5_cpu}")
    
    # Load VAE
    try:
        # Always use the resolved model_id (which now points to HuggingFace if local dir is not diffusers format)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    except Exception as e:
        print(f"Failed to load VAE: {e}")
        raise e
    
    # Setup scheduler
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction', 
        use_flow_sigmas=True, 
        num_train_timesteps=1000, 
        flow_shift=flow_shift
    )
    
    # Load pipeline
    try:
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        pipe.scheduler = scheduler
        
        # Apply memory optimizations
        if offload_model:
            print("Enabling model CPU offloading...")
            pipe.enable_model_cpu_offload()
        
        if t5_cpu:
            print("Moving T5 text encoder to CPU...")
            # Move text encoder to CPU to save GPU memory
            if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
                pipe.text_encoder = pipe.text_encoder.to('cpu')
        
        if not offload_model:
            pipe.to("cuda")
            
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        raise e
    
    print(f"Generating video with prompt: {prompt}")
    
    # Generate video
    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=sample_guide_scale,
            num_inference_steps=sample_steps,
        ).frames[0]
        
        # Determine output filename
        if output_path:
            final_output_path = output_path
        else:
            final_output_path = generate_output_filename(prompt, task)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(final_output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        # Export video
        export_to_video(output, final_output_path, fps=fps)
        print(f"Video saved to: {final_output_path}")
        
        return final_output_path
        
    except Exception as e:
        print(f"Failed to generate video: {e}")
        raise e


def main():
    """Example usage of the generate_wan_video function"""
    # Example with your specific parameters
    output_path = generate_wan_video(
        prompt="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
        task="Wan2.1-T2V-1.3B-Diffusers",
        size="832*480",
        offload_model=True,
        t5_cpu=True,
        sample_shift=8,
        sample_guide_scale=6.0,
        model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_frames=81,
        fps=16
    )
    print("Video generation completed successfully!")
    return output_path


if __name__ == "__main__":
    main()
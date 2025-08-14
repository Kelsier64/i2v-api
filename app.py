import os
import sys
from datetime import datetime
from typing import Optional
import asyncio
import uuid
import json
import torch
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
OUTPUT_DIR = "./videos"

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


def generate_output_filename(videos_dir, task):
    """Generate output filename based on UUID and timestamp."""
    # Generate a unique UUID for the video
    video_uuid = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create videos directory if it doesn't exist
    os.makedirs(videos_dir, exist_ok=True)
    
    filename = f"{task}_{timestamp}_{video_uuid}.mp4"
    return os.path.join(videos_dir, filename)


# Pydantic models for API
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(
        default="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
        description="Text prompt for video generation"
    )
    negative_prompt: str = Field(
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        description="Negative prompt to avoid certain content"
    )
    task: str = Field(default="t2v-1.3B", description="Model variant")
    model_id: Optional[str] = Field(default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", description="HuggingFace model ID")
    size: str = Field(default="832*480", description="Video resolution in format 'width*height'")
    num_frames: int = Field(default=81, ge=1, le=200, description="Number of frames to generate")
    sample_guide_scale: float = Field(default=6.0, ge=1.0, le=20.0, description="Guidance scale for sampling")
    sample_shift: Optional[float] = Field(default=None, description="Flow shift parameter (auto-determined if None)")
    sample_steps: int = Field(default=50, ge=10, le=100, description="Number of sampling steps")
    fps: int = Field(default=16, ge=1, le=60, description="Output video frame rate")
    offload_model: bool = Field(default=True, description="Enable CPU offloading to reduce GPU memory")
    t5_cpu: bool = Field(default=True, description="Keep T5 text encoder on CPU")


class VideoGenerationResponse(BaseModel):
    status: str
    message: str
    video_filename: Optional[str] = None
    task_id: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Wan Diffusers Video Generation API",
    description="API for generating videos using Wan2.1 diffusion models with queue management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the queue processor on startup."""
    # Start the queue processor as a background task
    asyncio.create_task(process_video_generation_queue())

# Global variable to store active generation tasks
active_tasks = {}

# Global queue for video generation tasks
video_generation_queue = asyncio.Queue(maxsize=10)  # Limit to 10 pending tasks
executor = ThreadPoolExecutor(max_workers=2)  # Limit concurrent generations

# Global pipeline instance to persist across tasks
global_pipeline = None
global_pipeline_config = None
pipeline_lock = threading.Lock()

# Task processing status
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class VideoGenerationTask:
    def __init__(self, task_id: str, request: "VideoGenerationRequest"):
        self.task_id = task_id
        self.request = request
        self.status = TaskStatus.PENDING
        self.message = "Task queued for processing"
        self.video_filename = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None


def load_pipeline_if_needed(model_id, offload_model=True, t5_cpu=True, flow_shift=5.0):
    """
    Load pipeline only if not already loaded or if configuration changed.
    
    Args:
        model_id (str): HuggingFace model ID
        offload_model (bool): Enable CPU offloading to reduce GPU memory
        t5_cpu (bool): Keep T5 text encoder on CPU
        flow_shift (float): Flow shift parameter
    
    Returns:
        WanPipeline: The loaded pipeline instance
    """
    global global_pipeline, global_pipeline_config
    
    current_config = {
        'model_id': model_id,
        'offload_model': offload_model,
        't5_cpu': t5_cpu,
        'flow_shift': flow_shift
    }
    
    with pipeline_lock:
        # Check if pipeline needs to be loaded/reloaded
        if global_pipeline is None or global_pipeline_config != current_config:
            print(f"Loading pipeline with config: {current_config}")
            
            # Unload existing pipeline if present
            if global_pipeline is not None:
                print("Unloading existing pipeline...")
                unload_pipeline()
            
            # Load VAE
            try:
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
                global_pipeline = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
                global_pipeline.scheduler = scheduler
                
                # Apply memory optimizations
                if offload_model:
                    print("Enabling model CPU offloading...")
                    global_pipeline.enable_model_cpu_offload()
                
                if t5_cpu:
                    print("Moving T5 text encoder to CPU...")
                    # Move text encoder to CPU to save GPU memory
                    if hasattr(global_pipeline, 'text_encoder') and global_pipeline.text_encoder is not None:
                        global_pipeline.text_encoder = global_pipeline.text_encoder.to('cpu')
                
                if not offload_model:
                    global_pipeline.to("cuda")
                
                global_pipeline_config = current_config
                print("Pipeline loaded successfully")
                    
            except Exception as e:
                print(f"Failed to load pipeline: {e}")
                global_pipeline = None
                global_pipeline_config = None
                raise e
        
        return global_pipeline


def unload_pipeline():
    """Unload the global pipeline to free GPU memory."""
    global global_pipeline, global_pipeline_config
    
    if global_pipeline is not None:
        print("Unloading pipeline to free GPU memory...")
        try:
            # Move pipeline components to CPU and clear CUDA cache
            if hasattr(global_pipeline, 'to'):
                global_pipeline.to('cpu')
            
            # Clear references
            global_pipeline = None
            global_pipeline_config = None
            
            # Force garbage collection and clear CUDA cache
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("Pipeline unloaded successfully")
        except Exception as e:
            print(f"Error during pipeline unload: {e}")


def is_queue_empty_and_no_processing():
    """Check if queue is empty and no tasks are currently processing."""
    queue_empty = video_generation_queue.qsize() == 0
    
    # Check if any tasks are currently processing
    processing_tasks = sum(1 for task in active_tasks.values() 
                          if task.status == TaskStatus.PROCESSING)
    
    return queue_empty and processing_tasks == 0
    prompt="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    task="t2v-1.3B",
    model_id=None,
    size="832*480",
    num_frames=81,
    sample_guide_scale=6.0,
    sample_shift=None,
    sample_steps=50,
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
        str: The output path where the video was saved
    """
    
    # Parse video dimensions
    width, height = parse_size(size)

    # Determine flow shift
    flow_shift = get_flow_shift(width, height, sample_shift)
    
    print(f"Using model: {model_id}")
    print(f"Video size: {width}x{height}")
    print(f"Flow shift: {flow_shift}")
    print(f"Model offloading: {offload_model}")
    print(f"T5 CPU: {t5_cpu}")
    
    # Load or reuse existing pipeline
    try:
        pipe = load_pipeline_if_needed(model_id, offload_model, t5_cpu, flow_shift)
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
            final_output_path = generate_output_filename(OUTPUT_DIR, task)

        # Export video
        export_to_video(output, final_output_path, fps=fps)
        print(f"Video saved to: {final_output_path}")
        
        return final_output_path
        
    except Exception as e:
        print(f"Failed to generate video: {e}")
        raise e


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Wan Diffusers Video Generation API with Queue Management & Pipeline Optimization",
        "version": "1.0.1",
        "features": [
            "Asynchronous video generation with queue management",
            "Multiple concurrent workers",
            "Real-time queue status monitoring",
            "Task progress tracking",
            "Intelligent pipeline management - pipeline persists during queue processing",
            "Automatic pipeline unloading when queue is empty to save GPU memory"
        ],
        "endpoints": {
            "generate": "/generate - POST request to generate video (queued)",
            "status": "/status/{task_id} - GET request to check generation status",
            "queue_status": "/queue/status - GET request to check queue status",
            "pipeline_status": "/pipeline/status - GET request to check pipeline status and GPU memory",
            "pipeline_unload": "/pipeline/unload - POST request to manually unload pipeline",
            "download": "/download/{filename} - GET request to download generated video",
            "list": "/list - GET request to list all generated videos",
            "health": "/health - GET request to check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_count": torch.cuda.device_count() if gpu_available else 0
    }


@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    """
    Generate a video based on the provided prompt and parameters.
    Returns immediately with a task ID for async processing using a queue.
    """
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Create task object
    task = VideoGenerationTask(task_id, request)
    
    # Store task info
    active_tasks[task_id] = task
    
    try:
        # Add task to queue (this will raise QueueFull if queue is at max capacity)
        await video_generation_queue.put(task)
        
        return VideoGenerationResponse(
            status="accepted",
            message=f"Video generation queued. Position in queue: {video_generation_queue.qsize()}",
            task_id=task_id
        )
        
    except asyncio.QueueFull:
        # Remove from active tasks if queue is full
        del active_tasks[task_id]
        raise HTTPException(
            status_code=503, 
            detail="Server busy. Too many pending requests. Please try again later."
        )



@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a video generation task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    
    response = {
        "task_id": task_id,
        "status": task.status,
        "message": task.message,
        "created_at": task.created_at.isoformat(),
        "queue_position": None
    }
    
    # Add queue position if task is still pending
    if task.status == TaskStatus.PENDING:
        # Calculate position in queue
        queue_items = list(video_generation_queue._queue)
        try:
            position = next(i for i, queued_task in enumerate(queue_items) if queued_task.task_id == task_id) + 1
            response["queue_position"] = position
        except StopIteration:
            response["queue_position"] = 0
    
    if task.started_at:
        response["started_at"] = task.started_at.isoformat()
    
    if task.completed_at:
        response["completed_at"] = task.completed_at.isoformat()
    
    if task.video_filename:
        response["video_filename"] = task.video_filename
    
    if task.error:
        response["error"] = task.error
    
    return response


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Download a generated video file."""
    # Look for the file in the videos directory
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


@app.get("/list")
async def list_videos():
    """List all generated video files with metadata if available."""
    
    videos_dir = OUTPUT_DIR
    
    # Create videos directory if it doesn't exist
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir, exist_ok=True)
        return {"videos": []}
    
    video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
    video_files.sort(key=lambda x: os.path.getctime(os.path.join(videos_dir, x)), reverse=True)
    
    videos_list = []
    for f in video_files:
        video_path = os.path.join(videos_dir, f)
        metadata_path = video_path.replace('.mp4', '.json')
        
        video_info = {
            "filename": f,
            "size": os.path.getsize(video_path),
            "created": datetime.fromtimestamp(os.path.getctime(video_path)).isoformat()
        }
        
        # Add metadata if available
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as meta_file:
                    metadata = json.load(meta_file)
                    video_info["prompt"] = metadata.get("prompt", "")
                    video_info["task"] = metadata.get("task", "")
                    video_info["generation_params"] = metadata.get("generation_params", {})
            except Exception as e:
                print(f"Warning: Failed to read metadata for {f}: {e}")
        
        videos_list.append(video_info)
    
    return {"videos": videos_list}


@app.get("/queue/status")
async def get_queue_status():
    """Get information about the current queue status."""
    queue_size = video_generation_queue.qsize()
    
    # Count tasks by status
    status_counts = {
        TaskStatus.PENDING: 0,
        TaskStatus.PROCESSING: 0,
        TaskStatus.COMPLETED: 0,
        TaskStatus.FAILED: 0
    }
    
    for task in active_tasks.values():
        status_counts[task.status] += 1
    
    return {
        "queue_size": queue_size,
        "max_queue_size": video_generation_queue._maxsize,
        "processing_workers": executor._max_workers,
        "task_counts": status_counts,
        "queue_available": queue_size < video_generation_queue._maxsize
    }


@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get information about the current pipeline status."""
    global global_pipeline, global_pipeline_config
    
    is_loaded = global_pipeline is not None
    config = global_pipeline_config if is_loaded else None
    
    return {
        "pipeline_loaded": is_loaded,
        "configuration": config,
        "queue_empty": is_queue_empty_and_no_processing(),
        "gpu_memory_info": {
            "available": torch.cuda.is_available(),
            "allocated": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            "cached": torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        } if torch.cuda.is_available() else None
    }


@app.post("/pipeline/unload")
async def manual_unload_pipeline():
    """Manually unload the pipeline to free GPU memory."""
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, unload_pipeline)
        return {
            "status": "success",
            "message": "Pipeline unloaded successfully"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to unload pipeline: {str(e)}"
        }


async def process_video_generation_queue():
    """Background task processor that continuously processes the video generation queue."""
    print("🚀 Starting video generation queue processor...")
    
    while True:
        try:
            # Get next task from queue
            task = await video_generation_queue.get()
            
            # Update task status
            task.status = TaskStatus.PROCESSING
            task.message = "Processing video generation..."
            task.started_at = datetime.now()
            
            print(f"📹 Processing task {task.task_id}")
            
            # Process the task in a thread pool
            loop = asyncio.get_event_loop()
            try:
                video_path = await loop.run_in_executor(
                    executor,
                    process_single_video_generation,
                    task
                )
                
                # Update task on success
                task.status = TaskStatus.COMPLETED
                task.message = "Video generated successfully"
                task.video_filename = os.path.basename(video_path)
                task.completed_at = datetime.now()
                
                print(f"✅ Completed task {task.task_id}: {video_path}")
                
            except Exception as e:
                # Update task on error
                task.status = TaskStatus.FAILED
                task.message = "Video generation failed"
                task.error = str(e)
                task.completed_at = datetime.now()
                
                print(f"❌ Failed task {task.task_id}: {e}")
            
            finally:
                # Mark task as done in queue
                video_generation_queue.task_done()
                
                # Check if queue is empty and no other tasks are processing
                # If so, unload the pipeline to free GPU memory
                await asyncio.sleep(0.1)  # Small delay to allow other tasks to start
                if is_queue_empty_and_no_processing():
                    print("📤 Queue is empty, unloading pipeline to free GPU memory...")
                    await loop.run_in_executor(executor, unload_pipeline)

                
        except Exception as e:
            print(f"❌ Queue processor error: {e}")
            await asyncio.sleep(1)  # Brief pause before continuing


def process_single_video_generation(task: VideoGenerationTask) -> str:
    """Process a single video generation task in a separate thread."""
    request = task.request
    
    return generate_wan_video(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        task=request.task,
        model_id=request.model_id,
        size=request.size,
        num_frames=request.num_frames,
        sample_guide_scale=request.sample_guide_scale,
        sample_shift=request.sample_shift,
        sample_steps=request.sample_steps,
        fps=request.fps,
        offload_model=request.offload_model,
        t5_cpu=request.t5_cpu,
        output_path=None
    )


def main():
    print("Starting Wan Diffusers API server...")
    print("API Documentation available at: http://localhost:8000/docs")
    print("Alternative docs at: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
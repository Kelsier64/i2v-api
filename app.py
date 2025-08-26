import os
import sys
import gc
import weakref
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio
import uuid
import json
import torch
import threading
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("./videos")

# Constants
MAX_WORKERS = 2
MEMORY_CLEANUP_THRESHOLD = 0.1  # Clean up when less than 10% GPU memory available

class MemoryManager:
    """Enhanced memory management for GPU resources."""
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory information in GB."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "cached": 0, "total": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        return {
            "allocated": allocated,
            "cached": cached, 
            "total": total,
            "free": free
        }
    
    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive GPU memory cleanup."""
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleaned up")
    
    @staticmethod
    def should_cleanup_memory() -> bool:
        """Check if memory cleanup is needed."""
        memory_info = MemoryManager.get_gpu_memory_info()
        if memory_info["total"] > 0:
            free_ratio = memory_info["free"] / memory_info["total"]
            return free_ratio < MEMORY_CLEANUP_THRESHOLD
        return False

def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string like '1280*720' into (width, height)."""
    try:
        width, height = map(int, size_str.split('*'))
        return width, height
    except ValueError as e:
        logger.error(f"Invalid size format: {size_str}")
        raise ValueError(f"Invalid size format. Expected 'width*height', got: {size_str}") from e


def get_flow_shift(width: int, height: int, sample_shift: Optional[float] = None) -> float:
    """Determine flow shift based on resolution."""
    if sample_shift is not None:
        return sample_shift
    
    # Auto-determine based on resolution
    return 3.0 if height == 480 or width == 480 else 5.0


def generate_output_filename(videos_dir: Path, task: str) -> str:
    """Generate output filename based on UUID and timestamp."""
    # Generate a unique UUID for the video
    video_uuid = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create videos directory if it doesn't exist
    videos_dir.mkdir(exist_ok=True)
    
    filename = f"{task}_{timestamp}_{video_uuid}.mp4"
    return str(videos_dir / filename)


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
    num_frames: int = Field(default=81, ge=1, description="Number of frames to generate")
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


# Initialize FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("🚀 Starting application...")
    
    # Startup
    asyncio.create_task(process_video_generation_queue())
    yield
    
    # Shutdown
    logger.info("📤 Shutting down application...")
    await cleanup_resources()

app = FastAPI(
    title="Wan Diffusers Video Generation API",
    description="Optimized API for generating videos using Wan2.1 diffusion models with enhanced queue management",
    version="1.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for enhanced state management
active_tasks: Dict[str, 'VideoGenerationTask'] = {}
video_generation_queue = asyncio.Queue()
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Enhanced pipeline management
global_pipeline = None
global_pipeline_config = None
pipeline_lock = threading.Lock()
pipeline_last_used = None

async def cleanup_resources():
    """Clean up resources on shutdown."""
    logger.info("Cleaning up resources...")
    if global_pipeline is not None:
        unload_pipeline()
    executor.shutdown(wait=True)
    MemoryManager.cleanup_gpu_memory()

# Task processing status
class TaskStatus:
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class VideoGenerationTask:
    """Enhanced task management with better error handling and monitoring."""
    
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
        self.progress = 0.0
        
    @property
    def duration(self) -> Optional[float]:
        """Get task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "video_filename": self.video_filename,
            "error": self.error
        }


def load_pipeline_if_needed(model_id: str, offload_model: bool = True, t5_cpu: bool = True, flow_shift: float = 5.0) -> WanPipeline:
    """
    Optimized pipeline loading with enhanced memory management.
    
    Args:
        model_id: HuggingFace model ID
        offload_model: Enable CPU offloading to reduce GPU memory
        t5_cpu: Keep T5 text encoder on CPU
        flow_shift: Flow shift parameter
    
    Returns:
        WanPipeline: The loaded pipeline instance
    """
    global global_pipeline, global_pipeline_config, pipeline_last_used
    
    current_config = {
        'model_id': model_id,
        'offload_model': offload_model,
        't5_cpu': t5_cpu,
        'flow_shift': flow_shift
    }
    
    with pipeline_lock:
        # Check if pipeline needs to be loaded/reloaded
        if global_pipeline is None or global_pipeline_config != current_config:
            logger.info(f"Loading pipeline with config: {current_config}")
            
            # Cleanup memory before loading
            if global_pipeline is not None:
                logger.info("Unloading existing pipeline...")
                unload_pipeline()
            
            # Additional memory cleanup if needed
            if MemoryManager.should_cleanup_memory():
                MemoryManager.cleanup_gpu_memory()
            
            try:
                # Load VAE with error handling
                logger.info("Loading VAE...")
                vae = AutoencoderKLWan.from_pretrained(
                    model_id, 
                    subfolder="vae", 
                    torch_dtype=torch.float32,
                    local_files_only=False
                )
                
                # Setup scheduler
                scheduler = UniPCMultistepScheduler(
                    prediction_type='flow_prediction', 
                    use_flow_sigmas=True, 
                    num_train_timesteps=1000, 
                    flow_shift=flow_shift
                )
                
                # Load pipeline with better error handling
                logger.info("Loading main pipeline...")
                global_pipeline = WanPipeline.from_pretrained(
                    model_id, 
                    vae=vae, 
                    torch_dtype=torch.bfloat16,
                    local_files_only=False
                )
                global_pipeline.scheduler = scheduler
                
                # Apply memory optimizations
                if offload_model:
                    logger.info("Enabling model CPU offloading...")
                    global_pipeline.enable_model_cpu_offload()
                
                if t5_cpu:
                    logger.info("Moving T5 text encoder to CPU...")
                    if hasattr(global_pipeline, 'text_encoder') and global_pipeline.text_encoder is not None:
                        global_pipeline.text_encoder = global_pipeline.text_encoder.to('cpu')
                
                if not offload_model:
                    global_pipeline.to("cuda")
                
                global_pipeline_config = current_config
                pipeline_last_used = datetime.now()
                
                # Log memory usage after loading
                memory_info = MemoryManager.get_gpu_memory_info()
                logger.info(f"Pipeline loaded. GPU memory: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")
                    
            except Exception as e:
                logger.error(f"Failed to load pipeline: {e}")
                global_pipeline = None
                global_pipeline_config = None
                MemoryManager.cleanup_gpu_memory()
                raise e
        else:
            # Update last used time for existing pipeline
            pipeline_last_used = datetime.now()
        
        return global_pipeline


def unload_pipeline():
    """Unload the pipeline and free GPU memory without moving to CPU."""
    global global_pipeline, global_pipeline_config

    if global_pipeline is not None:
        logger.info("Unloading pipeline to free GPU memory...")
        try:
            # Just clear references, do not move to CPU
            global_pipeline = None
            global_pipeline_config = None

            # Enhanced memory cleanup
            MemoryManager.cleanup_gpu_memory()

            # Log memory status after cleanup
            memory_info = MemoryManager.get_gpu_memory_info()
            logger.info(f"Pipeline unloaded. GPU memory freed: {memory_info['free']:.2f}GB available")

        except Exception as e:
            logger.error(f"Error during pipeline unload: {e}")


def is_queue_empty_and_no_processing() -> bool:
    """Check if queue is empty and no tasks are currently processing."""
    queue_empty = video_generation_queue.qsize() == 0
    
    # Check if any tasks are currently processing
    processing_tasks = sum(1 for task in active_tasks.values() if task.status == TaskStatus.PROCESSING)
    
    return queue_empty and processing_tasks == 0




def generate_wan_video(
    prompt: str = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window.",
    negative_prompt: str = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    task: str = "t2v-1.3B",
    model_id: Optional[str] = None,
    size: str = "832*480",
    num_frames: int = 81,
    sample_guide_scale: float = 6.0,
    sample_shift: Optional[float] = None,
    sample_steps: int = 50,
    fps: int = 16,
    offload_model: bool = True,
    t5_cpu: bool = True,
    output_path: Optional[str] = None
) -> str:
    """
    Optimized video generation using Wan2.1 model with enhanced error handling.
    
    Args:
        prompt: Text prompt for video generation
        negative_prompt: Negative prompt to avoid certain content
        task: Model variant - 't2v-1.3B' or 't2v-14B'
        model_id: HuggingFace model ID (overrides task)
        size: Video resolution in format 'width*height'
        num_frames: Number of frames to generate
        sample_guide_scale: Guidance scale for sampling
        sample_shift: Flow shift parameter (auto-determined if None)
        sample_steps: Number of sampling steps
        fps: Output video frame rate
        offload_model: Enable CPU offloading to reduce GPU memory
        t5_cpu: Keep T5 text encoder on CPU
        output_path: Output video file path (auto-generated if None)
    
    Returns:
        str: The output path where the video was saved
    """
    
    # Parse video dimensions with validation
    try:
        width, height = parse_size(size)
    except ValueError as e:
        logger.error(f"Invalid size format: {size}")
        raise ValueError(f"Invalid video size: {size}") from e

    # Determine flow shift
    flow_shift = get_flow_shift(width, height, sample_shift)
    
    logger.info(f"Generating video: {width}x{height}, {num_frames} frames, flow_shift={flow_shift}")
    logger.info(f"Model: {model_id}, CPU offload: {offload_model}, T5 CPU: {t5_cpu}")
    
    # Load or reuse existing pipeline
    try:
        pipe = load_pipeline_if_needed(model_id, offload_model, t5_cpu, flow_shift)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise RuntimeError(f"Pipeline loading failed: {e}") from e
    
    logger.info(f"Generating video with prompt: {prompt[:100]}...")
    
    # Generate video with enhanced error handling
    try:
        # Check memory before generation
        memory_info = MemoryManager.get_gpu_memory_info()
        logger.info(f"Pre-generation memory: {memory_info['allocated']:.2f}GB allocated")
        
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
        final_output_path = output_path or generate_output_filename(OUTPUT_DIR, task)

        # Export video with validation
        export_to_video(output, final_output_path, fps=fps)
        
        # Validate output file
        if not Path(final_output_path).exists():
            raise RuntimeError(f"Video export failed - file not created: {final_output_path}")
        
        file_size = Path(final_output_path).stat().st_size
        logger.info(f"Video saved: {final_output_path} ({file_size / 1024 / 1024:.2f}MB)")
        
        # Cleanup memory after generation
        if MemoryManager.should_cleanup_memory():
            MemoryManager.cleanup_gpu_memory()
        
        return final_output_path
        
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        # Cleanup on error
        MemoryManager.cleanup_gpu_memory()
        raise RuntimeError(f"Video generation failed: {e}") from e


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with enhanced API information."""
    memory_info = MemoryManager.get_gpu_memory_info()
    
    return {
        "message": "Optimized Wan Diffusers Video Generation API with Enhanced Queue Management",
        "version": "1.1.0",
        "status": "operational",
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "memory_allocated_gb": memory_info["allocated"],
            "memory_free_gb": memory_info["free"]
        },
        "features": [
            "Asynchronous video generation with enhanced queue management",
            "Optimized memory management and GPU resource handling",
            "Real-time task progress monitoring",
            "Intelligent pipeline lifecycle management",
            "Automatic resource cleanup and optimization",
            "Enhanced error handling and recovery"
        ],
        "endpoints": {
            "generate": "/generate - POST request to generate video (queued)",
            "status": "/status/{task_id} - GET request to check generation status",
            "queue_status": "/queue/status - GET request to check queue status",
            "queue_clean": "/queue/clean - GET request to clean the queue and remove all tasks",
            "pipeline_status": "/pipeline/status - GET request to check pipeline status and GPU memory",
            "pipeline_unload": "/pipeline/unload - POST request to manually unload pipeline",
            "download": "/download/{filename} - GET request to download generated video",
            "list": "/list - GET request to list all generated videos",
            "videos_clean": "/videos/clean - GET request to clean videos directory",
            "health": "/health - GET request to check API health"
        }
    }


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with detailed system information."""
    gpu_available = torch.cuda.is_available()
    memory_info = MemoryManager.get_gpu_memory_info()
    
    # Check queue health
    queue_health = video_generation_queue.qsize() < video_generation_queue._maxsize
    
    # Check for any failed tasks recently
    recent_failures = sum(1 for task in active_tasks.values() 
                         if task.status == TaskStatus.FAILED and 
                         (datetime.now() - task.created_at).total_seconds() < 3600)
    
    return {
        "status": "healthy" if queue_health and recent_failures < 5 else "degraded",
        "gpu_available": gpu_available,
        "gpu_count": torch.cuda.device_count() if gpu_available else 0,
        "memory_info": memory_info,
        "queue_health": queue_health,
        "recent_failures": recent_failures,
        "pipeline_loaded": global_pipeline is not None,
        "active_tasks": len(active_tasks),
        "queue_size": video_generation_queue.qsize()
    }


@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    """
    Enhanced video generation endpoint with improved validation and error handling.
    Returns immediately with a task ID for async processing using a queue.
    """
    # Generate unique task ID
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    # Validate request parameters
    try:
        parse_size(request.size)  # Validate size format
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid size format: {request.size}")
    
    if request.num_frames < 1 or request.num_frames > 241:
        raise HTTPException(status_code=400, detail="num_frames must be between 1 and 241")
    
    if request.sample_steps < 10 or request.sample_steps > 100:
        raise HTTPException(status_code=400, detail="sample_steps must be between 10 and 100")
    
    # Create task object
    task = VideoGenerationTask(task_id, request)
    
    # Store task info
    active_tasks[task_id] = task
    
    try:
        # Add task to queue (this will raise QueueFull if queue is at max capacity)
        video_generation_queue.put_nowait(task)
        
        queue_position = video_generation_queue.qsize()
        logger.info(f"Task {task_id} queued at position {queue_position}")
        
        return VideoGenerationResponse(
            status="accepted",
            message=f"Video generation queued. Position: {queue_position}",
            task_id=task_id
        )
        
    except asyncio.QueueFull:
        # Remove from active tasks if queue is full
        del active_tasks[task_id]
        logger.warning(f"Queue full, rejecting task {task_id}")
        raise HTTPException(
            status_code=503, 
            detail="Server busy. Too many pending requests. Please try again later."
        )



@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Enhanced task status endpoint with detailed progress information."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = active_tasks[task_id]
    response = task.to_dict()
    
    # Add queue position if task is still pending
    if task.status == TaskStatus.PENDING:
        try:
            # Calculate position in queue more efficiently
            queue_items = list(video_generation_queue._queue)
            position = next(
                (i + 1 for i, queued_task in enumerate(queue_items) 
                 if queued_task.task_id == task_id), 
                0
            )
            response["queue_position"] = position
        except Exception:
            response["queue_position"] = None
    
    return response


@app.get("/download/{filename}")
async def download_video(filename: str):
    """Enhanced video download endpoint with security validation."""
    # Security: Validate filename to prevent directory traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")
    
    # Look for the file in the videos directory
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        logger.warning(f"Requested file not found: {filename}")
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Additional security check
    if not str(file_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    return FileResponse(
        path=str(file_path),
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


@app.get("/queue/clean")
async def clean_queue():
    """Clean the video generation queue and remove all active tasks."""
    global active_tasks
    
    # Clear the queue
    while not video_generation_queue.empty():
        try:
            video_generation_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
    
    # Count tasks before cleaning
    total_tasks = len(active_tasks)
    pending_tasks = sum(1 for task in active_tasks.values() if task.status == TaskStatus.PENDING)
    processing_tasks = sum(1 for task in active_tasks.values() if task.status == TaskStatus.PROCESSING)
    
    # Clear active tasks
    active_tasks.clear()
    
    logger.info(f"Queue cleaned: removed {total_tasks} tasks ({pending_tasks} pending, {processing_tasks} processing)")
    
    return {
        "status": "success",
        "message": "Queue and active tasks cleaned successfully",
        "tasks_removed": {
            "total": total_tasks,
            "pending": pending_tasks,
            "processing": processing_tasks
        },
        "queue_size": video_generation_queue.qsize()
    }


@app.get("/videos/clean")
async def clean_videos():
    """Clean the videos directory by removing all generated video files."""
    videos_dir = OUTPUT_DIR
    
    # Ensure videos directory exists
    if not videos_dir.exists():
        return {
            "status": "success",
            "message": "Videos directory does not exist",
            "files_removed": 0,
            "space_freed_mb": 0
        }
    
    # Count and remove video files
    removed_files = 0
    total_size = 0
    
    try:
        for file_path in videos_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.json']:
                file_size = file_path.stat().st_size
                total_size += file_size
                file_path.unlink()
                removed_files += 1
                logger.info(f"Removed file: {file_path.name}")
        
        space_freed_mb = total_size / (1024 * 1024)
        
        logger.info(f"Videos directory cleaned: removed {removed_files} files, freed {space_freed_mb:.2f}MB")
        
        return {
            "status": "success",
            "message": f"Videos directory cleaned successfully",
            "files_removed": removed_files,
            "space_freed_mb": round(space_freed_mb, 2)
        }
        
    except Exception as e:
        logger.error(f"Error cleaning videos directory: {e}")
        return {
            "status": "error",
            "message": f"Failed to clean videos directory: {str(e)}",
            "files_removed": removed_files,
            "space_freed_mb": round(total_size / (1024 * 1024), 2)
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
# Wan Diffusers Video Generation API

A FastAPI-based REST API for generating videos using the Wan2.1 diffusion model with **queue management** and **multithreading support**.

## 🚀 Key Features

- **Queue-based processing**: Handles multiple concurrent requests efficiently
- **Multithreading support**: Process multiple videos simultaneously with configurable workers
- **Real-time status tracking**: Monitor queue position and task progress
- **Automatic load balancing**: Prevents server overload with queue limits
- **Thread-safe operations**: Safe concurrent access to resources

## 🏗️ Architecture

The API uses:
- **AsyncIO Queue**: For managing video generation requests
- **ThreadPoolExecutor**: For concurrent video processing (configurable workers)
- **Task Status Tracking**: Real-time monitoring of task progress
- **Queue Overflow Protection**: Graceful handling when server is busy

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Option 1: Using the startup script
python start_api.py

# Option 2: Using the main script
python main.py

# Option 3: Using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Health Check
```
GET /health
```
Check if the API is running and get GPU information.

### Generate Video (Async with Queue)
```
POST /generate
```
Submit video generation request to the queue. Returns immediately with a task ID.

**Request Body:**
```json
{
  "prompt": "A majestic dragon flying over mountains",
  "negative_prompt": "blurry, low quality",
  "size": "832*480",
  "num_frames": 81,
  "sample_guide_scale": 6.0,
  "sample_steps": 50,
  "fps": 16,
  "offload_model": true,
  "t5_cpu": true
}
```

**Response:**
```json
{
  "status": "accepted",
  "message": "Video generation queued. Position in queue: 2",
  "task_id": "task_20240815_123456_789012"
}
```

### Check Queue Status
```
GET /queue/status
```
Get information about the current queue and processing status.

**Response:**
```json
{
  "queue_size": 3,
  "max_queue_size": 10,
  "processing_workers": 2,
  "task_counts": {
    "pending": 3,
    "processing": 2,
    "completed": 15,
    "failed": 1
  },
  "queue_available": true
}
```

### Check Task Status
```
GET /status/{task_id}
```
Check the status of a video generation task with queue position.

**Response:**
```json
{
  "task_id": "task_20240815_123456_789012",
  "status": "pending",
  "message": "Task queued for processing", 
  "queue_position": 3,
  "created_at": "2024-08-15T12:34:56.789012",
  "started_at": null,
  "completed_at": null
}
```

**Status Values:**
- `pending`: Task is queued waiting for processing
- `processing`: Task is currently being processed
- `completed`: Task finished successfully
- `failed`: Task failed with error

### Download Video
```
GET /download/{filename}
```
Download a generated video file.

### List Videos
```
GET /list
```
List all generated video files with metadata.

## ⚙️ Queue Configuration

You can modify the queue settings in `app.py`:

```python
# Queue settings (in app.py)
video_generation_queue = asyncio.Queue(maxsize=10)  # Max pending tasks
executor = ThreadPoolExecutor(max_workers=2)        # Concurrent workers
```

### Recommended Settings by Hardware

| GPU Memory | Max Workers | Queue Size | Notes |
|------------|-------------|------------|--------|
| 8GB | 1 | 5 | Conservative for single GPU |
| 16GB | 2 | 10 | Good balance |
| 24GB+ | 3-4 | 15-20 | High throughput |

### Queue Behavior

- **Queue Full**: Returns HTTP 503 with retry message
- **Multiple Workers**: Process videos concurrently on different GPU contexts
- **Memory Management**: Each worker uses model offloading independently
- **Fair Processing**: First-in-first-out queue order

## 🔧 Video Generation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | - | Text description of the video to generate |
| `negative_prompt` | string | - | What to avoid in the generation |
| `size` | string | "832*480" | Video resolution (width*height) |
| `num_frames` | integer | 81 | Number of frames (1-200) |
| `sample_guide_scale` | float | 6.0 | Guidance scale (1.0-20.0) |
| `sample_steps` | integer | 50 | Sampling steps (10-100) |
| `fps` | integer | 16 | Output video frame rate |
| `offload_model` | boolean | true | Enable CPU offloading for memory |
| `t5_cpu` | boolean | true | Keep text encoder on CPU |

## 💻 Example Usage

### Python Client

```python
import requests

# Start async generation
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "A cat playing piano in a cozy room",
    "num_frames": 81,
    "fps": 16
})

task_id = response.json()["task_id"]

# Check status
status = requests.get(f"http://localhost:8000/status/{task_id}")
print(status.json())

# Download when complete
if status.json()["status"] == "completed":
    video_path = status.json()["video_path"]
    video_data = requests.get(f"http://localhost:8000/download/{video_path}")
    
    with open("my_video.mp4", "wb") as f:
        f.write(video_data.content)
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Generate video (async)
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "num_frames": 41,
    "sample_steps": 30
  }'

# Check status
curl http://localhost:8000/status/task_20240814_123456_789

# List videos
curl http://localhost:8000/list

# Download video
curl -O http://localhost:8000/download/video_filename.mp4
```

### JavaScript/Fetch

```javascript
// Generate video
const response = await fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'A robot dancing in space',
    num_frames: 81,
    fps: 16
  })
});

const result = await response.json();
const taskId = result.task_id;

// Poll for completion
const checkStatus = async () => {
  const statusResponse = await fetch(`http://localhost:8000/status/${taskId}`);
  const status = await statusResponse.json();
  
  if (status.status === 'completed') {
    // Download the video
    window.open(`http://localhost:8000/download/${status.video_path}`);
  } else if (status.status === 'processing') {
    setTimeout(checkStatus, 5000); // Check again in 5 seconds
  }
};

checkStatus();
```

## 🧪 Testing

### Test Basic API Functionality
Run the original example client:
```bash
python api_client_example.py
```

### Test Queue Management
Run the new queue-based client to test multithreading:
```bash
python queue_client_example.py
```

This will:
1. Check API health and queue status
2. Submit multiple video generation requests
3. Monitor queue positions and processing
4. Demonstrate queue overflow handling
5. Download completed videos

### Test Multiple Concurrent Requests
```bash
# Submit multiple requests quickly to test queue
for i in {1..5}; do
  curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Test video '${i}'", "num_frames": 17}' &
done
wait

# Check queue status
curl http://localhost:8000/queue/status
```

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Running the Original Script

You can still run the original video generation script directly:

```bash
python main.py test
```

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB GPU memory for default settings
- FastAPI and dependencies (see requirements.txt)

## 🔍 Troubleshooting

### Common Issues

1. **GPU Memory Error**: Reduce `num_frames`, enable `offload_model=true`, or use `t5_cpu=true`
2. **Slow Generation**: Use fewer `sample_steps` or smaller `size`
3. **API Not Starting**: Check if port 8000 is available or change the port
4. **Model Download**: First run may take time to download the model

### Performance Tips

- Use `offload_model=true` and `t5_cpu=true` for lower memory usage
- Reduce `num_frames` for faster generation
- Reduce `sample_steps` for faster (but potentially lower quality) results
- Use smaller resolutions for testing

## 📄 License

[Your license information here]

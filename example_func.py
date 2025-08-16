import requests,time
BASE_URL = "http://localhost:8000"

def get_queue_status():
    """Get current queue status"""
    response = requests.get(f"{BASE_URL}/queue/status")
    if response.status_code == 200:
        status = response.json()
        print("📊 Queue Status:")
        print(f"  Queue size: {status['queue_size']}/{status['max_queue_size']}")
        print(f"  Processing workers: {status['processing_workers']}")
        print(f"  Task counts: {status['task_counts']}")
        print(f"  Queue available: {status['queue_available']}")
        return status
    else:
        print(f"❌ Failed to get queue status: {response.text}")
        return None

def submit_video_generation(prompt, task_name=None):
    """Submit a video generation request and return task ID"""
    request_data = {
        "prompt": prompt,
        "negative_prompt": "blurry, low quality, distorted",
        "task": task_name or "t2v-1.3B",
        "size": "832*480",
        "sample_guide_scale": 6.0,
        "sample_steps": 30,  # Reduced for faster testing
        "num_frames": 17,  # Reduced for faster testing
        "fps": 16,
    }
    
    print(f"🎬 Submitting: {prompt[:50]}...")
    
    response = requests.post(f"{BASE_URL}/generate", json=request_data)
    
    if response.status_code == 200:
        result = response.json()
        task_id = result["task_id"]
        print(f"✅ Task submitted: {task_id}")
        print(f"📋 Status: {result['message']}")
        return task_id
    elif response.status_code == 503:
        print(f"⚠️ Server busy: {response.json().get('detail', 'Unknown error')}")
        return None
    else:
        print(f"❌ Failed to submit: {response.text}")
        return None
    
def download_video(filename):
    """Download a video file"""
    response = requests.get(f"{BASE_URL}/download/{filename}")
    
    if response.status_code == 200:
        local_filename = f"downloaded_{filename}"
        with open(local_filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {local_filename}")
        return True
    else:
        print(f"❌ Failed to download {filename}: {response.text}")
        return False

def monitor_task(task_id, poll_interval=10):
    """Monitor a task until completion"""
    print(f"👀 Monitoring task: {task_id}")
    
    while True:
        response = requests.get(f"{BASE_URL}/status/{task_id}")
        if response.status_code != 200:
            print(f"❌ Failed to get task status: {response.text}")
            break
            
        status = response.json()
        status_str = status['status']
        message = status['message']
        
        if status.get('queue_position'):
            print(f"📊 {status_str.upper()}: {message} (Position: {status['queue_position']})")
        else:
            print(f"📊 {status_str.upper()}: {message}")
        
        if status_str == 'completed':
            video_filename = status.get('video_filename')
            print(f"✅ Video completed: {video_filename}")
            return video_filename
        elif status_str == 'failed':
            error = status.get('error', 'Unknown error')
            print(f"❌ Task failed: {error}")
            break
        
        time.sleep(poll_interval)
    
    return None

def download_video(filename):
    """Download a video file"""
    response = requests.get(f"{BASE_URL}/download/{filename}")
    
    if response.status_code == 200:
        local_filename = f"downloaded_{filename}"
        with open(local_filename, "wb") as f:
            f.write(response.content)
        print(f"✅ Downloaded: {local_filename}")
        return True
    else:
        print(f"❌ Failed to download {filename}: {response.text}")
        return False
    
def main():
    task_id = submit_video_generation("A cat playing piano in a cozy jazz club", "test-cat-piano")
    video_filename = monitor_task(task_id)
    if video_filename:
        download_video(video_filename)

if __name__ == "__main__":
    main()
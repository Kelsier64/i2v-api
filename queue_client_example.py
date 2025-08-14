#!/usr/bin/env python3
"""
Example client demonstrating the queue-based video generation API
"""

import requests
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print("❌ API health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running.")
        return False

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

def monitor_task(task_id, poll_interval=5):
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

def test_multiple_submissions():
    """Test submitting multiple videos and monitoring them"""
    print("\n🚀 Testing multiple video submissions...")
    
    # Define multiple prompts
    prompts = [
        "A cat playing piano in a cozy jazz club",
        "A robot dancing in a futuristic city",
        "A dragon flying over mountains at sunset",
        "A spaceship landing on an alien planet",
        "A magical forest with glowing flowers"
    ]
    
    # Submit all requests
    task_ids = []
    for i, prompt in enumerate(prompts):
        task_id = submit_video_generation(prompt, f"test-{i+1}")
        if task_id:
            task_ids.append((task_id, prompt))
        
        # Check queue status after each submission
        get_queue_status()
        print()
        
        # Small delay between submissions
        time.sleep(1)
    
    print(f"\n📋 Submitted {len(task_ids)} tasks. Monitoring progress...")
    
    # Monitor all tasks
    completed_videos = []
    for task_id, prompt in task_ids:
        print(f"\n--- Monitoring: {prompt[:30]}... ---")
        video_filename = monitor_task(task_id, poll_interval=3)
        if video_filename:
            completed_videos.append(video_filename)
    
    print(f"\n🎉 Completed {len(completed_videos)} videos!")
    
    # Download first completed video as example
    if completed_videos:
        print(f"\n📥 Downloading first video: {completed_videos[0]}")
        download_video(completed_videos[0])
    
    return completed_videos

def test_queue_overflow():
    """Test what happens when queue is full"""
    print("\n⚠️ Testing queue overflow behavior...")
    
    # Get current queue status
    status = get_queue_status()
    if not status:
        return
    
    max_queue = status['max_queue_size']
    current_size = status['queue_size']
    
    print(f"📊 Current queue: {current_size}/{max_queue}")
    
    # Try to fill the queue
    tasks_to_submit = max_queue - current_size + 2  # Try to submit 2 extra
    print(f"🎯 Will attempt to submit {tasks_to_submit} tasks...")
    
    submitted = 0
    rejected = 0
    
    for i in range(tasks_to_submit):
        task_id = submit_video_generation(f"Queue test video {i+1}", f"overflow-test-{i+1}")
        if task_id:
            submitted += 1
        else:
            rejected += 1
        
        if i % 3 == 0:  # Check queue status every few submissions
            get_queue_status()
        
        time.sleep(0.5)  # Small delay
    
    print(f"\n📈 Results: {submitted} submitted, {rejected} rejected")
    get_queue_status()

def main():
    """Main function demonstrating queue-based API usage"""
    print("🔧 Wan Diffusers Queue-Based API Client")
    print("=" * 50)
    
    # Test API health
    if not test_api_health():
        return
    
    print("\n" + "=" * 50)
    
    # Get initial queue status
    print("\n📊 Initial queue status:")
    get_queue_status()
    
    # Test 1: Multiple submissions
    test_multiple_submissions()
    
    print("\n" + "=" * 50)
    
    # Test 2: Queue overflow (optional - uncomment to test)
    # test_queue_overflow()
    
    print("\n✨ Queue demo completed!")
    print("💡 Try the API documentation at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()

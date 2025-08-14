#!/usr/bin/env python3
"""
Startup script for the Wan Diffusers API
"""

import uvicorn
from app import app

if __name__ == "__main__":
    print("🚀 Starting Wan Diffusers Video Generation API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("📚 Alternative docs: http://localhost:8000/redoc")
    print("💡 Health check: http://localhost:8000/health")
    print("📋 List videos: http://localhost:8000/list")
    print()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Set to True for development
        access_log=True
    )

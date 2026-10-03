import asyncio
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from reiduq.pipeline.processor import LiveProcessor

live_router = APIRouter()

# Global state for cameras
cameras: Dict[str, LiveProcessor] = {}
background_tasks = []

@live_router.on_event("startup")
async def startup_event():
    # Load from environment variables (e.g., CAMERA_ID=cam_01, STREAM_URL=0 for webcam)
    cam_id = os.environ.get("CAMERA_ID", "cam_01")
    stream_url = os.environ.get("STREAM_URL", "0") # 0 = local webcam default
    
    # Check for multiple cameras (e.g. CAMERA_01_URL, CAMERA_02_URL)
    # We will just setup the primary one from env variables
    processor = LiveProcessor(camera_id=cam_id, stream_url=int(stream_url) if stream_url.isdigit() else stream_url)
    cameras[cam_id] = processor
    
    # Start the processing loop in background
    task = asyncio.create_task(processor.process_loop())
    background_tasks.append(task)

@live_router.on_event("shutdown")
async def shutdown_event():
    for cam in cameras.values():
        cam.stop()
    for task in background_tasks:
        task.cancel()

@live_router.get("/vehicles")
async def get_vehicles():
    """Return all unique vehicles currently tracked across all cameras."""
    tracked = []
    for cam_id, processor in cameras.items():
        for track_id, global_id in processor.reid_cache.items():
            tracked.append({"camera_id": cam_id, "track_id": track_id, "vehicle_id": global_id})
    return {"status": "ok", "vehicles": tracked}

@live_router.get("/vehicles/{vehicle_id}")
async def get_vehicle(vehicle_id: str):
    """Get metadata for a specific global vehicle ID."""
    return {"status": "ok", "vehicle_id": vehicle_id, "details": "Lookup FAISS/Store here"}

@live_router.get("/cameras")
async def get_cameras():
    return {"status": "ok", "cameras": list(cameras.keys())}

@live_router.get("/camera/{camera_id}/vehicles")
async def get_camera_vehicles(camera_id: str):
    if camera_id not in cameras:
        return {"error": "Camera not found"}
    processor = cameras[camera_id]
    return {"status": "ok", "camera_id": camera_id, "tracked_vehicles": processor.reid_cache}

from pydantic import BaseModel

class CameraConfig(BaseModel):
    stream_url: str

@live_router.post("/camera/{camera_id}/config")
async def update_camera_config(camera_id: str, config: CameraConfig):
    if camera_id not in cameras:
        # Create a new one
        processor = LiveProcessor(camera_id=camera_id, stream_url=int(config.stream_url) if config.stream_url.isdigit() else config.stream_url)
        cameras[camera_id] = processor
        task = asyncio.create_task(processor.process_loop())
        background_tasks.append(task)
        return {"status": "created", "camera_id": camera_id, "stream_url": config.stream_url}
        
    # Update existing
    old_processor = cameras[camera_id]
    old_processor.stop()
    
    # Wait for the old task to finish (we just let it die in background)
    
    new_processor = LiveProcessor(camera_id=camera_id, stream_url=int(config.stream_url) if config.stream_url.isdigit() else config.stream_url)
    cameras[camera_id] = new_processor
    task = asyncio.create_task(new_processor.process_loop())
    background_tasks.append(task)
    
    return {"status": "updated", "camera_id": camera_id, "stream_url": config.stream_url}

@live_router.websocket("/ws/vehicles")
async def websocket_vehicles(websocket: WebSocket, camera_id: str = None):
    # If no camera specified, use the first one available
    cam_id = camera_id if camera_id else list(cameras.keys())[0]
    if cam_id not in cameras:
        await websocket.close(code=1000)
        return
        
    processor = cameras[cam_id]
    await processor.add_client(websocket)
    
    try:
        while True:
            # Keep connection alive, listen for disconnects
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        processor.websockets.remove(websocket)

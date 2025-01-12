from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import open3d as o3d
import os
import uuid
import cv2
import base64
import asyncio
import json
from datetime import datetime

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@app.post("/upload-ply/")
async def upload_ply(file: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file.filename.endswith('.ply'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .ply files are allowed"}
            )

        # Generate unique filename
        unique_filename = f"{str(uuid.uuid4())}.ply"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)

        # Load and process the point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Extract points and convert to list for JSON serialization
        points = np.asarray(pcd.points).tolist()
        
        # Extract colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors).tolist()
        else:
            colors = [[0.5, 0.5, 0.5] for _ in points]  # Default gray color

        # Calculate bounding box
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound.tolist()
        max_bound = bbox.max_bound.tolist()

        # Basic statistics
        stats = {
            "num_points": len(points),
            "bounding_box": {
                "min": min_bound,
                "max": max_bound
            }
        }

        return {
            "filename": unique_filename,
            "points": points,
            "colors": colors,
            "statistics": stats,
            "success": True
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/point-cloud/{filename}")
async def get_point_cloud(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "File not found"}
            )

        # Load point cloud
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points).tolist()
        colors = np.asarray(pcd.colors).tolist() if pcd.has_colors() else None

        return {
            "points": points,
            "colors": colors,
            "success": True
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/point-cloud/{filename}")
async def delete_point_cloud(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return {"success": True, "message": "File deleted successfully"}
        return JSONResponse(
            status_code=404,
            content={"error": "File not found"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

class CameraManager:
    def __init__(self):
        self.cap = None
        self.is_recording = False
        self.recording_frames = []
        
    async def initialize(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Get metrics
                fps = 10
                #self.cap.get(cv2.CAP_PROP_FPS)
                frame_count = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
                # Convert frame to base64
                _, buffer = cv2.imencode('.jpg', frame)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                if self.is_recording:
                    self.recording_frames.append(frame)
                
                return {
                    'frame': frame_b64,
                    'metrics': {
                        'fps': fps,
                        'width': width,
                        'height': height,                     
                        'timestamp': datetime.now().isoformat()
                    }
                }
        return None
    
    def start_recording(self):
        self.is_recording = True
        self.recording_frames = []
    
    def stop_recording(self):
        if self.is_recording and self.recording_frames:
            self.is_recording = False
            # Save video
            height, width = self.recording_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi',
                fourcc, 20.0, (width, height)
            )
            for frame in self.recording_frames:
                out.write(frame)
            out.release()
            self.recording_frames = []

camera_manager = CameraManager()

@app.websocket("/ws/camera")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await camera_manager.initialize()
    
    try:
        while True:
            # Receive commands from client
            command = await websocket.receive_text()
            command_data = json.loads(command)
            
            if command_data.get('action') == 'start_recording':
                camera_manager.start_recording()
            elif command_data.get('action') == 'stop_recording':
                camera_manager.stop_recording()
            elif command_data.get('action') == 'get_frame':
                frame_data = camera_manager.get_frame()
                if frame_data:
                    await websocket.send_json(frame_data)
            
            await asyncio.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"Error: {e}")
    finally:
        camera_manager.release()

@app.on_event("shutdown")
def shutdown_event():
    camera_manager.release()


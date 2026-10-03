import asyncio
import cv2
import torch
import ultralytics.nn.tasks

# PyTorch 2.6 defaults weights_only=True, which breaks YOLO. Monkey-patch the loader:
_original_load = ultralytics.nn.tasks.torch.load
def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
ultralytics.nn.tasks.torch.load = _safe_load

from ultralytics import YOLO

import numpy as np
import torchvision.transforms as T
from pathlib import Path
from reiduq.models.builder import build_encoder
from reiduq.retrieval.index import GalleryIndex, GalleryEntry
from reiduq.search.gallery_store import GalleryStore, GalleryEntryRow
from reiduq.core.settings import settings
from fastapi import WebSocket
from collections import defaultdict
import time
import base64

class LiveProcessor:
    def __init__(self, camera_id: str, stream_url: str):
        self.camera_id = camera_id
        self.stream_url = stream_url
        
        # 1. Detection & Tracking (YOLOv8 + BoT-SORT/ByteTrack built-in)
        # Classes: 2=car, 3=motorcycle, 5=bus, 7=truck (COCO)
        self.yolo = YOLO("yolov8n.pt")
        
        # 2. Re-ID Model (OSNet)
        self.encoder = build_encoder("osnet_ain", embed_dim=512, pretrained=True)
        self.encoder.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # 3. Vector DB & State
        self.out_dir = Path(settings.model_registry_path) / "gallery"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.out_dir / "index.faiss"
        self.store_path = self.out_dir / "store.db"
        
        self.store = GalleryStore(self.store_path)
        try:
            self.index = GalleryIndex.load(self.index_path)
        except Exception:
            self.index = GalleryIndex(dim=512)
            
        self.track_history = defaultdict(list)
        self.reid_cache = {}  # track_id -> global_id
        self.websockets = []
        self.is_running = False
        
        self.similarity_threshold = 0.70  # Cosine similarity threshold for identity matching
        self.reid_interval_frames = 10    # Run Re-ID every N frames per track
        
    async def process_loop(self):
        self.is_running = True
        cap = cv2.VideoCapture(self.stream_url)
        frame_count = 0
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"[{self.camera_id}] Stream disconnected. Reconnecting in 5s...")
                await asyncio.sleep(5)
                cap = cv2.VideoCapture(self.stream_url)
                continue
                
            frame_count += 1
            
            # Resize frame for performance if it's too large
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
            
            # Detect and track (persist=True enables tracking)
            results = self.yolo.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
            
            frame_data = {
                "camera_id": self.camera_id,
                "timestamp": time.time(),
                "vehicles": [],
                "image": None
            }
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                cls_ids = results[0].boxes.cls.cpu().numpy()
                
                for box, track_id, conf, cls_id in zip(boxes, track_ids, confs, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Run Re-ID if it's a new track, or periodically
                    if track_id not in self.reid_cache or frame_count % self.reid_interval_frames == 0:
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if crop.size > 0:
                            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            tensor = self.transform(rgb_crop).unsqueeze(0)
                            
                            with torch.no_grad():
                                vec = self.encoder(tensor)
                                vec = torch.nn.functional.normalize(vec, dim=1)
                            embedding = vec.cpu().numpy()[0].astype(np.float32)
                            
                            # Match against FAISS
                            if self.index._index.ntotal > 0:
                                dists, idxs = self.index._index.search(np.expand_dims(embedding, axis=0), 1)
                                best_dist = dists[0][0]
                                best_idx = idxs[0][0]
                                
                                # FAISS Inner Product is cosine similarity if normalized
                                if best_dist > self.similarity_threshold and best_idx < len(self.index.entries):
                                    global_id = self.index.entries[best_idx].uid
                                else:
                                    global_id = f"V{self.index._index.ntotal + 1:04d}"
                                    self._add_to_index(embedding, global_id, rgb_crop)
                            else:
                                global_id = f"V1"
                                self._add_to_index(embedding, global_id, rgb_crop)
                                
                            self.reid_cache[track_id] = global_id
                    
                    global_id = self.reid_cache.get(track_id, f"T{track_id}")
                    
                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    label = f"{global_id} ({self.yolo.names[int(cls_id)]})"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    frame_data["vehicles"].append({
                        "vehicle_id": global_id,
                        "track_id": int(track_id),
                        "type": self.yolo.names[int(cls_id)],
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            # Encode frame to JPEG for WebSocket
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_data["image"] = base64.b64encode(buffer).decode('utf-8')
            
            # Broadcast to WebSockets
            dead_ws = []
            for ws in self.websockets:
                try:
                    await ws.send_json(frame_data)
                except Exception:
                    dead_ws.append(ws)
            for ws in dead_ws:
                self.websockets.remove(ws)
                
            await asyncio.sleep(0.01) # Yield
            
        cap.release()

    def _add_to_index(self, embedding, global_id, crop):
        # Save crop
        crop_path = self.out_dir / "crops" / f"{global_id}.jpg"
        crop_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        
        # Add to FAISS
        try:
            existing_vecs = self.index._index.reconstruct_n(0, self.index._index.ntotal) if self.index._index.ntotal > 0 else np.empty((0, 512), dtype=np.float32)
            all_vecs = np.vstack([existing_vecs, embedding])
            entries = list(getattr(self.index, 'entries', []))
        except Exception:
            all_vecs = np.expand_dims(embedding, axis=0)
            entries = []
            
        new_entry = GalleryEntry(uid=global_id, identity=0, camera_id=self.camera_id)
        entries.append(new_entry)
        
        new_index = GalleryIndex(dim=512)
        new_index.add(all_vecs, entries)
        new_index.build()
        new_index.save(self.index_path)
        self.index = new_index
        
        # Add to SQLite Store
        self.store.add(GalleryEntryRow(
            faiss_position=len(entries) - 1,
            uid=global_id,
            image_path=str(crop_path),
            dataset="live_stream",
            camera_id=self.camera_id,
            visibility=1.0
        ))

    async def add_client(self, websocket: WebSocket):
        await websocket.accept()
        self.websockets.append(websocket)
        
    def stop(self):
        self.is_running = False

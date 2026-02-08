"""
VIGIL V6.0 - Vehicle-Installed Guard for Injury Limitation
Complete Professional Web Dashboard with Modern Web Interface
Features: Multi-Camera Grid, Zone Management, Real-time Analytics, Settings
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from collections import deque, defaultdict
import asyncio
import io
import base64

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Try to import PDF generation library
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger_temp = logging.getLogger('VIGIL_WEB')
    logger_temp.warning("reportlab not installed - PDF reports disabled. Install with: pip install reportlab")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VIGIL_WEB')

# FastAPI app
app = FastAPI(
    title="VIGIL V6 Professional",
    description="Vehicle-Installed Guard for Injury Limitation - Complete Dashboard",
    version="6.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
cameras = {}
detection_engine = None
camera_stats = {}
system_stats = {
    'start_time': None,
    'total_detections': 0,
    'total_violations': 0,
    'peak_detections': 0,
    'uptime_seconds': 0,
    'system_enabled': True
}
detection_history = defaultdict(lambda: deque(maxlen=60))
fps_history = defaultdict(lambda: deque(maxlen=30))
event_log = deque(maxlen=100)
camera_zones = defaultdict(list)  # Zones per camera
camera_settings = {}
direction_stats = defaultdict(lambda: defaultdict(int))  # Direction tracking per camera
tracking_objects = defaultdict(dict)  # Track objects across frames for direction

# Frame caching for multi-client support
camera_frames = {}  # Stores latest processed frame per camera
frame_ready_events = {}  # Events to signal new frames

# Zone and violation persistence
ZONES_FILE = Path(__file__).parent / "zones.json"
VIOLATIONS_FILE = Path(__file__).parent / "violations.json"
violations_log = deque(maxlen=500)  # Store last 500 violations
violation_history = defaultdict(lambda: deque(maxlen=60))  # Violations per minute for graphing

# Recording configuration
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)
camera_recorders = {}  # VideoWriter objects per camera
recording_info = {}  # Recording metadata per camera

# Tamper detection configuration
TAMPER_DETECTION_CONFIG = {
    'enabled': True,
    'darkness_threshold': 25,  # Average brightness below this = blocked
    'static_threshold': 0.98,  # Frame similarity above this = static/covered
    'disconnect_timeout': 5,  # Seconds without frame = disconnected
    'warning_delay': 10,  # Seconds before triggering warning
    'check_interval': 1.0,  # How often to check for tampering
}

# Tamper detection state per camera
tamper_state = defaultdict(lambda: {
    'last_frame': None,
    'last_frame_time': None,
    'darkness_start': None,
    'static_start': None,
    'disconnect_start': None,
    'is_tampered': False,
    'tamper_type': None,
    'warning_sent': False,
    'last_check_time': None,
})

# Email configuration (user should update these)
EMAIL_CONFIG = {
    'enabled': False,  # Set to True to enable email alerts
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',
    'recipient_emails': ['alert@example.com']
}

# Audio alarm configuration
AUDIO_ALARM_CONFIG = {
    'enabled': True,
    'volume': 0.7,  # 0.0 to 1.0
    'sound_type': 'alert',  # 'alert', 'siren', 'beep'
    'repeat_interval': 5,  # seconds between repeated alarms
    'cooldown': 10,  # seconds before same zone can trigger again
}







# Report configuration
REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class CameraConfig(BaseModel):
    id: int
    name: str
    source: int
    enabled: bool = True
    recording: bool = False


class Zone(BaseModel):
    id: str
    camera_id: int
    name: str
    type: str  # restricted, warning, safe
    points: List[List[float]]
    active: bool = True


class DetectionEvent(BaseModel):
    timestamp: str
    camera_id: int
    event_type: str
    description: str
    confidence: float


class Settings(BaseModel):
    detection_confidence: float = 0.5
    zone_transparency: float = 0.3
    alert_threshold: int = 3
    recording_enabled: bool = False
    email_alerts: bool = False


def load_zones():
    """Load zones from JSON file"""
    global camera_zones
    try:
        if ZONES_FILE.exists():
            with open(ZONES_FILE, 'r') as f:
                data = json.load(f)
                camera_zones = defaultdict(list, {int(k): v for k, v in data.items()})
                logger.info(f"Loaded {sum(len(v) for v in camera_zones.values())} zones from {ZONES_FILE}")
        else:
            logger.info("No zones file found, starting with empty zones")
    except Exception as e:
        logger.error(f"Failed to load zones: {e}")


def save_zones():
    """Save zones to JSON file"""
    try:
        with open(ZONES_FILE, 'w') as f:
            json.dump(dict(camera_zones), f, indent=2)
        logger.info(f"Saved zones to {ZONES_FILE}")
    except Exception as e:
        logger.error(f"Failed to save zones: {e}")


def load_violations():
    """Load violations from JSON file"""
    global violations_log
    try:
        if VIOLATIONS_FILE.exists():
            with open(VIOLATIONS_FILE, 'r') as f:
                data = json.load(f)
                violations_log = deque(data, maxlen=500)
                logger.info(f"Loaded {len(violations_log)} violations from {VIOLATIONS_FILE}")
        else:
            logger.info("No violations file found, starting fresh")
    except Exception as e:
        logger.error(f"Failed to load violations: {e}")


def save_violations():
    """Save violations to JSON file"""
    try:
        with open(VIOLATIONS_FILE, 'w') as f:
            json.dump(list(violations_log), f, indent=2)
        logger.info(f"Saved {len(violations_log)} violations to {VIOLATIONS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save violations: {e}")


# ============== RECORDING SYSTEM ==============

def start_recording(camera_id: int):
    """Start recording for a specific camera"""
    global camera_recorders, recording_info
    
    if camera_id in camera_recorders and camera_recorders[camera_id] is not None:
        logger.warning(f"Camera {camera_id} is already recording")
        return False
    
    try:
        # Create filename with timestamp
        start_time = datetime.now()
        filename = f"camera_{camera_id}_{start_time.strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = RECORDINGS_DIR / filename
        
        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0  # Recording FPS
        frame_size = (480, 360)  # Match processed frame size
        
        # Create VideoWriter
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, frame_size)
        
        if not writer.isOpened():
            logger.error(f"Failed to create video writer for camera {camera_id}")
            return False
        
        camera_recorders[camera_id] = writer
        recording_info[camera_id] = {
            'start_time': start_time,
            'filename': filename,
            'filepath': str(filepath),
            'frame_count': 0
        }
        
        logger.info(f"Started recording camera {camera_id} to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to start recording for camera {camera_id}: {e}")
        return False


def stop_recording(camera_id: int):
    """Stop recording for a specific camera"""
    global camera_recorders, recording_info
    
    if camera_id not in camera_recorders or camera_recorders[camera_id] is None:
        logger.warning(f"Camera {camera_id} is not recording")
        return None
    
    try:
        writer = camera_recorders[camera_id]
        writer.release()
        
        info = recording_info.get(camera_id, {})
        end_time = datetime.now()
        start_time = info.get('start_time', end_time)
        duration = (end_time - start_time).total_seconds()
        
        # Rename file to include end timestamp
        old_filepath = Path(info.get('filepath', ''))
        if old_filepath.exists():
            new_filename = f"camera_{camera_id}_{start_time.strftime('%Y%m%d_%H%M%S')}_to_{end_time.strftime('%H%M%S')}.mp4"
            new_filepath = RECORDINGS_DIR / new_filename
            old_filepath.rename(new_filepath)
            info['filepath'] = str(new_filepath)
            info['filename'] = new_filename
        
        info['end_time'] = end_time
        info['duration'] = duration
        
        camera_recorders[camera_id] = None
        
        logger.info(f"Stopped recording camera {camera_id}. Duration: {duration:.1f}s, Frames: {info.get('frame_count', 0)}")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to stop recording for camera {camera_id}: {e}")
        camera_recorders[camera_id] = None
        return None


def write_frame_to_recording(camera_id: int, frame):
    """Write a frame to the recording if active"""
    global camera_recorders, recording_info
    
    if camera_id not in camera_recorders or camera_recorders[camera_id] is None:
        return
    
    try:
        writer = camera_recorders[camera_id]
        if writer is not None and writer.isOpened():
            # Ensure frame is correct size
            if frame.shape[:2] != (360, 480):
                frame = cv2.resize(frame, (480, 360))
            writer.write(frame)
            recording_info[camera_id]['frame_count'] = recording_info[camera_id].get('frame_count', 0) + 1
    except Exception as e:
        logger.error(f"Error writing frame to recording: {e}")


# ============== TAMPER DETECTION SYSTEM ==============

def check_tamper_detection(camera_id: int, frame, ret: bool):
    """
    Check for camera tampering: blocked/covered camera or disconnection.
    Returns tuple: (is_tampered, tamper_type, message)
    """
    global tamper_state, violations_log
    
    if not TAMPER_DETECTION_CONFIG['enabled']:
        return False, None, None
    
    current_time = time.time()
    state = tamper_state[camera_id]
    
    # Check interval to avoid excessive processing
    check_interval = TAMPER_DETECTION_CONFIG.get('check_interval', 1.0)
    if state['last_check_time'] and (current_time - state['last_check_time']) < check_interval:
        return state['is_tampered'], state['tamper_type'], None
    
    state['last_check_time'] = current_time
    warning_delay = TAMPER_DETECTION_CONFIG.get('warning_delay', 10)
    
    # Check 1: Camera disconnected (no frame received)
    if not ret or frame is None:
        if state['disconnect_start'] is None:
            state['disconnect_start'] = current_time
            logger.warning(f"Camera {camera_id}: No frame detected, monitoring for disconnect...")
        
        disconnect_duration = current_time - state['disconnect_start']
        
        if disconnect_duration >= warning_delay and not state['warning_sent']:
            state['is_tampered'] = True
            state['tamper_type'] = 'DISCONNECTED'
            state['warning_sent'] = True
            
            # Log tamper event
            log_tamper_event(camera_id, 'DISCONNECTED', f"Camera disconnected for {disconnect_duration:.0f}s")
            
            return True, 'DISCONNECTED', f"⚠️ TAMPER ALERT: Camera {camera_id} DISCONNECTED for {disconnect_duration:.0f}s"
        
        return state['is_tampered'], state['tamper_type'], None
    else:
        # Frame received, reset disconnect timer
        state['disconnect_start'] = None
    
    # Check 2: Camera blocked/covered (darkness detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    darkness_threshold = TAMPER_DETECTION_CONFIG.get('darkness_threshold', 25)
    
    if avg_brightness < darkness_threshold:
        if state['darkness_start'] is None:
            state['darkness_start'] = current_time
            logger.warning(f"Camera {camera_id}: Low brightness detected ({avg_brightness:.1f}), monitoring...")
        
        darkness_duration = current_time - state['darkness_start']
        
        if darkness_duration >= warning_delay and not state['warning_sent']:
            state['is_tampered'] = True
            state['tamper_type'] = 'BLOCKED'
            state['warning_sent'] = True
            
            # Log tamper event
            log_tamper_event(camera_id, 'BLOCKED', f"Camera appears blocked/covered (brightness: {avg_brightness:.1f})")
            
            return True, 'BLOCKED', f"⚠️ TAMPER ALERT: Camera {camera_id} BLOCKED/COVERED for {darkness_duration:.0f}s"
        
        return state['is_tampered'], state['tamper_type'], None
    else:
        # Brightness normal, reset darkness timer
        state['darkness_start'] = None
    
    # Static image detection DISABLED - cameras may legitimately show static scenes
    # Only detect blocked (darkness) and disconnected cameras
    
    # If we reach here, no tampering detected - reset state if was previously tampered
    if state['is_tampered']:
        logger.info(f"Camera {camera_id}: Tamper condition cleared")
        state['is_tampered'] = False
        state['tamper_type'] = None
        state['warning_sent'] = False
    
    return False, None, None


def log_tamper_event(camera_id: int, tamper_type: str, description: str):
    """Log a tamper event to violations"""
    global violations_log, event_log
    
    timestamp = datetime.now()
    
    tamper_event = {
        'timestamp': timestamp.isoformat(),
        'camera_id': camera_id,
        'camera_name': cameras.get(camera_id, {}).get('position', f'Camera {camera_id}'),
        'type': 'TAMPER',
        'tamper_type': tamper_type,
        'description': description,
        'severity': 'HIGH'
    }
    
    violations_log.append(tamper_event)
    event_log.append({
        'timestamp': timestamp.isoformat(),
        'camera_id': camera_id,
        'event_type': 'tamper_alert',
        'description': f"TAMPER: {tamper_type} - {description}",
        'confidence': 1.0
    })
    
    # Save violations
    save_violations()
    
    logger.warning(f"TAMPER EVENT logged: Camera {camera_id} - {tamper_type}: {description}")


def draw_tamper_warning(frame, camera_id):
    """Draw tamper warning overlay on frame"""
    state = tamper_state[camera_id]
    
    if not state['is_tampered']:
        return frame
    
    h, w = frame.shape[:2]
    
    # Red border
    cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 8)
    
    # Warning overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h//3), (w, 2*h//3), (0, 0, 200), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Warning text
    tamper_type = state['tamper_type'] or 'TAMPERED'
    warning_text = f"⚠️ CAMERA {tamper_type} ⚠️"
    text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h // 2
    
    cv2.putText(frame, warning_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Blinking effect using timestamp
    if int(time.time() * 2) % 2 == 0:
        cv2.putText(frame, "CHECK CAMERA IMMEDIATELY", (w//4 - 40, text_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return frame


@app.on_event("startup")
async def startup_event():
    """Initialize cameras and detection engine on startup"""
    global detection_engine, system_stats
    
    logger.info("="*70)
    logger.info("VIGIL V6.0 Professional - Complete System Initialization")
    logger.info("="*70)
    
    system_stats['start_time'] = datetime.now()
    
    # Load persisted data
    load_zones()
    load_violations()
    
    # Initialize YOLO detection engine
    if YOLO_AVAILABLE:
        try:
            detection_engine = YOLO('yolov8n.pt')
            logger.info("YOLO detection engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO: {e}")
            detection_engine = None
    else:
        logger.warning("YOLO not available - running without AI detection")
    
    # Initialize cameras
    initialize_cameras()
    
    logger.info("System ready - Complete Dashboard accessible")
    logger.info("="*70)


def camera_capture_thread(camera_id):
    """Background thread to continuously capture and process frames"""
    global cameras, camera_frames, frame_ready_events, camera_stats, fps_history
    
    logger.info(f"Starting capture thread for Camera {camera_id}")
    
    fps_start = time.time()
    frame_count = 0
    
    while cameras.get(camera_id, {}).get('running', False):
        try:
            camera = cameras[camera_id]
            cap = camera['capture']
            
            ret, frame = cap.read()
            
            # Check for tampering (including disconnection)
            is_tampered, tamper_type, tamper_msg = check_tamper_detection(camera_id, frame, ret)
            
            if ret and frame is not None:
                # Resize for performance
                frame = cv2.resize(frame, (480, 360))
                
                # Process frame with detection
                processed_frame, detections, violations = process_frame_with_detection(frame, camera_id)
                
                if processed_frame is not None:
                    # Draw tamper warning if tampered
                    if is_tampered:
                        processed_frame = draw_tamper_warning(processed_frame, camera_id)
                    
                    # Write to recording if active
                    if cameras[camera_id].get('recording', False):
                        write_frame_to_recording(camera_id, processed_frame)
                    
                    # Encode frame to JPEG
                    encode_ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if encode_ret:
                        camera_frames[camera_id] = buffer.tobytes()
                        frame_ready_events[camera_id].set()
                
                # Calculate FPS every 20 frames
                frame_count += 1
                if frame_count % 20 == 0:
                    fps_end = time.time()
                    fps = 20 / (fps_end - fps_start)
                    camera_stats[camera_id]['fps'] = fps
                    fps_history[camera_id].append(fps)
                    fps_start = time.time()
                    
            else:
                # No frame - create error frame with tamper warning
                error_frame = np.zeros((360, 480, 3), dtype=np.uint8)
                if is_tampered:
                    error_frame = draw_tamper_warning(error_frame, camera_id)
                else:
                    cv2.putText(error_frame, 'No Feed', (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                encode_ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                camera_frames[camera_id] = buffer.tobytes()
                frame_ready_events[camera_id].set()
                time.sleep(0.5)
                
            time.sleep(0.02)  # ~30 FPS target
            
        except Exception as e:
            logger.error(f"Camera {camera_id} capture error: {e}")
            time.sleep(1)
    
    logger.info(f"Capture thread for Camera {camera_id} stopped")


def initialize_cameras():
    """Initialize camera feeds with smart detection"""
    global cameras, camera_stats
    
    camera_positions = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']
    
    # Detect working cameras
    available_cameras = []
    logger.info("Scanning for available camera devices...")
    
    # Check devices 0-9 to find all cameras (primary video nodes vary based on USB enumeration)
    for device_idx in range(10):
        try:
            cap = cv2.VideoCapture(device_idx)
            if cap.isOpened():
                # Set camera properties for better compatibility
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if camera actually works (read a frame)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    available_cameras.append({'idx': device_idx, 'cap': cap})
                    logger.info(f"Found working camera at device index {device_idx}")
                    
                    # Stop at 4 cameras
                    if len(available_cameras) >= 4:
                        break
                else:
                    cap.release()
        except Exception as e:
            pass
    
    logger.info(f"Detected {len(available_cameras)} working camera(s): {[c['idx'] for c in available_cameras]}")
    
    # Initialize cameras with background frame capture
    for logical_id in range(min(4, len(available_cameras))):
        cam_info = available_cameras[logical_id]
        device_idx = cam_info['idx']
        cap = cam_info['cap']
        try:
            if cap.isOpened():
                cameras[logical_id] = {
                    'capture': cap,
                    'device_index': device_idx,
                    'name': f'Camera {logical_id}',
                    'position': camera_positions[logical_id],
                    'active': True,
                    'recording': False,
                    'last_frame': None,
                    'frame_lock': threading.Lock(),
                    'running': True
                }
                camera_stats[logical_id] = {
                    'detections': 0,
                    'violations': 0,
                    'fps': 0,
                    'total_detections': 0,
                    'last_detection_time': None,
                    'status': 'active',
                    'resolution': '640x480'
                }
                camera_settings[logical_id] = {
                    'brightness': 128,
                    'contrast': 128,
                    'saturation': 128,
                    'enabled': True,
                    'detection_enabled': True
                }
                camera_frames[logical_id] = None
                frame_ready_events[logical_id] = threading.Event()
                
                # Start background frame capture thread
                thread = threading.Thread(target=camera_capture_thread, args=(logical_id,), daemon=True)
                thread.start()
                
                logger.info(f"Camera {logical_id} ({cameras[logical_id]['position']}) mapped to device {device_idx}")
        except Exception as e:
            logger.error(f"Camera {logical_id} initialization error: {e}")


def point_in_rect(px, py, rect):
    """Check if point is inside rectangle defined by [x, y, width, height]"""
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def check_zone_violation(camera_id, person_x, person_y, person_box):
    """Check if person is violating any restricted zone"""
    global camera_zones, violations_log, camera_stats, esp32_buzzer_state
    
    zones = camera_zones.get(camera_id, [])
    
    for zone in zones:
        if not zone.get('active', True):
            continue
        
        zone_type = zone.get('type', 'restricted')
        zone_rect = zone.get('rect', [])  # [x, y, width, height]
        
        if len(zone_rect) == 4:
            # Check if person center is in zone
            if point_in_rect(person_x, person_y, zone_rect):
                zone_name = zone.get('name', 'Unknown Zone')
                zone_id = zone.get('id', f"{camera_id}_{zone_name}")
                violation_key = f"{camera_id}_{zone_id}"
                
                # Handle different zone types
                if zone_type == 'restricted':
                    # Track this active violation
                    esp32_buzzer_state['current_violations'].add(violation_key)
                    
                    # Log violation
                    violation = {
                        'timestamp': datetime.now().isoformat(),
                        'camera_id': camera_id,
                        'camera_name': cameras.get(camera_id, {}).get('name', f'Camera {camera_id}'),
                        'zone_id': zone.get('id'),
                        'zone_name': zone_name,
                        'zone_type': zone_type,
                        'person_box': person_box
                    }
                    violations_log.append(violation)
                    
                    # Update stats
                    camera_stats[camera_id]['violations'] = camera_stats[camera_id].get('violations', 0) + 1
                    system_stats['total_violations'] += 1
                    
                    # Trigger ESP32 buzzer based on settings
                    if ESP32_BUZZER_CONFIG['enabled'] and ESP32_BUZZER_CONFIG['connected']:
                        sound = ESP32_BUZZER_SETTINGS.get('restricted_sound', 'continuous').upper()
                        if sound == 'CONTINUOUS':
                            trigger_esp32_buzzer('ALERT')  # Continuous alarm
                        elif sound == 'SIREN':
                            trigger_esp32_buzzer('SIREN')  # Siren sound
                        else:
                            trigger_esp32_buzzer('BEEP')  # Beep sound
                    
                    # Save violations periodically (every 10 violations)
                    if len(violations_log) % 10 == 0:
                        save_violations()
                    
                    return {'zone_name': zone_name, 'zone_type': zone_type, 'violation': violation}
                
                elif zone_type == 'warning':
                    # Track warning zone violations too
                    esp32_buzzer_state['current_violations'].add(violation_key)
                    
                    # Trigger ESP32 buzzer based on settings
                    if ESP32_BUZZER_CONFIG['enabled'] and ESP32_BUZZER_CONFIG['connected']:
                        sound = ESP32_BUZZER_SETTINGS.get('warning_sound', 'beep').upper()
                        if sound == 'CONTINUOUS':
                            trigger_esp32_buzzer('WARNING')  # Continuous warning
                        elif sound == 'SIREN':
                            trigger_esp32_buzzer('SIREN')  # Siren sound
                        else:
                            trigger_esp32_buzzer('BEEP')  # Beep sound
                    # Just return warning, don't log as violation
                    return {'zone_name': zone_name, 'zone_type': 'warning', 'alert_only': True}
    
    return None


def draw_zones_on_frame(frame, camera_id):
    """Draw all active zones on the camera frame"""
    zones = camera_zones.get(camera_id, [])
    
    for zone in zones:
        if not zone.get('active', True):
            continue
        
        zone_type = zone.get('type', 'restricted')
        zone_rect = zone.get('rect', [])  # [x, y, width, height]
        zone_name = zone.get('name', 'Zone')
        
        if len(zone_rect) == 4:
            x, y, w, h = zone_rect
            
            # Color based on type
            if zone_type == 'restricted':
                color = (0, 0, 255)  # Red
                label_bg = (0, 0, 200)
            elif zone_type == 'warning':
                color = (0, 255, 255)  # Yellow
                label_bg = (0, 200, 200)
            else:  # safe
                color = (0, 255, 0)  # Green
                label_bg = (0, 200, 0)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw semi-transparent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            
            # Draw label
            label = f"{zone_name} ({zone_type})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x, y-label_h-8), (x+label_w+8, y), label_bg, -1)
            cv2.putText(frame, label, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# Global barrier detection settings - IMPROVED to reduce false positives
barrier_settings = {
    'enabled': True,
    'min_area': 5000,  # Minimum barrier area in pixels (increased to filter small objects)
    'max_area': 300000,  # Maximum barrier area
    'min_width': 120,  # Minimum width (barriers are typically wide)
    'min_height': 50,  # Minimum height (increased)
    'aspect_ratio_min': 1.5,  # Min aspect ratio - barriers are wider than tall (filters jackets)
    'aspect_ratio_max': 8.0,  # Max aspect ratio (not too wide/thin)
    'min_color_coverage': 0.35,  # Minimum color coverage (35% - more strict)
    'min_saturation': 150,  # Higher saturation for industrial colors (filters clothes)
    'min_value': 120,  # Higher brightness threshold
    'edge_density_min': 0.03,  # Minimum edge density (barriers have defined structure)
    'max_in_upper_frame': 0.25,  # Reject detections in top 25% of frame
    'require_stripe_pattern': True,  # Prefer striped barriers
    'min_stripe_alternation': 3,  # Minimum color alternations for stripe detection
    'show_debug': False  # Show color detection overlay
}


def detect_industrial_barriers(frame):
    """
    Detect industrial safety barriers (red and yellow striped) using advanced color analysis.
    IMPROVED algorithm to significantly reduce false positives (jackets, clothing, etc).
    Returns list of barrier bounding boxes.
    """
    if frame is None or not barrier_settings['enabled']:
        return []
    
    barriers = []
    h, w = frame.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Also get grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Define STRICT color ranges for industrial barriers
    # Red color - very specific for industrial red (not clothing red)
    min_sat = barrier_settings['min_saturation']
    min_val = barrier_settings['min_value']
    
    # Industrial red is very saturated and bright
    lower_red1 = np.array([0, min_sat, min_val])
    upper_red1 = np.array([6, 255, 255])  # Very narrow hue range
    lower_red2 = np.array([174, min_sat, min_val])  # Very narrow hue range
    upper_red2 = np.array([180, 255, 255])
    
    # Industrial yellow - bright warning yellow only
    lower_yellow = np.array([22, min_sat + 40, min_val + 60])  # Very saturated yellow
    upper_yellow = np.array([32, 255, 255])
    
    # Industrial orange - traffic cone orange
    lower_orange = np.array([12, min_sat + 30, min_val + 40])
    upper_orange = np.array([22, 255, 255])
    
    # Create masks for each color
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Combine red masks
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    # Combine all warning color masks
    mask_combined = cv2.bitwise_or(mask_red, mask_yellow)
    mask_combined = cv2.bitwise_or(mask_combined, mask_orange)
    
    # Morphological operations - aggressive noise removal
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_medium = np.ones((7, 7), np.uint8)
    
    # Remove noise first - more aggressive
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel_small, iterations=3)
    
    # Close small gaps
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Find contours - use original mask, not dilated (more precise)
    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area - strict
        if area < barrier_settings['min_area'] or area > barrier_settings['max_area']:
            continue
        
        # Get bounding rectangle
        x, y, bw, bh = cv2.boundingRect(contour)
        
        # Filter by size
        if bw < barrier_settings['min_width'] or bh < barrier_settings['min_height']:
            continue
        
        # Filter by aspect ratio - barriers are WIDER than tall (important to filter jackets)
        aspect_ratio = bw / bh if bh > 0 else 0
        if aspect_ratio < barrier_settings['aspect_ratio_min'] or aspect_ratio > barrier_settings['aspect_ratio_max']:
            continue
        
        # VALIDATION 1: Check contour solidity (barriers are relatively solid shapes)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.4:  # Increased threshold - too fragmented likely not a barrier
            continue
        
        # VALIDATION 2: Check color distribution in the ROI
        roi_red = mask_red[y:y+bh, x:x+bw]
        roi_yellow = mask_yellow[y:y+bh, x:x+bw]
        roi_orange = mask_orange[y:y+bh, x:x+bw]
        
        red_pixels = cv2.countNonZero(roi_red)
        yellow_pixels = cv2.countNonZero(roi_yellow)
        orange_pixels = cv2.countNonZero(roi_orange)
        total_colored = red_pixels + yellow_pixels + orange_pixels
        roi_area = bw * bh
        
        # Calculate color coverage percentage
        color_coverage = total_colored / roi_area if roi_area > 0 else 0
        
        # Minimum coverage threshold - stricter
        if color_coverage < barrier_settings['min_color_coverage']:
            continue
        
        # VALIDATION 3: Check edge density (barriers have defined edges/structure)
        roi_edges = edges[y:y+bh, x:x+bw]
        edge_pixels = cv2.countNonZero(roi_edges)
        edge_density = edge_pixels / roi_area if roi_area > 0 else 0
        
        if edge_density < barrier_settings['edge_density_min']:
            continue  # Too smooth, might be a colored surface not a barrier
        
        # VALIDATION 4: Check for rectangular shape (barriers are usually rectangular)
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        if rectangularity < 0.5:  # Increased - must be more rectangular
            continue
        
        # VALIDATION 5: Position check - barriers are typically in lower portion of frame
        center_y = y + bh / 2
        max_upper = barrier_settings.get('max_in_upper_frame', 0.25)
        if center_y < h * max_upper:  # Too high in frame, likely not a barrier
            continue
        
        # VALIDATION 6: Check for stripe pattern (alternating colors)
        has_stripes = False
        stripe_score = 0
        if bw > 50:  # Only check for stripes if wide enough
            # Sample horizontal line through middle of ROI
            mid_y = bh // 2
            if mid_y < roi_red.shape[0]:
                red_line = roi_red[mid_y, :]
                yellow_line = roi_yellow[mid_y, :]
                orange_line = roi_orange[mid_y, :]
                
                # Count color transitions (stripe pattern indicator)
                combined_line = np.maximum(red_line, np.maximum(yellow_line, orange_line))
                transitions = np.sum(np.abs(np.diff(combined_line > 0)))
                min_alternations = barrier_settings.get('min_stripe_alternation', 3)
                
                if transitions >= min_alternations:
                    has_stripes = True
                    stripe_score = min(1.0, transitions / 10)
        
        # VALIDATION 7: Reject if detection overlaps with person-shaped regions
        # Jackets are typically tall and narrow, barriers are wide and short
        if bh > bw * 0.8:  # If height is more than 80% of width, likely not a barrier
            continue
        
        # Determine barrier type with stricter criteria
        is_striped = has_stripes or ((red_pixels > 500 and yellow_pixels > 500) or \
                     (red_pixels > 500 and orange_pixels > 500))
        
        # Calculate confidence with multiple factors
        if is_striped:
            barrier_type = "STRIPED"
            confidence = min(1.0, (color_coverage * 1.5 + rectangularity * 0.3 + stripe_score * 0.5 + solidity * 0.2))
        elif red_pixels > max(yellow_pixels, orange_pixels) * 1.5:  # Must be significantly more red
            barrier_type = "RED"
            confidence = min(1.0, (red_pixels / roi_area) * 2.5 + rectangularity * 0.3)
        elif orange_pixels > yellow_pixels * 1.5:
            barrier_type = "ORANGE"
            confidence = min(1.0, (orange_pixels / roi_area) * 2.5 + rectangularity * 0.3)
        else:
            barrier_type = "YELLOW"
            confidence = min(1.0, (yellow_pixels / roi_area) * 2.5 + rectangularity * 0.3)
        
        # Only accept high-confidence detections - stricter threshold
        if confidence < 0.45:
            continue
        
        # Prefer striped barriers if configured
        if barrier_settings.get('require_stripe_pattern', False) and not is_striped:
            if confidence < 0.6:  # Non-striped needs higher confidence
                continue
        
        barriers.append({
            'box': (x, y, x + bw, y + bh),
            'area': area,
            'type': barrier_type,
            'coverage': color_coverage,
            'confidence': confidence,
            'is_striped': is_striped,
            'solidity': solidity,
            'rectangularity': rectangularity,
            'has_stripes': has_stripes
        })
    
    # Remove overlapping detections (keep highest confidence)
    barriers = remove_overlapping_barriers(barriers)
    
    return barriers


def remove_overlapping_barriers(barriers):
    """Remove overlapping barrier detections, keeping the highest confidence ones."""
    if len(barriers) <= 1:
        return barriers
    
    # Sort by confidence (highest first)
    barriers = sorted(barriers, key=lambda x: x.get('confidence', 0), reverse=True)
    
    filtered = []
    for barrier in barriers:
        bx1, by1, bx2, by2 = barrier['box']
        
        is_overlapping = False
        for existing in filtered:
            ex1, ey1, ex2, ey2 = existing['box']
            
            # Calculate intersection
            ix1 = max(bx1, ex1)
            iy1 = max(by1, ey1)
            ix2 = min(bx2, ex2)
            iy2 = min(by2, ey2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                barrier_area = (bx2 - bx1) * (by2 - by1)
                overlap_ratio = intersection / barrier_area if barrier_area > 0 else 0
                
                if overlap_ratio > 0.5:  # More than 50% overlap
                    is_overlapping = True
                    break
        
        if not is_overlapping:
            filtered.append(barrier)
    
    return filtered


def is_person_behind_barrier(person_box, barriers, frame_height):
    """
    Determine if a person is behind (occluded by) any detected barrier.
    Uses spatial relationship analysis.
    """
    px1, py1, px2, py2 = person_box
    person_center_x = (px1 + px2) / 2
    person_center_y = (py1 + py2) / 2
    person_bottom_y = py2
    person_height = py2 - py1
    
    for barrier in barriers:
        bx1, by1, bx2, by2 = barrier['box']
        barrier_center_y = (by1 + by2) / 2
        barrier_height = by2 - by1
        
        # Check horizontal overlap
        horizontal_overlap = not (px2 < bx1 or px1 > bx2)
        
        if not horizontal_overlap:
            continue
        
        # Calculate overlap amount
        overlap_left = max(px1, bx1)
        overlap_right = min(px2, bx2)
        overlap_width = overlap_right - overlap_left
        person_width = px2 - px1
        overlap_ratio = overlap_width / person_width if person_width > 0 else 0
        
        # Person is considered behind barrier if:
        # 1. Significant horizontal overlap (>30%)
        # 2. Person's bottom is at or above barrier center (they're "behind" it)
        # 3. Or person appears to be physically behind the barrier plane
        
        if overlap_ratio > 0.3:
            # Check vertical relationship
            # If person bottom is above barrier center + some tolerance
            vertical_tolerance = barrier_height * 0.5
            
            if person_bottom_y <= barrier_center_y + vertical_tolerance:
                return True, barrier
            
            # Alternative: if person is significantly smaller (further away) 
            # and overlaps with barrier
            expected_near_height = frame_height * 0.4  # Expected height if close
            if person_height < expected_near_height * 0.5 and overlap_ratio > 0.5:
                return True, barrier
    
    return False, None


# Track violations per frame cycle for buzzer control
current_frame_violations = set()

def process_frame_with_detection(frame, camera_id):
    """Process frame with YOLO detection, barrier detection, and filtering - OPTIMIZED"""
    global detection_engine, camera_stats, detection_history, system_stats, event_log
    global current_frame_violations, esp32_buzzer_state
    
    if frame is None:
        return None, 0, 0
    
    h, w = frame.shape[:2]
    
    # Draw zones first (so they appear behind detections)
    draw_zones_on_frame(frame, camera_id)
    
    detections = 0
    violations = 0
    barriers_detected = 0
    
    try:
        # STEP 1: Detect industrial barriers using color analysis (red/yellow)
        color_barriers = detect_industrial_barriers(frame)
        
        # Draw detected color-based barriers
        for barrier in color_barriers:
            bx1, by1, bx2, by2 = barrier['box']
            barrier_type = barrier['type']
            barriers_detected += 1
            
            # Color based on barrier type
            if barrier_type == "STRIPED":
                # Orange color for striped red/yellow barriers
                color = (0, 165, 255)  # Orange in BGR
                label = "⚠ BARRIER (STRIPED)"
            elif barrier_type == "RED":
                color = (0, 0, 255)  # Red in BGR
                label = "⚠ BARRIER (RED)"
            else:  # YELLOW
                color = (0, 255, 255)  # Yellow in BGR
                label = "⚠ BARRIER (YELLOW)"
            
            # Draw barrier box with thick border
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)
            
            # Draw diagonal stripes pattern to indicate barrier
            stripe_spacing = 15
            for i in range(bx1, bx2, stripe_spacing):
                cv2.line(frame, (i, by1), (min(i + (by2 - by1), bx2), by2), color, 1)
            
            # Label with background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (bx1, by1 - label_h - 8), (bx1 + label_w + 8, by1), color, -1)
            cv2.putText(frame, label, (bx1 + 4, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if YOLO_AVAILABLE and detection_engine and camera_settings.get(camera_id, {}).get('detection_enabled', True):
            # Run YOLO detection with optimizations
            results = detection_engine(frame, conf=0.5, verbose=False, imgsz=320, device='cpu', half=False)
            
            # Collect YOLO-detected barriers (fence, wall, etc.) as backup
            yolo_barriers = []
            person_boxes = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # YOLO barrier classes (fence, wall, etc.)
                    if cls in [88, 89, 90]:
                        yolo_barriers.append({
                            'box': (x1, y1, x2, y2),
                            'type': 'YOLO',
                            'conf': conf
                        })
                        # Draw YOLO barriers in blue (lower priority than color-detected)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
                        cv2.putText(frame, 'FENCE/BARRIER', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
                        barriers_detected += 1
                    
                    # Collect persons
                    elif cls == 0:  # Person class
                        person_boxes.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf
                        })
            
            # Combine all barriers (color-detected + YOLO-detected)
            all_barriers = color_barriers + yolo_barriers
            
            # Process each detected person
            for person in person_boxes:
                px1, py1, px2, py2 = person['box']
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                
                # Check if person is behind any barrier using enhanced detection
                is_behind, blocking_barrier = is_person_behind_barrier(person['box'], all_barriers, h)
                
                if not is_behind:
                    # Count and draw visible persons
                    detections += 1
                    
                    # Check if person is in any zone
                    zone_violation = check_zone_violation(camera_id, person_center_x, person_center_y, person['box'])
                    
                    if zone_violation:
                        violations += 1
                        # Draw in RED for violation
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
                        cv2.putText(frame, f'VIOLATION: {zone_violation["zone_name"]}', (px1, py1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
                    else:
                        # Normal detection - green box
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {person["conf"]:.2f}', (px1, py1-5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    # Draw ignored persons in gray with reason
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (128, 128, 128), 1)
                    barrier_type = blocking_barrier.get('type', 'BARRIER') if blocking_barrier else 'BARRIER'
                    cv2.putText(frame, f'BEHIND {barrier_type}', (px1, py1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)
            
            # Update stats (throttled)
            camera_stats[camera_id]['detections'] = detections
            camera_stats[camera_id]['total_detections'] += detections
            camera_stats[camera_id]['barriers'] = barriers_detected
            
            # Update history every 2 seconds only
            if len(detection_history[camera_id]) == 0 or \
               (datetime.now() - detection_history[camera_id][-1]['timestamp']).total_seconds() > 2:
                detection_history[camera_id].append({
                    'timestamp': datetime.now(),
                    'count': detections
                })
            
            system_stats['total_detections'] += detections
            if detections > system_stats['peak_detections']:
                system_stats['peak_detections'] = detections
        
        # Lightweight overlay with barrier count
        add_professional_overlay(frame, camera_id, detections, violations, barriers_detected)
        
        # Update last violation time if we detected violations this frame
        if violations > 0:
            esp32_buzzer_state['last_violation_time'] = time.time()
        
        # Check if buzzer should stop (no violations for timeout period)
        if ESP32_BUZZER_CONFIG['enabled'] and ESP32_BUZZER_CONFIG['connected'] and esp32_buzzer_state['is_playing']:
            last_violation = esp32_buzzer_state.get('last_violation_time')
            timeout = esp32_buzzer_state.get('violation_timeout', 2.0)
            if last_violation and (time.time() - last_violation) > timeout:
                stop_esp32_buzzer()
                logger.info("ESP32 Buzzer: Auto-stopped - no violations for %.1f seconds", timeout)
        
    except Exception as e:
        logger.error(f"Detection processing error on camera {camera_id}: {e}")
    
    return frame, detections, violations


def add_professional_overlay(frame, camera_id, detections, violations, barriers=0):
    """Add professional information overlay to frame"""
    h, w = frame.shape[:2]
    
    # Top bar - dark semi-transparent
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Camera name
    cam_name = cameras.get(camera_id, {}).get('position', f'Camera {camera_id}')
    cv2.putText(frame, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    (ts_w, _), _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, timestamp, (w-ts_w-10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Bottom status bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Status indicators
    status_y = h - 12
    
    # Detection count
    det_text = f'PERSONS: {detections}'
    det_color = (0, 255, 0) if detections == 0 else (0, 165, 255) if detections < 3 else (0, 0, 255)
    cv2.putText(frame, det_text, (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2)
    
    # Barrier count (if any)
    if barriers > 0:
        barrier_text = f'BARRIERS: {barriers}'
        cv2.putText(frame, barrier_text, (130, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    
    # FPS
    fps = camera_stats.get(camera_id, {}).get('fps', 0)
    fps_text = f'FPS: {fps:.1f}'
    fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(frame, fps_text, (w-120, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 2)
    
    # Status indicator
    status_color = (0, 255, 0) if detections == 0 else (0, 0, 255)
    cv2.circle(frame, (w//2, h-18), 8, status_color, -1)


def generate_camera_frame(camera_id: int):
    """Generate video stream from cached frames - supports multiple clients"""
    global camera_frames, frame_ready_events
    
    if camera_id not in cameras:
        error_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        cv2.putText(error_frame, f'Camera {camera_id} Not Available', 
                   (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return
    
    # Stream cached frames
    while True:
        try:
            # Wait for new frame
            frame_ready_events[camera_id].wait(timeout=1.0)
            frame_ready_events[camera_id].clear()
            
            if camera_frames[camera_id] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + camera_frames[camera_id] + b'\r\n')
            else:
                # No frame yet
                no_feed = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(no_feed, 'Initializing...', (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                ret, buffer = cv2.imencode('.jpg', no_feed, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except GeneratorExit:
            break
        except Exception as e:
            logger.error(f"Stream error camera {camera_id}: {e}")
            time.sleep(0.1)
            time.sleep(0.05)


@app.get("/")
async def read_root():
    """Serve main dashboard HTML with no-cache headers"""
    return HTMLResponse(
        content=get_complete_dashboard_html(),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


@app.get("/api/cameras")
async def get_cameras():
    """Get list of active cameras with current stats"""
    camera_list = []
    for cam_id, cam_info in cameras.items():
        stats = camera_stats.get(cam_id, {})
        settings = camera_settings.get(cam_id, {})
        camera_list.append({
            'id': cam_id,
            'name': cam_info['name'],
            'position': cam_info['position'],
            'device_index': cam_info['device_index'],
            'active': cam_info['active'],
            'recording': cam_info.get('recording', False),
            'detections': stats.get('detections', 0),
            'violations': stats.get('violations', 0),
            'fps': round(stats.get('fps', 0), 1),
            'total_detections': stats.get('total_detections', 0),
            'status': stats.get('status', 'inactive'),
            'resolution': stats.get('resolution', '640x480'),
            'detection_enabled': settings.get('detection_enabled', True)
        })
    return JSONResponse(content={'cameras': camera_list})


@app.get("/api/camera/{camera_id}/stream")
async def get_camera_stream(camera_id: int):
    """Stream video from specific camera"""
    return StreamingResponse(
        generate_camera_frame(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/system/stats")
async def get_system_stats():
    """Get overall system statistics"""
    uptime = (datetime.now() - system_stats['start_time']).total_seconds() if system_stats['start_time'] else 0
    
    # Calculate average FPS
    avg_fps = 0
    active_cams = 0
    for cam_id in cameras.keys():
        if camera_stats.get(cam_id, {}).get('fps', 0) > 0:
            avg_fps += camera_stats[cam_id]['fps']
            active_cams += 1
    if active_cams > 0:
        avg_fps /= active_cams
    
    # Calculate detection rate
    detection_rate = system_stats['total_detections'] / (uptime / 60) if uptime > 60 else 0
    
    return JSONResponse(content={
        'uptime_seconds': int(uptime),
        'uptime_formatted': str(timedelta(seconds=int(uptime))),
        'total_cameras': len(cameras),
        'active_cameras': active_cams,
        'total_detections': system_stats['total_detections'],
        'total_violations': system_stats['total_violations'],
        'peak_detections': system_stats['peak_detections'],
        'average_fps': round(avg_fps, 1),
        'detection_rate_per_minute': round(detection_rate, 2),
        'yolo_available': YOLO_AVAILABLE,
        'system_status': 'operational'
    })


@app.get("/api/events")
async def get_events():
    """Get recent detection events"""
    return JSONResponse(content={'events': list(event_log)[-50:]})


@app.get("/api/violations")
async def get_violations(camera_id: Optional[int] = None, limit: int = 100):
    """Get violation history with optional camera filter"""
    violations = list(violations_log)
    
    # Filter by camera if specified
    if camera_id is not None:
        violations = [v for v in violations if v.get('camera_id') == camera_id]
    
    # Return most recent violations
    violations = violations[-limit:]
    violations.reverse()  # Most recent first
    
    return JSONResponse(content={
        'violations': violations,
        'total_count': len(violations_log),
        'filtered_count': len(violations)
    })


@app.get("/api/violations/export")
async def export_violations(format: str = 'json'):
    """Export violations to JSON or CSV"""
    violations = list(violations_log)
    
    if format == 'csv':
        import io
        import csv
        
        output = io.StringIO()
        if violations:
            writer = csv.DictWriter(output, fieldnames=violations[0].keys())
            writer.writeheader()
            writer.writerows(violations)
        
        from fastapi.responses import Response
        return Response(
            content=output.getvalue(),
            media_type='text/csv',
            headers={'Content-Disposition': f'attachment; filename=violations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )
    else:
        # JSON export
        from fastapi.responses import Response
        return Response(
            content=json.dumps(violations, indent=2),
            media_type='application/json',
            headers={'Content-Disposition': f'attachment; filename=violations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'}
        )


@app.get("/api/violations/stats")
async def get_violation_stats():
    """Get violation statistics"""
    violations = list(violations_log)
    
    # Calculate stats
    total = len(violations)
    by_camera = defaultdict(int)
    by_zone_type = defaultdict(int)
    recent_1h = 0
    recent_24h = 0
    
    now = datetime.now()
    for v in violations:
        ts = datetime.fromisoformat(v['timestamp'])
        by_camera[v['camera_id']] += 1
        by_zone_type[v.get('zone_type', 'unknown')] += 1
        
        delta = (now - ts).total_seconds()
        if delta < 3600:
            recent_1h += 1
        if delta < 86400:
            recent_24h += 1
    
    return JSONResponse(content={
        'total_violations': total,
        'violations_last_hour': recent_1h,
        'violations_last_24h': recent_24h,
        'by_camera': dict(by_camera),
        'by_zone_type': dict(by_zone_type),
        'average_per_hour': round(recent_24h / 24, 2) if recent_24h > 0 else 0
    })


# ============== PDF REPORT GENERATION ==============

@app.get("/api/reports/generate")
async def generate_pdf_report(period: str = "daily"):
    """Generate a PDF report for violations"""
    if not PDF_AVAILABLE:
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': 'PDF generation not available. Install reportlab: pip install reportlab'}
        )
    
    try:
        violations = list(violations_log)
        now = datetime.now()
        
        # Filter by period
        if period == "daily":
            cutoff = now - timedelta(days=1)
            period_name = "Daily"
        elif period == "weekly":
            cutoff = now - timedelta(days=7)
            period_name = "Weekly"
        elif period == "monthly":
            cutoff = now - timedelta(days=30)
            period_name = "Monthly"
        else:
            cutoff = now - timedelta(days=1)
            period_name = "Daily"
        
        filtered_violations = [
            v for v in violations 
            if datetime.fromisoformat(v['timestamp']) > cutoff
        ]
        
        # Generate PDF
        filename = f"vigil_report_{period}_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
        filepath = REPORTS_DIR / filename
        
        doc = SimpleDocTemplate(str(filepath), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1e3a5f')
        )
        story.append(Paragraph(f"VIGIL V6 - {period_name} Security Report", title_style))
        story.append(Spacer(1, 12))
        
        # Report info
        info_style = ParagraphStyle('Info', parent=styles['Normal'], fontSize=11)
        story.append(Paragraph(f"<b>Generated:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}", info_style))
        story.append(Paragraph(f"<b>Report Period:</b> {cutoff.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}", info_style))
        story.append(Paragraph(f"<b>Total Violations:</b> {len(filtered_violations)}", info_style))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Calculate stats
        by_camera = defaultdict(int)
        by_zone_type = defaultdict(int)
        by_hour = defaultdict(int)
        
        for v in filtered_violations:
            by_camera[f"Camera {v['camera_id']}"] += 1
            by_zone_type[v.get('zone_type', 'unknown').upper()] += 1
            ts = datetime.fromisoformat(v['timestamp'])
            by_hour[ts.hour] += 1
        
        # Stats table
        stats_data = [
            ['Metric', 'Value'],
            ['Total Violations', str(len(filtered_violations))],
            ['Cameras with Violations', str(len(by_camera))],
            ['Most Active Camera', max(by_camera, key=by_camera.get) if by_camera else 'N/A'],
            ['Peak Hour', f"{max(by_hour, key=by_hour.get):02d}:00" if by_hour else 'N/A'],
            ['Restricted Zone Violations', str(by_zone_type.get('RESTRICTED', 0))],
            ['Warning Zone Violations', str(by_zone_type.get('WARNING', 0))],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a5f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 25))
        
        # Violations by Camera
        if by_camera:
            story.append(Paragraph("Violations by Camera", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            cam_data = [['Camera', 'Violations', 'Percentage']]
            total = len(filtered_violations)
            for cam, count in sorted(by_camera.items(), key=lambda x: x[1], reverse=True):
                pct = f"{(count/total*100):.1f}%" if total > 0 else "0%"
                cam_data.append([cam, str(count), pct])
            
            cam_table = Table(cam_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            cam_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(cam_table)
            story.append(Spacer(1, 25))
        
        # Violations by Zone Type
        if by_zone_type:
            story.append(Paragraph("Violations by Zone Type", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            zone_data = [['Zone Type', 'Violations', 'Severity']]
            severity_map = {'RESTRICTED': 'HIGH', 'WARNING': 'MEDIUM', 'SAFE': 'LOW'}
            for zone_type, count in sorted(by_zone_type.items(), key=lambda x: x[1], reverse=True):
                severity = severity_map.get(zone_type, 'UNKNOWN')
                zone_data.append([zone_type, str(count), severity])
            
            zone_table = Table(zone_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
            zone_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dc3545')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(zone_table)
            story.append(Spacer(1, 25))
        
        # Recent Violations Detail
        story.append(Paragraph("Recent Violations (Last 20)", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        recent = filtered_violations[-20:] if len(filtered_violations) > 20 else filtered_violations
        if recent:
            detail_data = [['Time', 'Camera', 'Zone', 'Type']]
            for v in reversed(recent):
                ts = datetime.fromisoformat(v['timestamp']).strftime('%m/%d %H:%M:%S')
                detail_data.append([
                    ts,
                    f"Cam {v['camera_id']}",
                    v.get('zone_name', 'Unknown')[:15],
                    v.get('zone_type', 'N/A').upper()
                ])
            
            detail_table = Table(detail_data, colWidths=[1.5*inch, 1*inch, 1.8*inch, 1.2*inch])
            detail_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6c757d')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dee2e6')),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(detail_table)
        else:
            story.append(Paragraph("No violations recorded in this period.", styles['Normal']))
        
        story.append(Spacer(1, 30))
        
        # Footer
        footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9, textColor=colors.gray)
        story.append(Paragraph("━" * 60, footer_style))
        story.append(Paragraph("Generated by VIGIL V6.0 - AI-Powered Pedestrian Detection System", footer_style))
        story.append(Paragraph(f"Report ID: RPT-{now.strftime('%Y%m%d%H%M%S')}", footer_style))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"PDF report generated: {filepath}")
        
        return FileResponse(
            path=str(filepath),
            filename=filename,
            media_type='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return JSONResponse(
            status_code=500,
            content={'status': 'error', 'message': str(e)}
        )


@app.get("/api/reports/list")
async def list_reports():
    """List all generated reports"""
    reports = []
    for f in REPORTS_DIR.glob("*.pdf"):
        stats = f.stat()
        reports.append({
            'filename': f.name,
            'size': stats.st_size,
            'created': datetime.fromtimestamp(stats.st_ctime).isoformat(),
            'download_url': f'/api/reports/download/{f.name}'
        })
    
    return JSONResponse(content={
        'reports': sorted(reports, key=lambda x: x['created'], reverse=True)
    })


@app.get("/api/reports/download/{filename}")
async def download_report(filename: str):
    """Download a specific report"""
    filepath = REPORTS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type='application/pdf'
    )


# ============== AUDIO ALARM SYSTEM ==============

@app.get("/api/audio/settings")
async def get_audio_settings():
    """Get audio alarm settings"""
    return JSONResponse(content=AUDIO_ALARM_CONFIG)


@app.post("/api/audio/settings")
async def update_audio_settings(request: dict):
    """Update audio alarm settings"""
    global AUDIO_ALARM_CONFIG
    
    if 'enabled' in request:
        AUDIO_ALARM_CONFIG['enabled'] = bool(request['enabled'])
    if 'volume' in request:
        AUDIO_ALARM_CONFIG['volume'] = max(0.0, min(1.0, float(request['volume'])))
    if 'sound_type' in request:
        AUDIO_ALARM_CONFIG['sound_type'] = request['sound_type']
    if 'cooldown' in request:
        AUDIO_ALARM_CONFIG['cooldown'] = int(request['cooldown'])
    
    logger.info(f"Audio settings updated: {AUDIO_ALARM_CONFIG}")
    return JSONResponse(content={'status': 'success', 'settings': AUDIO_ALARM_CONFIG})


@app.post("/api/audio/test")
async def test_audio_alarm(request: dict = None):
    """Trigger a test alarm sound (handled by frontend)"""
    sound_type = request.get('sound_type', AUDIO_ALARM_CONFIG['sound_type']) if request else AUDIO_ALARM_CONFIG['sound_type']
    return JSONResponse(content={
        'status': 'success',
        'message': 'Test alarm triggered',
        'sound_type': sound_type,
        'volume': AUDIO_ALARM_CONFIG['volume']
    })


@app.post("/api/violations/alert")
async def send_violation_alert(violation: dict):
    """Send email alert for violation (if configured)"""
    if not EMAIL_CONFIG.get('enabled', False):
        return JSONResponse(content={'status': 'disabled', 'message': 'Email alerts not configured'})
    
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = ', '.join(EMAIL_CONFIG['recipient_emails'])
        msg['Subject'] = f"🚨 VIGIL Alert: Zone Violation Detected"
        
        body = f"""
        VIGIL V6 - Zone Violation Alert
        
        Time: {violation.get('timestamp')}
        Camera: {violation.get('camera_name')}
        Zone: {violation.get('zone_name')} ({violation.get('zone_type')})
        
        A person has entered a restricted zone. Please review the camera feed.
        
        ---
        This is an automated alert from VIGIL AI Detection System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            server.send_message(msg)
        
        logger.info(f"Email alert sent for violation in zone {violation.get('zone_name')}")
        return JSONResponse(content={'status': 'success', 'message': 'Email alert sent'})
        
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
        return JSONResponse(content={'status': 'error', 'message': str(e)})


@app.get("/api/camera/{camera_id}/zones")
async def get_camera_zones(camera_id: int):
    """Get all zones for a camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return JSONResponse(content={'zones': camera_zones.get(camera_id, [])})


@app.post("/api/camera/{camera_id}/zones")
async def add_camera_zone(camera_id: int, zone_data: Dict[str, Any] = Body(...)):
    """Add zone to camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Generate unique ID for zone
    import uuid
    zone_data['id'] = str(uuid.uuid4())
    zone_data['camera_id'] = camera_id
    zone_data['active'] = zone_data.get('active', True)
    zone_data['created_at'] = datetime.now().isoformat()
    
    camera_zones[camera_id].append(zone_data)
    save_zones()
    
    event_log.append({
        'timestamp': datetime.now().isoformat(),
        'camera_id': camera_id,
        'event_type': 'zone_created',
        'description': f'Zone "{zone_data.get("name", "Unknown")}" ({zone_data.get("type", "unknown")}) created',
        'confidence': 1.0
    })
    
    return JSONResponse(content={'status': 'success', 'zone': zone_data})


@app.put("/api/camera/{camera_id}/zones/{zone_id}")
async def update_camera_zone(camera_id: int, zone_id: str, zone_data: Dict[str, Any] = Body(...)):
    """Update existing zone"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    zones = camera_zones.get(camera_id, [])
    for i, zone in enumerate(zones):
        if zone.get('id') == zone_id:
            # Update zone fields
            zone.update(zone_data)
            zone['updated_at'] = datetime.now().isoformat()
            camera_zones[camera_id][i] = zone
            save_zones()
            
            event_log.append({
                'timestamp': datetime.now().isoformat(),
                'camera_id': camera_id,
                'event_type': 'zone_updated',
                'description': f'Zone "{zone["name"]}" updated',
                'confidence': 1.0
            })
            
            return JSONResponse(content={'status': 'success', 'zone': zone})
    
    raise HTTPException(status_code=404, detail="Zone not found")


@app.delete("/api/camera/{camera_id}/zones/{zone_id}")
async def delete_camera_zone(camera_id: int, zone_id: str):
    """Delete zone from camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    zones = camera_zones.get(camera_id, [])
    for i, zone in enumerate(zones):
        if zone.get('id') == zone_id:
            removed_zone = camera_zones[camera_id].pop(i)
            save_zones()
            
            event_log.append({
                'timestamp': datetime.now().isoformat(),
                'camera_id': camera_id,
                'event_type': 'zone_deleted',
                'description': f'Zone "{removed_zone["name"]}" deleted',
                'confidence': 1.0
            })
            
            return JSONResponse(content={'status': 'success', 'message': 'Zone deleted'})
    
    raise HTTPException(status_code=404, detail="Zone not found")


@app.post("/api/camera/{camera_id}/detection")
async def toggle_camera_detection(camera_id: int, request: dict):
    """Toggle detection for a specific camera"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    enabled = request.get('enabled', True)
    camera_settings[camera_id]['detection_enabled'] = enabled
    
    event_log.append({
        'timestamp': datetime.now().isoformat(),
        'camera_id': camera_id,
        'event_type': 'detection_toggle',
        'description': f'Detection {"enabled" if enabled else "disabled"}',
        'confidence': 1.0
    })
    
    return JSONResponse(content={'status': 'success', 'detection_enabled': enabled})


@app.post("/api/camera/{camera_id}/recording")
async def toggle_camera_recording(camera_id: int, request: dict):
    """Toggle recording for a specific camera - now actually records to MP4"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    recording = request.get('recording', False)
    result_info = None
    
    if recording:
        # Start recording
        success = start_recording(camera_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start recording")
        cameras[camera_id]['recording'] = True
        description = 'Recording STARTED'
        result_info = recording_info.get(camera_id, {})
    else:
        # Stop recording
        result_info = stop_recording(camera_id)
        cameras[camera_id]['recording'] = False
        description = 'Recording STOPPED'
    
    event_log.append({
        'timestamp': datetime.now().isoformat(),
        'camera_id': camera_id,
        'event_type': 'recording_toggle',
        'description': description,
        'confidence': 1.0
    })
    
    response_data = {
        'status': 'success', 
        'recording': recording
    }
    
    if result_info:
        response_data['recording_info'] = {
            'filename': result_info.get('filename', ''),
            'filepath': result_info.get('filepath', ''),
            'frame_count': result_info.get('frame_count', 0),
            'duration': result_info.get('duration', 0) if not recording else None
        }
    
    return JSONResponse(content=response_data)


@app.get("/api/recordings")
async def list_recordings():
    """List all saved recordings"""
    recordings = []
    
    try:
        for file in sorted(RECORDINGS_DIR.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True):
            stat = file.stat()
            recordings.append({
                'filename': file.name,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'path': str(file)
            })
    except Exception as e:
        logger.error(f"Error listing recordings: {e}")
    
    return JSONResponse(content={'recordings': recordings, 'count': len(recordings)})


@app.get("/api/recordings/{filename}")
async def download_recording(filename: str):
    """Download a recording file"""
    filepath = RECORDINGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return FileResponse(
        path=str(filepath),
        filename=filename,
        media_type='video/mp4'
    )


@app.delete("/api/recordings/{filename}")
async def delete_recording(filename: str):
    """Delete a recording file"""
    filepath = RECORDINGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    
    try:
        filepath.unlink()
        return JSONResponse(content={'status': 'success', 'message': f'Deleted {filename}'})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete: {e}")


@app.get("/api/tamper/status")
async def get_tamper_status():
    """Get tamper detection status for all cameras"""
    status = {}
    
    for camera_id in cameras.keys():
        state = tamper_state[camera_id]
        status[camera_id] = {
            'is_tampered': state['is_tampered'],
            'tamper_type': state['tamper_type'],
            'warning_sent': state['warning_sent'],
            'camera_name': cameras.get(camera_id, {}).get('position', f'Camera {camera_id}')
        }
    
    return JSONResponse(content={
        'tamper_status': status,
        'config': TAMPER_DETECTION_CONFIG
    })


@app.post("/api/tamper/settings")
async def update_tamper_settings(request: dict):
    """Update tamper detection settings"""
    global TAMPER_DETECTION_CONFIG
    
    if 'enabled' in request:
        TAMPER_DETECTION_CONFIG['enabled'] = request['enabled']
    if 'darkness_threshold' in request:
        TAMPER_DETECTION_CONFIG['darkness_threshold'] = int(request['darkness_threshold'])
    if 'warning_delay' in request:
        TAMPER_DETECTION_CONFIG['warning_delay'] = int(request['warning_delay'])
    if 'static_threshold' in request:
        TAMPER_DETECTION_CONFIG['static_threshold'] = float(request['static_threshold'])
    
    return JSONResponse(content={
        'status': 'success',
        'config': TAMPER_DETECTION_CONFIG
    })


@app.post("/api/system/power")
async def toggle_system_power(request: dict):
    """Toggle system power on/off"""
    global system_stats
    
    enabled = request.get('enabled', True)
    system_stats['system_enabled'] = enabled
    
    # Disable/enable all camera detection
    for cam_id in camera_settings.keys():
        camera_settings[cam_id]['detection_enabled'] = enabled
    
    event_log.append({
        'timestamp': datetime.now().isoformat(),
        'camera_id': -1,
        'event_type': 'system_power',
        'description': f'System {"enabled" if enabled else "disabled"}',
        'confidence': 1.0
    })
    
    return JSONResponse(content={'status': 'success', 'enabled': enabled})


@app.post("/api/system/shutdown")
async def shutdown_server():
    """Shutdown the server gracefully"""
    import os
    import signal
    
    logger.info("Shutdown requested via API")
    
    # Release all cameras
    for cam_id, cam_data in cameras.items():
        try:
            if cam_data is not None and isinstance(cam_data, dict):
                cam_data['running'] = False  # Stop capture thread
                cap = cam_data.get('capture')
                if cap is not None and hasattr(cap, 'release'):
                    cap.release()
                    logger.info(f"Camera {cam_id} released")
        except Exception as e:
            logger.error(f"Error releasing camera {cam_id}: {e}")
    
    event_log.append({
        'timestamp': datetime.now().isoformat(),
        'camera_id': -1,
        'event_type': 'system_shutdown',
        'description': 'Server shutdown requested',
        'confidence': 1.0
    })
    
    # Schedule shutdown after response
    async def do_shutdown():
        await asyncio.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    
    asyncio.create_task(do_shutdown())
    
    return JSONResponse(content={'status': 'shutting_down', 'message': 'Server is shutting down...'})


@app.get("/api/barrier/settings")
async def get_barrier_settings():
    """Get current barrier detection settings"""
    return JSONResponse(content=barrier_settings)


@app.post("/api/barrier/settings")
async def update_barrier_settings(request: dict):
    """Update barrier detection settings"""
    global barrier_settings
    
    if 'enabled' in request:
        barrier_settings['enabled'] = bool(request['enabled'])
    if 'min_area' in request:
        barrier_settings['min_area'] = int(request['min_area'])
    if 'max_area' in request:
        barrier_settings['max_area'] = int(request['max_area'])
    if 'min_width' in request:
        barrier_settings['min_width'] = int(request['min_width'])
    if 'min_height' in request:
        barrier_settings['min_height'] = int(request['min_height'])
    if 'min_saturation' in request:
        barrier_settings['min_saturation'] = int(request['min_saturation'])
    if 'min_value' in request:
        barrier_settings['min_value'] = int(request['min_value'])
    if 'min_color_coverage' in request:
        barrier_settings['min_color_coverage'] = float(request['min_color_coverage'])
    if 'show_debug' in request:
        barrier_settings['show_debug'] = bool(request['show_debug'])
    
    logger.info(f"Barrier settings updated: {barrier_settings}")
    
    return JSONResponse(content={'status': 'success', 'settings': barrier_settings})


@app.get("/api/system/params")
async def get_system_params():
    """Get system parameters (CPU, Memory, Temperature)"""
    import psutil
    
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        
        # Get temperature (try different methods)
        temp = 0
        try:
            temps = psutil.sensors_temperatures()
            if 'coretemp' in temps:
                temp = temps['coretemp'][0].current
            elif 'cpu_thermal' in temps:
                temp = temps['cpu_thermal'][0].current
            else:
                # Fallback: estimate based on CPU usage
                temp = 30 + (cpu_percent * 0.6)
        except:
            # Fallback: estimate based on CPU usage
            temp = 30 + (cpu_percent * 0.6)
        
        return JSONResponse(content={
            'cpu': round(cpu_percent, 1),
            'memory': round(mem_percent, 1),
            'temperature': round(temp, 1)
        })
    except Exception as e:
        logger.error(f"Error getting system params: {e}")
        return JSONResponse(content={
            'cpu': 0,
            'memory': 0,
            'temperature': 0
        })


@app.delete("/api/camera/{camera_id}/zones/{zone_id}")
async def delete_camera_zone(camera_id: int, zone_id: str):
    """Delete zone from camera"""
    if camera_id not in camera_zones:
        raise HTTPException(status_code=404, detail="No zones found")
    
    camera_zones[camera_id] = [z for z in camera_zones[camera_id] if z['id'] != zone_id]
    return JSONResponse(content={'status': 'success'})


@app.get("/api/camera/{camera_id}/directions")
async def get_camera_directions(camera_id: int):
    """Get direction statistics for camera"""
    return JSONResponse(content={'directions': dict(direction_stats.get(camera_id, {}))})


@app.get("/api/camera/{camera_id}/history")
async def get_camera_history(camera_id: int):
    """Get detection history for camera"""
    history = list(detection_history.get(camera_id, []))
    return JSONResponse(content={'history': [
        {'timestamp': h['timestamp'].isoformat(), 'count': h['count']} 
        for h in history
    ]})


@app.post("/api/camera/{camera_id}/settings")
async def update_camera_settings(camera_id: int, settings: Dict[str, Any]):
    """Update camera settings"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    camera_settings[camera_id].update(settings)
    return JSONResponse(content={'status': 'success', 'settings': camera_settings[camera_id]})


@app.get("/api/camera/{camera_id}/settings")
async def get_camera_settings(camera_id: int):
    """Get camera settings"""
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    return JSONResponse(content=camera_settings.get(camera_id, {}))


def get_complete_dashboard_html():
    """Generate complete professional HTML dashboard with ALL features on ONE page"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>VIGIL V6.0 - Complete Control Center</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        /* Animation keyframes */
        @keyframes pulse {
            0% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.02); }
            100% { opacity: 1; transform: scale(1); }
        }
        
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        @keyframes tamper-alert {
            0%, 100% { border-color: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
            50% { border-color: #fbbf24; box-shadow: 0 0 20px rgba(251, 191, 36, 0.7); }
        }
        
        .recording-active {
            animation: pulse 1s infinite !important;
        }
        
        .tamper-warning {
            animation: tamper-alert 0.5s infinite !important;
            border: 4px solid #ef4444 !important;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #ffffff;
            color: #1a1a1a;
            height: 100vh;
            overflow: hidden;
        }
        
        /* Touch Mode Styles */
        body.touch-mode .control-btn,
        body.touch-mode .tab-btn,
        body.touch-mode .camera-control-btn,
        body.touch-mode .zone-tool-btn,
        body.touch-mode button {
            min-height: 48px !important;
            min-width: 48px !important;
            font-size: 16px !important;
            padding: 12px 20px !important;
        }
        
        body.touch-mode .camera-controls {
            gap: 12px !important;
            padding: 15px !important;
        }
        
        body.touch-mode .camera-card {
            margin-bottom: 15px;
        }
        
        body.touch-mode .camera-grid {
            grid-template-columns: 1fr !important;
            gap: 15px !important;
        }
        
        body.touch-mode .camera-feed {
            height: 300px !important;
        }
        
        body.touch-mode .tab-nav {
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }
        
        body.touch-mode .tab-btn {
            white-space: nowrap;
            padding: 15px 20px !important;
        }
        
        body.touch-mode .stat-card {
            padding: 20px !important;
        }
        
        body.touch-mode .stat-value {
            font-size: 32px !important;
        }
        
        body.touch-mode .stat-label {
            font-size: 14px !important;
        }
        
        body.touch-mode input,
        body.touch-mode select {
            min-height: 48px !important;
            font-size: 16px !important;
            padding: 12px !important;
        }
        
        body.touch-mode .right-panel {
            width: 350px !important;
        }
        
        body.touch-mode .log-entry {
            padding: 12px !important;
            font-size: 14px !important;
        }
        
        body.touch-mode .camera-stat-item {
            padding: 12px !important;
        }
        
        body.touch-mode .camera-stat-value {
            font-size: 20px !important;
        }
        
        /* Touch Toggle Switch */
        .touch-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-right: 15px;
        }
        
        .touch-toggle-label {
            font-size: 12px;
            color: #6b7280;
            font-weight: 500;
        }
        
        .touch-switch {
            position: relative;
            width: 50px;
            height: 26px;
            background: #e5e7eb;
            border-radius: 13px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .touch-switch.active {
            background: #10b981;
        }
        
        .touch-switch::after {
            content: '';
            position: absolute;
            width: 22px;
            height: 22px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .touch-switch.active::after {
            transform: translateX(24px);
        }
        
        .touch-icon {
            font-size: 16px;
        }
        
        /* COMPACT DISPLAY MODE - For small screens like 1024x600 */
        body.compact-mode .header {
            padding: 8px 12px !important;
        }
        
        body.compact-mode .header h1 {
            font-size: 14px !important;
        }
        
        body.compact-mode .header .subtitle {
            display: none !important;
        }
        
        body.compact-mode .tab-nav {
            padding: 4px 8px !important;
        }
        
        body.compact-mode .tab-btn {
            padding: 6px 10px !important;
            font-size: 11px !important;
        }
        
        body.compact-mode .camera-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 6px !important;
            padding: 6px !important;
        }
        
        body.compact-mode .camera-card {
            border-radius: 4px !important;
        }
        
        body.compact-mode .camera-header {
            padding: 4px 8px !important;
        }
        
        body.compact-mode .camera-title {
            font-size: 10px !important;
        }
        
        body.compact-mode .camera-status {
            font-size: 8px !important;
        }
        
        body.compact-mode .status-dot {
            width: 5px !important;
            height: 5px !important;
        }
        
        body.compact-mode .camera-feed {
            height: 180px !important;
        }
        
        body.compact-mode .camera-controls {
            display: none !important;
        }
        
        body.compact-mode .camera-stats {
            display: none !important;
        }
        
        body.compact-mode .left-panel,
        body.compact-mode .right-panel {
            display: none !important;
        }
        
        body.compact-mode .main-container {
            grid-template-columns: 1fr !important;
            width: 100% !important;
        }
        
        body.compact-mode .center-section {
            width: 100% !important;
            margin: 0 !important;
            padding: 8px !important;
        }
        
        body.compact-mode .stat-card {
            padding: 6px !important;
        }
        
        body.compact-mode .stat-value {
            font-size: 14px !important;
        }
        
        body.compact-mode .stat-label {
            font-size: 8px !important;
        }
        
        body.compact-mode .logo-icon {
            width: 28px !important;
            height: 28px !important;
            font-size: 14px !important;
        }
        
        body.compact-mode .touch-toggle-label,
        body.compact-mode .touch-icon {
            display: none !important;
        }
        
        body.compact-mode .status-badge {
            font-size: 9px !important;
            padding: 3px 8px !important;
        }
        
        /* Ultra compact for 1024x600 showing all 4 cameras */
        body.ultra-compact-mode .header {
            padding: 2px 8px !important;
            min-height: 28px !important;
        }
        
        body.ultra-compact-mode .header h1 {
            font-size: 11px !important;
        }
        
        body.ultra-compact-mode .header .subtitle {
            display: none !important;
        }
        
        body.ultra-compact-mode .tab-nav {
            padding: 1px 4px !important;
            min-height: 24px !important;
        }
        
        body.ultra-compact-mode .tab-btn {
            padding: 3px 6px !important;
            font-size: 9px !important;
        }
        
        body.ultra-compact-mode .camera-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 3px !important;
            padding: 3px !important;
            height: calc(100vh - 55px) !important;
        }
        
        body.ultra-compact-mode .camera-card {
            border-radius: 2px !important;
            height: calc((100vh - 65px) / 2) !important;
            overflow: hidden !important;
        }
        
        body.ultra-compact-mode .camera-header {
            padding: 1px 4px !important;
            min-height: 16px !important;
        }
        
        body.ultra-compact-mode .camera-title {
            font-size: 8px !important;
        }
        
        body.ultra-compact-mode .camera-status {
            font-size: 6px !important;
        }
        
        body.ultra-compact-mode .status-dot {
            width: 4px !important;
            height: 4px !important;
        }
        
        body.ultra-compact-mode .camera-feed {
            height: calc(100% - 18px) !important;
        }
        
        body.ultra-compact-mode .camera-controls {
            display: none !important;
        }
        
        body.ultra-compact-mode .camera-stats {
            display: none !important;
        }
        
        body.ultra-compact-mode .left-panel,
        body.ultra-compact-mode .right-panel {
            display: none !important;
        }
        
        body.ultra-compact-mode .main-container {
            grid-template-columns: 1fr !important;
            width: 100% !important;
            height: calc(100vh - 55px) !important;
        }
        
        body.ultra-compact-mode .center-section {
            width: 100% !important;
            margin: 0 !important;
            padding: 3px !important;
            height: 100% !important;
        }
        
        body.ultra-compact-mode .camera-feed img {
            object-fit: contain !important;
        }
        
        body.ultra-compact-mode .logo-section {
            gap: 8px !important;
        }
        
        body.ultra-compact-mode .logo-icon {
            width: 20px !important;
            height: 20px !important;
            font-size: 10px !important;
        }
        
        body.ultra-compact-mode .touch-toggle,
        body.ultra-compact-mode .status-badge {
            display: none !important;
        }
        
        body.ultra-compact-mode .display-mode-toggle {
            gap: 2px !important;
            margin-right: 4px !important;
        }
        
        body.ultra-compact-mode .display-mode-btn {
            padding: 2px 5px !important;
            font-size: 8px !important;
        }
        
        body.ultra-compact-mode .title-section .subtitle {
            display: none !important;
        }
        
        body.ultra-compact-mode .logo-section {
            gap: 8px !important;
        }
        
        /* Display mode toggle button */
        .display-mode-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-right: 10px;
        }
        
        .display-mode-btn {
            padding: 4px 10px;
            background: #6b7280;
            color: white;
            border: none;
            border-radius: 4px;
            font-size: 10px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .display-mode-btn:hover {
            background: #4b5563;
        }
        
        .display-mode-btn.active {
            background: #10b981;
        }
        
        /* Shutdown button */
        .shutdown-btn {
            width: 32px;
            height: 32px;
            background: #dc2626;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .shutdown-btn:hover {
            background: #b91c1c;
            transform: scale(1.05);
        }
        
        body.ultra-compact-mode .shutdown-btn {
            width: 24px !important;
            height: 24px !important;
            font-size: 12px !important;
        }
        
        .header {
            background: #ffffff;
            border-bottom: 2px solid #f59e0b;
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo-section {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #f59e0b, #ef4444);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
        }
        
        .title-section h1 {
            font-size: 20px;
            font-weight: 600;
            color: #1a1a1a;
            letter-spacing: -0.5px;
        }
        
        .title-section .subtitle {
            font-size: 12px;
            color: #6b7280;
            margin-top: 2px;
            font-weight: 400;
        }
        
        .status-badge {
            background: #10b981;
            color: #ffffff;
            padding: 6px 16px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        
        /* Tab Navigation */
        .tab-nav {
            display: flex;
            background: #f9fafb;
            border-bottom: 2px solid #e5e7eb;
            padding: 0 20px;
            gap: 5px;
        }
        
        .tab-btn {
            padding: 12px 24px;
            background: transparent;
            border: none;
            color: #6b7280;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tab-btn:hover {
            color: #1a1a1a;
            background: #ffffff;
        }
        
        .tab-btn.active {
            color: #f59e0b;
            border-bottom-color: #f59e0b;
            background: #ffffff;
        }
        
        .tab-content {
            display: none;
            overflow-y: auto;
            max-height: calc(100vh - 180px);
        }
        
        .tab-content.active {
            display: block;
        }
        
        .main-container {
            display: grid;
            grid-template-columns: 250px 1fr 300px;
            gap: 0;
            height: calc(100vh - 130px);
            overflow: hidden;
        }
        
        .left-panel {
            background: #f9fafb;
            border-right: 1px solid #e5e7eb;
            padding: 20px;
            overflow-y: auto;
        }
        
        .center-section {
            padding: 20px;
            overflow-y: auto;
            background: #ffffff;
        }
        
        .right-panel {
            background: #f9fafb;
            border-left: 1px solid #e5e7eb;
            padding: 20px;
            overflow-y: auto;
        }
        
        .stats-compact {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        
        .stat-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }
        
        .stat-label {
            font-size: 11px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
            font-weight: 500;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #1a1a1a;
        }
        
        .camera-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }
        
        .camera-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            height: fit-content;
        }
        
        .camera-header {
            background: #f9fafb;
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .camera-title {
            font-weight: 600;
            font-size: 13px;
            color: #1a1a1a;
        }
        
        .camera-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
        }
        
        .camera-feed {
            width: 100%;
            height: 400px;
            background: #000;
            display: block;
            position: relative;
            overflow: hidden;
        }
        
        .camera-feed img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        
        .zone-canvas {
            z-index: 10;
            background: rgba(0, 0, 0, 0.05);
        }
        
        .camera-controls {
            display: flex !important;
            gap: 8px !important;
            padding: 10px !important;
            background: #ffffff !important;
            border-top: 2px solid #e5e7eb !important;
            flex-wrap: wrap !important;
            flex-shrink: 0 !important;
            min-height: 50px !important;
        }
        
        .control-btn {
            flex: 1 !important;
            min-width: 90px !important;
            padding: 8px 12px !important;
            color: white !important;
            border: none !important;
            border-radius: 6px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transition: transform 0.1s, box-shadow 0.1s !important;
        }
        
        .control-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15) !important;
        }
        
        .control-btn:active {
            transform: translateY(0);
        }
        
        /* Power Switch */
        .power-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 26px;
        }
        
        .power-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        
        .power-slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #dc2626;
            transition: 0.4s;
            border-radius: 26px;
        }
        
        .power-slider:before {
            position: absolute;
            content: "";
            height: 20px;
            width: 20px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: 0.4s;
            border-radius: 50%;
        }
        
        .power-switch input:checked + .power-slider {
            background-color: #10b981;
        }
        
        .power-switch input:checked + .power-slider:before {
            transform: translateX(24px);
        }
        
        .camera-stats {
            display: grid !important;
            grid-template-columns: repeat(3, 1fr);
            background: #f9fafb;
            border-top: 1px solid #e5e7eb;
            margin-top: auto;
        }
        
        .camera-stat-item {
            padding: 12px;
            text-align: center;
            border-right: 1px solid #e5e7eb;
        }
        
        .camera-stat-item:last-child {
            border-right: none;
        }
        
        .camera-stat-label {
            font-size: 10px;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 4px;
            font-weight: 500;
        }
        
        .camera-stat-value {
            font-size: 18px;
            font-weight: 700;
            color: #1a1a1a;
        }
        
        .panel {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .panel-header {
            font-size: 13px;
            font-weight: 600;
            color: #1a1a1a;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e5e7eb;
        }
        
        .panel-compact {
            padding: 8px;
        }
        
        .panel-header-compact {
            font-size: 11px;
            margin-bottom: 8px;
            padding-bottom: 6px;
        }
        
        .system-info-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e5e7eb;
            font-size: 13px;
        }
        
        .system-info-item:last-child {
            border-bottom: none;
        }
        
        .info-label {
            color: #6b7280;
            font-weight: 400;
        }
        
        .info-value {
            color: #1a1a1a;
            font-weight: 600;
        }
        
        .activity-log {
            max-height: 200px;
            overflow-y: auto;
        }
        
        .log-entry {
            padding: 6px 8px;
            margin-bottom: 5px;
            background: #0a0a0a;
            border-left: 2px solid #00d4ff;
            border-radius: 3px;
            font-size: 10px;
        }
        
        .log-time {
            color: #666;
            font-size: 9px;
        }
        
        .log-message {
            color: #e0e0e0;
            margin-top: 3px;
        }
        
        .control-button {
            width: 100%;
            padding: 10px;
            background: #f59e0b;
            border: 1px solid #f59e0b;
            color: #ffffff;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
            margin-bottom: 8px;
        }
        
        .control-button:hover {
            background: #d97706;
            border-color: #d97706;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0a0a0a;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #0f3460;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #00d4ff;
        }
        
        /* Tabs Navigation */
        .tabs-nav {
            display: flex;
            gap: 10px;
            background: #0f1419;
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        
        .tab-button {
            padding: 12px 24px;
            background: transparent;
            border: 1px solid #0f3460;
            color: #888;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 12px;
        }
        
        .tab-button:hover {
            border-color: #00d4ff;
            color: #00d4ff;
        }
        
        .tab-button.active {
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border-color: #00d4ff;
            color: #00d4ff;
            box-shadow: 0 0 15px rgba(0,212,255,0.3);
        }
        
        .tab-content {
            display: none;
            overflow-y: auto;
            max-height: calc(100vh - 180px);
            padding-bottom: 20px;
        }
        
        .tab-content.active {
            display: block;
        }
        
        /* Zone Management */
        .zone-canvas-container {
            position: relative;
            display: inline-block;
        }
        
        .zone-canvas {
            position: absolute;
            top: 0;
            left: 0;
            cursor: crosshair;
            pointer-events: all;
        }
        
        .zone-tools {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        
        .zone-tool-btn {
            padding: 10px 20px;
            background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
            border: 1px solid #0f3460;
            color: #00d4ff;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 12px;
            font-weight: 600;
        }
        
        .zone-tool-btn:hover {
            background: linear-gradient(135deg, #00d4ff 0%, #0f3460 100%);
            color: #0a0a0a;
        }
        
        .zone-tool-btn.active {
            background: #00d4ff;
            color: #0a0a0a;
            box-shadow: 0 0 15px rgba(0,212,255,0.5);
        }
        
        .zone-list {
            margin-top: 20px;
        }
        
        .zone-item {
            padding: 12px;
            background: #0a0a0a;
            border-left: 3px solid #00d4ff;
            border-radius: 4px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .zone-item-name {
            font-weight: 600;
            color: #00d4ff;
        }
        
        .zone-item-type {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
        }
        
        .zone-item-actions {
            display: flex;
            gap: 8px;
        }
        
        .zone-action-btn {
            padding: 6px 12px;
            background: #0f3460;
            border: 1px solid #00d4ff;
            color: #00d4ff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
        }
        
        /* Direction Compass */
        .compass-container {
            width: 200px;
            height: 200px;
            margin: 20px auto;
            position: relative;
        }
        
        .compass-svg {
            width: 100%;
            height: 100%;
        }
        
        .direction-stat {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #0f3460;
            font-size: 13px;
        }
        
        .direction-label {
            color: #888;
        }
        
        .direction-count {
            color: #00d4ff;
            font-weight: 600;
        }
        
        /* Settings Panel */
        .setting-item {
            margin-bottom: 20px;
        }
        
        .setting-label {
            display: block;
            color: #888;
            font-size: 13px;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .setting-input {
            width: 100%;
            padding: 10px;
            background: #0a0a0a;
            border: 1px solid #0f3460;
            color: #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
        }
        
        .setting-input:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 10px rgba(0,212,255,0.3);
        }
        
        .setting-slider {
            width: 100%;
        }
        
        .setting-checkbox {
            width: 20px;
            height: 20px;
        }
        
        /* Camera Detail View */
        .camera-detail {
            background: #1a1a2e;
            border: 2px solid #0f3460;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .camera-detail-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #0f3460;
        }
        
        .camera-controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .camera-control-btn {
            padding: 8px 16px;
            background: #0f3460;
            border: 1px solid #00d4ff;
            color: #00d4ff;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }
        
        .camera-control-btn:hover {
            background: #00d4ff;
            color: #0a0a0a;
        }
        
        /* Chart Container */
        .chart-container {
            width: 100%;
            height: 200px;
            background: #0a0a0a;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo-section">
            <div class="logo-icon">V</div>
            <div class="title-section">
                <h1>VIGIL</h1>
                <div class="subtitle">Vehicle-Installed Guard for Injury Limitation</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <div class="display-mode-toggle">
                <button class="display-mode-btn" id="displayNormal" onclick="setDisplayMode('normal')" title="Normal View">🖥️</button>
                <button class="display-mode-btn" id="displayCompact" onclick="setDisplayMode('compact')" title="Compact (1280x720)">📱</button>
                <button class="display-mode-btn" id="displayUltra" onclick="setDisplayMode('ultra')" title="Ultra Compact (1024x600)">📺</button>
            </div>
            <div class="touch-toggle">
                <span class="touch-icon">👆</span>
                <span class="touch-toggle-label">Touch Mode</span>
                <div class="touch-switch" id="touchModeToggle" onclick="toggleTouchMode()"></div>
            </div>
            <div class="status-badge" id="systemStatus">OPERATIONAL</div>
            <button class="shutdown-btn" onclick="shutdownServer()" title="Shutdown Server">⏻</button>
        </div>
    </div>
    
    <!-- TAB NAVIGATION -->
    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('dashboard')">Dashboard</button>
        <button class="tab-btn" onclick="switchTab('cameras')">Cameras</button>
        <button class="tab-btn" onclick="switchTab('zones')">Zones</button>
        <button class="tab-btn" onclick="switchTab('settings')">Settings</button>
        <button class="tab-btn" onclick="switchTab('events')">Events</button>
    </div>
    
    <!-- DASHBOARD TAB -->
    <div id="tab-dashboard" class="tab-content active">
    <div class="main-container">
        <!-- LEFT PANEL - Controls & Quick Stats -->
        <div class="left-panel">
            <!-- System Power Control -->
            <div class="panel panel-compact" style="margin-bottom: 15px;">
                <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px;">
                    <span style="font-weight: 600; color: #1a1a1a; font-size: 13px;">SYSTEM POWER</span>
                    <label class="power-switch">
                        <input type="checkbox" id="systemPowerToggle" checked onchange="toggleSystemPower(this.checked)">
                        <span class="power-slider"></span>
                    </label>
                </div>
            </div>
            
            <!-- Quick Stats -->
            <div class="panel panel-compact">
                <div class="panel-header panel-header-compact">SYSTEM STATS</div>
                <div class="stats-compact">
                    <div class="stat-card">
                        <div class="stat-label">Total Detections</div>
                        <div class="stat-value" id="totalDetections">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Active Cameras</div>
                        <div class="stat-value" id="activeCameras">0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average FPS</div>
                        <div class="stat-value" id="avgFPS">0.0</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">System Uptime</div>
                        <div class="stat-value" id="uptime" style="font-size: 20px;">00:00:00</div>
                    </div>
                </div>
            </div>
            
            <!-- System Parameters Graph -->
            <div class="panel" style="margin-top: 15px;">
                <div class="panel-header">System Parameters</div>
                <div style="padding: 10px;">
                    <div class="param-item">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-size: 11px; color: #6b7280;">CPU Usage</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="cpuValue">0%</span>
                        </div>
                        <div style="width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                            <div id="cpuBar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #10b981, #059669); transition: width 0.3s;"></div>
                        </div>
                    </div>
                    <div class="param-item" style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-size: 11px; color: #6b7280;">Memory Usage</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="memValue">0%</span>
                        </div>
                        <div style="width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                            <div id="memBar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #3b82f6, #2563eb); transition: width 0.3s;"></div>
                        </div>
                    </div>
                    <div class="param-item" style="margin-top: 15px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-size: 11px; color: #6b7280;">Temperature</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="tempValue">0°C</span>
                        </div>
                        <div style="width: 100%; height: 8px; background: #e5e7eb; border-radius: 4px; overflow: hidden;">
                            <div id="tempBar" style="height: 100%; width: 0%; background: linear-gradient(90deg, #f59e0b, #ef4444); transition: width 0.3s;"></div>
                        </div>
                    </div>
                    <canvas id="systemParamsChart" style="width: 100%; height: 150px; margin-top: 15px;"></canvas>
                </div>
            </div>
        </div>
        
        <!-- CENTER SECTION - Camera Grid -->
        <div class="center-section">
            <div class="camera-grid" id="cameraGrid">
                <!-- Cameras will be loaded dynamically -->
            </div>
        </div>
        
        <!-- RIGHT PANEL - System Info & Activity -->
        <div class="right-panel">
            <!-- Detection Analytics Graph -->
            <div class="panel">
                <div class="panel-header">Detection Analytics</div>
                <div style="padding: 15px;">
                    <canvas id="detectionChart" style="width: 100%; height: 180px;"></canvas>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 11px; color: #6b7280;">Current Rate</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="currentDetRate">0/min</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                            <span style="font-size: 11px; color: #6b7280;">Peak Today</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="peakToday">0</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="font-size: 11px; color: #6b7280;">Total Today</span>
                            <span style="font-size: 11px; font-weight: 600; color: #1a1a1a;" id="totalToday">0</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- System Information -->
            <div class="panel">
                <div class="panel-header">System Information</div>
                <div class="system-info-item">
                    <span class="info-label">AI Engine</span>
                    <span class="info-value" id="aiEngine">YOLO v8</span>
                </div>
                <div class="system-info-item">
                    <span class="info-label">Detection Rate</span>
                    <span class="info-value" id="detectionRate">0.0/min</span>
                </div>
                <div class="system-info-item">
                    <span class="info-label">Peak Detections</span>
                    <span class="info-value" id="peakDetections">0</span>
                </div>
                <div class="system-info-item">
                    <span class="info-label">Total Violations</span>
                    <span class="info-value" id="totalViolations">0</span>
                </div>
            </div>
            
            <!-- Activity Log -->
            <div class="panel">
                <div class="panel-header">Activity Log</div>
                <div class="activity-log" id="activityLog">
                    <div class="log-entry">
                        <div class="log-time">System Initialized</div>
                        <div class="log-message">All systems operational</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    </div>
    
    <!-- CAMERAS TAB -->
    <div id="tab-cameras" class="tab-content">
        <div style="padding: 20px; max-width: 1920px; margin: 0 auto;">
            <div id="cameraDetailView">
                <!-- Camera details will be loaded here -->
            </div>
        </div>
    </div>
    
    <!-- ZONES TAB -->
    <div id="tab-zones" class="tab-content">
        <div style="padding: 20px; max-width: 1920px; margin: 0 auto;">
            <div class="panel">
                <div class="panel-header">Zone Management</div>
                
                <div class="setting-item">
                    <label class="setting-label">Select Camera</label>
                    <select id="zoneCamera" class="setting-input" onchange="loadZonesForCamera()">
                        <option value="">Select a camera...</option>
                    </select>
                </div>
                
                <div id="zoneManagement" style="display: none;">
                    <div class="zone-tools">
                        <button class="zone-tool-btn" onclick="startDrawing('polygon')">Draw Polygon</button>
                        <button class="zone-tool-btn" onclick="startDrawing('rectangle')">Draw Rectangle</button>
                        <button class="zone-tool-btn" onclick="startDrawing('circle')">Draw Circle</button>
                        <button class="zone-tool-btn" onclick="clearDrawing()">Clear</button>
                        <button class="zone-tool-btn" onclick="savePolygonZone()">Save Zone</button>
                    </div>
                    
                    <div class="setting-item">
                        <label class="setting-label">Zone Name</label>
                        <input type="text" id="zoneName" class="setting-input" placeholder="Enter zone name...">
                    </div>
                    
                    <div class="setting-item">
                        <label class="setting-label">Zone Type</label>
                        <select id="zoneType" class="setting-input">
                            <option value="restricted">Restricted (Red)</option>
                            <option value="warning">Warning (Yellow)</option>
                            <option value="safe">Safe (Green)</option>
                        </select>
                    </div>
                    
                    <div class="zone-canvas-container">
                        <img id="zonePreviewImage" style="max-width: 100%; display: block;">
                        <canvas id="zoneCanvas" class="zone-canvas"></canvas>
                    </div>
                    
                    <div class="zone-list" id="zoneList">
                        <!-- Zones will be listed here -->
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- SETTINGS TAB -->
    <div id="tab-settings" class="tab-content" style="overflow-y: auto; max-height: calc(100vh - 140px);">
        <div style="padding: 20px; max-width: 800px; margin: 0 auto; padding-bottom: 40px;">
            <div class="panel">
                <div class="panel-header">Detection Settings</div>
                
                <div class="setting-item">
                    <label class="setting-label">Detection Confidence Threshold</label>
                    <input type="range" id="confThreshold" class="setting-slider" min="0" max="100" value="50" oninput="updateConfidence(this.value)">
                    <div style="color: #00d4ff; margin-top: 5px; font-size: 14px;">Current: <span id="confValue">50</span>%</div>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Zone Transparency</label>
                    <input type="range" id="zoneTransparency" class="setting-slider" min="0" max="100" value="30" oninput="updateTransparency(this.value)">
                    <div style="color: #00d4ff; margin-top: 5px; font-size: 14px;">Current: <span id="transValue">30</span>%</div>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Alert Threshold (Detections)</label>
                    <input type="number" id="alertThreshold" class="setting-input" value="3" min="1" max="50">
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">
                        <input type="checkbox" id="emailAlerts" class="setting-checkbox"> Enable Email Alerts
                    </label>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">
                        <input type="checkbox" id="recordingEnabled" class="setting-checkbox"> Enable Auto Recording
                    </label>
                </div>
                
                <button class="control-button" onclick="saveSettings()">Save Settings</button>
            </div>
            
            <!-- DISPLAY MODE SETTINGS -->
            <div class="panel" style="margin-top: 20px;">
                <div class="panel-header" style="background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);">
                    🖥️ Display Mode Settings
                </div>
                
                <div style="padding: 15px; background: rgba(99, 102, 241, 0.1); border-radius: 8px; margin-bottom: 15px;">
                    <p style="color: #6366f1; margin: 0; font-size: 14px;">
                        <strong>Feature:</strong> Adjust the display layout to fit your screen size. 
                        Ultra Compact mode is optimized for 1024x600 displays showing all 4 cameras.
                    </p>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Display Mode</label>
                    <select id="displayModeSelect" class="setting-input" onchange="setDisplayMode(this.value)">
                        <option value="normal">🖥️ Normal (1920x1080+)</option>
                        <option value="compact">📱 Compact (1280x720)</option>
                        <option value="ultra">📺 Ultra Compact (1024x600)</option>
                    </select>
                </div>
                
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px;">
                    <div style="flex: 1; min-width: 200px; padding: 12px; background: rgba(16, 185, 129, 0.2); border-radius: 8px;">
                        <div style="color: #10b981; font-weight: bold; margin-bottom: 8px;">🖥️ Normal Mode</div>
                        <div style="color: #888; font-size: 12px;">• Full dashboard with side panels</div>
                        <div style="color: #888; font-size: 12px;">• All controls visible</div>
                        <div style="color: #888; font-size: 12px;">• Best for desktop monitors</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; padding: 12px; background: rgba(59, 130, 246, 0.2); border-radius: 8px;">
                        <div style="color: #3b82f6; font-weight: bold; margin-bottom: 8px;">📱 Compact Mode</div>
                        <div style="color: #888; font-size: 12px;">• Smaller UI elements</div>
                        <div style="color: #888; font-size: 12px;">• Hides side panels</div>
                        <div style="color: #888; font-size: 12px;">• Best for 1280x720 screens</div>
                    </div>
                    <div style="flex: 1; min-width: 200px; padding: 12px; background: rgba(139, 92, 246, 0.2); border-radius: 8px;">
                        <div style="color: #8b5cf6; font-weight: bold; margin-bottom: 8px;">📺 Ultra Compact Mode</div>
                        <div style="color: #888; font-size: 12px;">• Maximum space efficiency</div>
                        <div style="color: #888; font-size: 12px;">• All 4 cameras visible</div>
                        <div style="color: #888; font-size: 12px;">• Perfect for 1024x600 displays</div>
                    </div>
                </div>
                
                <div class="setting-item" style="margin-top: 15px;">
                    <label class="setting-label">
                        <input type="checkbox" id="autoDisplayMode" class="setting-checkbox" onchange="toggleAutoDisplayMode(this.checked)"> 
                        Auto-detect display mode based on screen size
                    </label>
                </div>
            </div>
            
            <!-- BARRIER DETECTION SETTINGS -->
            <div class="panel" style="margin-top: 20px;">
                <div class="panel-header" style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);">
                    ⚠️ Industrial Barrier Detection
                </div>
                
                <div style="padding: 15px; background: rgba(245, 158, 11, 0.1); border-radius: 8px; margin-bottom: 15px;">
                    <p style="color: #f59e0b; margin: 0; font-size: 14px;">
                        <strong>Feature:</strong> Automatically detects red and yellow industrial safety barriers 
                        and ignores persons detected behind them. This helps reduce false positives in industrial environments.
                    </p>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label" style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="barrierEnabled" class="setting-checkbox" checked onchange="toggleBarrierDetection(this.checked)"> 
                        <span>Enable Barrier Detection</span>
                        <span id="barrierStatus" style="padding: 4px 12px; border-radius: 12px; font-size: 12px; background: #10b981; color: white;">ACTIVE</span>
                    </label>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Minimum Barrier Size (pixels²)</label>
                    <input type="range" id="barrierMinArea" class="setting-slider" min="500" max="10000" value="2000" oninput="updateBarrierSetting('min_area', this.value)">
                    <div style="color: #f59e0b; margin-top: 5px; font-size: 14px;">Current: <span id="barrierMinValue">2000</span> px²</div>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Detection Sensitivity</label>
                    <select id="barrierSensitivity" class="setting-input" onchange="setBarrierSensitivity(this.value)">
                        <option value="low">Low (fewer detections, more accurate)</option>
                        <option value="medium" selected>Medium (balanced)</option>
                        <option value="high">High (more detections, may have false positives)</option>
                    </select>
                </div>
                
                <div style="display: flex; gap: 10px; flex-wrap: wrap; margin-top: 15px;">
                    <div style="flex: 1; min-width: 150px; padding: 12px; background: rgba(239, 68, 68, 0.2); border-radius: 8px; text-align: center;">
                        <div style="color: #ef4444; font-weight: bold;">🔴 Red Barriers</div>
                        <div style="color: #888; font-size: 12px; margin-top: 5px;">Detects solid red barriers</div>
                    </div>
                    <div style="flex: 1; min-width: 150px; padding: 12px; background: rgba(245, 158, 11, 0.2); border-radius: 8px; text-align: center;">
                        <div style="color: #f59e0b; font-weight: bold;">🟡 Yellow Barriers</div>
                        <div style="color: #888; font-size: 12px; margin-top: 5px;">Detects yellow warning barriers</div>
                    </div>
                    <div style="flex: 1; min-width: 150px; padding: 12px; background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(245, 158, 11, 0.2)); border-radius: 8px; text-align: center;">
                        <div style="color: #fb923c; font-weight: bold;">🚧 Striped Barriers</div>
                        <div style="color: #888; font-size: 12px; margin-top: 5px;">Detects red/yellow striped</div>
                    </div>
                </div>
            </div>
            
            <!-- AUDIO ALARM SETTINGS -->
            <div class="panel" style="margin-top: 20px;">
                <div class="panel-header" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                    🔊 Audio Alarm System
                </div>
                
                <div style="padding: 15px; background: rgba(239, 68, 68, 0.1); border-radius: 8px; margin-bottom: 15px;">
                    <p style="color: #ef4444; margin: 0; font-size: 14px;">
                        <strong>Feature:</strong> Plays audio alerts when zone violations are detected.
                        Configure volume, sound type, and cooldown period between alarms.
                    </p>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label" style="display: flex; align-items: center; gap: 10px;">
                        <input type="checkbox" id="audioEnabled" class="setting-checkbox" checked onchange="toggleAudioAlarm(this.checked)"> 
                        <span>Enable Audio Alarms</span>
                        <span id="audioStatus" style="padding: 4px 12px; border-radius: 12px; font-size: 12px; background: #10b981; color: white;">ACTIVE</span>
                    </label>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Alarm Volume</label>
                    <input type="range" id="audioVolume" class="setting-slider" min="0" max="100" value="70" oninput="updateAudioVolume(this.value)">
                    <div style="color: #ef4444; margin-top: 5px; font-size: 14px;">Volume: <span id="volumeValue">70</span>%</div>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Alarm Sound Type</label>
                    <select id="audioSoundType" class="setting-input" onchange="updateAudioSoundType(this.value)">
                        <option value="alert">🚨 Alert (Standard)</option>
                        <option value="siren">🚑 Siren (Loud)</option>
                        <option value="beep">🔔 Beep (Subtle)</option>
                        <option value="chime">🎵 Chime (Gentle)</option>
                    </select>
                </div>
                
                <div class="setting-item">
                    <label class="setting-label">Cooldown Period (seconds)</label>
                    <input type="number" id="audioCooldown" class="setting-input" value="10" min="1" max="60" onchange="updateAudioCooldown(this.value)">
                    <div style="color: #888; margin-top: 5px; font-size: 12px;">Time before same zone can trigger alarm again</div>
                </div>
                
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button class="control-button" onclick="testAudioAlarm('alert')" style="background: #ef4444;">
                        🔊 Test Alert
                    </button>
                    <button class="control-button" onclick="testAudioAlarm('siren')" style="background: #f97316;">
                        🚑 Test Siren
                    </button>
                    <button class="control-button" onclick="testAudioAlarm('beep')" style="background: #6366f1;">
                        🔔 Test Beep
                    </button>
                </div>
            </div>
            
            <!-- PDF REPORTS SETTINGS -->
            <div class="panel" style="margin-top: 20px;">
                <div class="panel-header" style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);">
                    📊 Reports & Analytics
                </div>
                
                <div style="padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; margin-bottom: 15px;">
                    <p style="color: #3b82f6; margin: 0; font-size: 14px;">
                        <strong>Feature:</strong> Generate professional PDF reports with violation statistics,
                        charts, and detailed event logs. Export reports for compliance and record-keeping.
                    </p>
                </div>
                
                <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px;">
                    <button class="control-button" onclick="generateReport('daily')" style="background: linear-gradient(135deg, #10b981, #059669); flex: 1; min-width: 150px;">
                        📄 Daily Report
                    </button>
                    <button class="control-button" onclick="generateReport('weekly')" style="background: linear-gradient(135deg, #3b82f6, #1d4ed8); flex: 1; min-width: 150px;">
                        📋 Weekly Report
                    </button>
                    <button class="control-button" onclick="generateReport('monthly')" style="background: linear-gradient(135deg, #8b5cf6, #6d28d9); flex: 1; min-width: 150px;">
                        📑 Monthly Report
                    </button>
                </div>
                
                <div class="panel-header" style="background: #374151; font-size: 14px;">
                    📁 Generated Reports
                </div>
                <div id="reportsList" style="max-height: 200px; overflow-y: auto; padding: 10px; background: #1a1a2e;">
                    <div style="color: #888; text-align: center; padding: 20px;">
                        Loading reports...
                    </div>
                </div>
                
                <button class="control-button" onclick="loadReportsList()" style="margin-top: 15px; background: #6b7280;">
                    🔄 Refresh Reports List
                </button>
            </div>
            
            <div class="panel" style="margin-top: 20px;">
                <div class="panel-header">Camera Configuration</div>
                <div id="cameraSettings">
                    <!-- Camera settings will be loaded here -->
                </div>
            </div>
        </div>
    </div>
    
    <!-- EVENTS TAB -->
    <div id="tab-events" class="tab-content">
        <div style="padding: 20px; max-width: 1200px; margin: 0 auto;">
            <div class="panel">
                <div class="panel-header">Event History</div>
                <div id="eventHistory" style="max-height: 600px; overflow-y: auto;">
                    <!-- Events will be loaded here -->
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let camerasInitialized = false;
        let systemEnabled = true;
        let detectionHistory = [];
        let systemParamsHistory = { cpu: [], mem: [], temp: [] };
        let detectionChart, systemParamsChart;
        
        // Initialize charts
        function initCharts() {
            // Detection Chart
            const detCanvas = document.getElementById('detectionChart');
            const detCtx = detCanvas.getContext('2d');
            detectionChart = { canvas: detCanvas, ctx: detCtx };
            
            // System Parameters Chart
            const sysCanvas = document.getElementById('systemParamsChart');
            const sysCtx = sysCanvas.getContext('2d');
            systemParamsChart = { canvas: sysCanvas, ctx: sysCtx };
            
            // Initialize with empty data
            for (let i = 0; i < 30; i++) {
                detectionHistory.push(0);
                systemParamsHistory.cpu.push(0);
                systemParamsHistory.mem.push(0);
                systemParamsHistory.temp.push(0);
            }
        }
        
        function drawDetectionChart() {
            const { canvas, ctx } = detectionChart;
            const width = canvas.width = canvas.offsetWidth;
            const height = canvas.height = canvas.offsetHeight;
            
            ctx.clearRect(0, 0, width, height);
            
            const maxVal = Math.max(...detectionHistory, 1);
            const padding = 30;
            const chartWidth = width - padding * 2;
            const chartHeight = height - padding * 2;
            const step = chartWidth / (detectionHistory.length - 1);
            
            // Draw grid
            ctx.strokeStyle = '#e5e7eb';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding + (chartHeight / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }
            
            // Draw line
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.beginPath();
            detectionHistory.forEach((val, idx) => {
                const x = padding + step * idx;
                const y = padding + chartHeight - (val / maxVal) * chartHeight;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Fill area
            ctx.lineTo(width - padding, height - padding);
            ctx.lineTo(padding, height - padding);
            ctx.closePath();
            ctx.fillStyle = 'rgba(16, 185, 129, 0.1)';
            ctx.fill();
        }
        
        function drawSystemParamsChart() {
            const { canvas, ctx } = systemParamsChart;
            const width = canvas.width = canvas.offsetWidth;
            const height = canvas.height = canvas.offsetHeight;
            
            ctx.clearRect(0, 0, width, height);
            
            const padding = 25;
            const chartWidth = width - padding * 2;
            const chartHeight = height - padding * 2;
            const step = chartWidth / (systemParamsHistory.cpu.length - 1);
            
            // Draw grid
            ctx.strokeStyle = '#e5e7eb';
            ctx.lineWidth = 1;
            for (let i = 0; i <= 4; i++) {
                const y = padding + (chartHeight / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(width - padding, y);
                ctx.stroke();
            }
            
            // Draw CPU line
            ctx.strokeStyle = '#10b981';
            ctx.lineWidth = 2;
            ctx.beginPath();
            systemParamsHistory.cpu.forEach((val, idx) => {
                const x = padding + step * idx;
                const y = padding + chartHeight - (val / 100) * chartHeight;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Draw Memory line
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            systemParamsHistory.mem.forEach((val, idx) => {
                const x = padding + step * idx;
                const y = padding + chartHeight - (val / 100) * chartHeight;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            
            // Draw Temp line
            ctx.strokeStyle = '#ef4444';
            ctx.lineWidth = 2;
            ctx.beginPath();
            systemParamsHistory.temp.forEach((val, idx) => {
                const x = padding + step * idx;
                const y = padding + chartHeight - (val / 100) * chartHeight;
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
        }
        
        async function shutdownServer() {
            if (!confirm('Are you sure you want to shutdown the VIGIL server?\\n\\nThis will stop all camera feeds and detection.')) {
                return;
            }
            
            try {
                document.getElementById('systemStatus').textContent = 'SHUTTING DOWN';
                document.getElementById('systemStatus').style.background = '#dc2626';
                
                const response = await fetch('/api/system/shutdown', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (response.ok) {
                    addLogEntry('Server shutdown initiated');
                    alert('Server is shutting down. Please close this browser tab.');
                }
            } catch (error) {
                console.error('Error shutting down server:', error);
                // Server may have already shut down
                alert('Server has been shut down.');
            }
        }
        
        async function toggleSystemPower(enabled) {
            systemEnabled = enabled;
            try {
                const response = await fetch('/api/system/power', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: enabled })
                });
                
                if (response.ok) {
                    addLogEntry(`System ${enabled ? 'enabled' : 'disabled'}`);
                    document.getElementById('systemStatus').textContent = enabled ? 'OPERATIONAL' : 'STANDBY';
                    document.getElementById('systemStatus').style.background = enabled ? '#10b981' : '#6b7280';
                }
            } catch (error) {
                console.error('Error toggling system power:', error);
            }
        }
        
        async function updateSystemParams() {
            try {
                const response = await fetch('/api/system/params');
                const data = await response.json();
                
                // Update displays
                document.getElementById('cpuValue').textContent = data.cpu + '%';
                document.getElementById('memValue').textContent = data.memory + '%';
                document.getElementById('tempValue').textContent = data.temperature + '\u00b0C';
                
                // Update bars
                document.getElementById('cpuBar').style.width = data.cpu + '%';
                document.getElementById('memBar').style.width = data.memory + '%';
                document.getElementById('tempBar').style.width = (data.temperature / 100 * 100) + '%';
                
                // Update history
                systemParamsHistory.cpu.push(data.cpu);
                systemParamsHistory.mem.push(data.memory);
                systemParamsHistory.temp.push(data.temperature);
                
                if (systemParamsHistory.cpu.length > 30) {
                    systemParamsHistory.cpu.shift();
                    systemParamsHistory.mem.shift();
                    systemParamsHistory.temp.shift();
                }
                
                drawSystemParamsChart();
            } catch (error) {
                console.error('Error updating system params:', error);
            }
        }
        
        function updateDetectionChart(totalDetections) {
            detectionHistory.push(totalDetections);
            if (detectionHistory.length > 30) {
                detectionHistory.shift();
            }
            drawDetectionChart();
            
            // Update analytics stats
            const rate = detectionHistory.slice(-5).reduce((a, b) => a + b, 0) / 5;
            document.getElementById('currentDetRate').textContent = rate.toFixed(1) + '/min';
            document.getElementById('peakToday').textContent = Math.max(...detectionHistory);
            document.getElementById('totalToday').textContent = totalDetections;
        }
        
        // Initialize charts on load
        window.addEventListener('load', () => {
            initCharts();
            drawDetectionChart();
            drawSystemParamsChart();
        });
        
        // Update system params every 3 seconds
        setInterval(updateSystemParams, 3000);
        
        // Initialize cameras on page load
        async function initializeCameras() {
            try {
                const response = await fetch('/api/cameras');
                const data = await response.json();
                const grid = document.getElementById('cameraGrid');
                grid.innerHTML = '';
                
                // Load zones for all cameras
                for (const camera of data.cameras) {
                    await loadCameraZones(camera.id);
                }
                
                data.cameras.forEach(camera => {
                    const card = document.createElement('div');
                    card.className = 'camera-card';
                    card.innerHTML = `
                        <div class="camera-header">
                            <div class="camera-title">Camera ${camera.id} - ${camera.position}</div>
                            <div class="camera-status">
                                <span class="status-dot"></span>
                                <span>ACTIVE</span>
                            </div>
                        </div>
                        <div class="camera-feed" style="position: relative;">
                            <img id="cameraImg${camera.id}" src="/api/camera/${camera.id}/stream" alt="Camera ${camera.id} Stream" style="display: block; width: 100%; height: 100%; object-fit: cover;">
                            <canvas id="zoneCanvas${camera.id}" class="zone-canvas" style="display: none; position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; pointer-events: all;"></canvas>
                        </div>
                        <div class="camera-controls" style="padding: 10px !important; display: flex !important; gap: 8px !important; flex-wrap: wrap !important; background: #ffffff !important; border-top: 2px solid #e5e7eb !important;">
                            <button class="control-btn" id="detectBtn${camera.id}" onclick="toggleDetection(${camera.id})" style="flex: 1 !important; min-width: 90px !important; padding: 8px 12px !important; background: #10b981 !important; color: white !important; border: none !important; border-radius: 6px !important; font-size: 12px !important; font-weight: 600 !important; cursor: pointer !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;">
                                <span id="detectLabel${camera.id}">⏸ Stop Detect</span>
                            </button>
                            <button class="control-btn" id="recordBtn${camera.id}" onclick="toggleRecording(${camera.id})" style="flex: 1 !important; min-width: 90px !important; padding: 8px 12px !important; background: #3b82f6 !important; color: white !important; border: none !important; border-radius: 6px !important; font-size: 12px !important; font-weight: 600 !important; cursor: pointer !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;">
                                <span id="recordLabel${camera.id}">● Start Rec</span>
                            </button>
                            <button class="control-btn" onclick="toggleZoneDrawing(${camera.id})" style="flex: 1 !important; min-width: 90px !important; padding: 8px 12px !important; background: #8b5cf6 !important; color: white !important; border: none !important; border-radius: 6px !important; font-size: 12px !important; font-weight: 600 !important; cursor: pointer !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;">
                                📐 Draw Zone
                            </button>
                        </div>
                        <div class="camera-stats">
                            <div class="camera-stat-item">
                                <div class="camera-stat-label">Detections</div>
                                <div class="camera-stat-value" id="cam${camera.id}-det">0</div>
                            </div>
                            <div class="camera-stat-item">
                                <div class="camera-stat-label">Violations</div>
                                <div class="camera-stat-value" id="cam${camera.id}-vio">0</div>
                            </div>
                            <div class="camera-stat-item">
                                <div class="camera-stat-label">FPS</div>
                                <div class="camera-stat-value" id="cam${camera.id}-fps">0</div>
                            </div>
                        </div>
                    `;
                    grid.appendChild(card);
                    console.log('Camera card created for camera:', camera.id);
                    console.log('Control buttons HTML:', card.querySelector('.camera-controls'));
                });
                
                camerasInitialized = true;
                addLogEntry(`${data.cameras.length} camera(s) initialized successfully`);
            } catch (error) {
                console.error('Error initializing cameras:', error);
                addLogEntry('Error loading cameras');
            }
        }
        
        // Update stats every 2 seconds
        setInterval(updateStats, 2000);
        
        async function updateStats() {
            try {
                const cameraResponse = await fetch('/api/cameras');
                const cameraData = await cameraResponse.json();
                
                const systemResponse = await fetch('/api/system/stats');
                const systemData = await systemResponse.json();
                
                // Update overview stats
                document.getElementById('totalDetections').textContent = systemData.total_detections;
                document.getElementById('activeCameras').textContent = systemData.active_cameras;
                document.getElementById('avgFPS').textContent = systemData.average_fps.toFixed(1);
                document.getElementById('uptime').textContent = systemData.uptime_formatted;
                
                // Update system info
                document.getElementById('detectionRate').textContent = systemData.detection_rate_per_minute.toFixed(1) + '/min';
                document.getElementById('peakDetections').textContent = systemData.peak_detections;
                document.getElementById('totalViolations').textContent = systemData.total_violations;
                document.getElementById('aiEngine').textContent = systemData.yolo_available ? 'YOLO v8' : 'Disabled';
                
                // Update camera stats
                cameraData.cameras.forEach(camera => {
                    const detEl = document.getElementById(`cam${camera.id}-det`);
                    const vioEl = document.getElementById(`cam${camera.id}-vio`);
                    const fpsEl = document.getElementById(`cam${camera.id}-fps`);
                    if (detEl) detEl.textContent = camera.detections;
                    if (vioEl) vioEl.textContent = camera.violations;
                    if (fpsEl) fpsEl.textContent = camera.fps;
                });
                
                const totalDet = systemData.total_detections;
                if (window.lastDetectionCount !== totalDet) {
                    window.lastDetectionCount = totalDet;
                    addLogEntry(`Detection count updated: ${totalDet} total`);
                }
                
                // Update detection chart
                updateDetectionChart(systemData.total_detections);
                
                // Check for new violations
                if (systemData.total_violations > lastViolationCount) {
                    const newViolations = systemData.total_violations - lastViolationCount;
                    lastViolationCount = systemData.total_violations;
                    
                    // Trigger violation alert - both old sound and new audio alarm
                    playSound('violation');
                    
                    // Trigger audio alarm system
                    const soundType = document.getElementById('audioSoundType')?.value || 'alert';
                    if (audioEnabled && typeof playAlarmSound === 'function') {
                        playAlarmSound(soundType, audioVolume);
                    }
                    
                    showNotification('🚨 Zone Violation', `${newViolations} new violation(s) detected!`, 'violation');
                    
                    // Flash affected camera cards
                    cameraData.cameras.forEach(camera => {
                        if (camera.violations > 0) {
                            flashViolationAlert(camera.id);
                        }
                    });
                }
                
                // Check tamper status (piggyback on stats update - no extra polling)
                checkTamperStatus();
                
            } catch (error) {
                console.error('Error updating stats:', error);
            }
        }
        
        function addLogEntry(message) {
            const logContainer = document.getElementById('activityLog');
            const timestamp = new Date().toLocaleTimeString();
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <div class="log-time">${timestamp}</div>
                <div class="log-message">${message}</div>
            `;
            logContainer.insertBefore(entry, logContainer.firstChild);
            
            while (logContainer.children.length > 20) {
                logContainer.removeChild(logContainer.lastChild);
            }
        }
        
        function refreshSystem() {
            location.reload();
        }
        
        // Toggle detection for a camera
        async function toggleDetection(cameraId) {
            const btn = document.getElementById(`detectBtn${cameraId}`);
            const label = document.getElementById(`detectLabel${cameraId}`);
            const isEnabled = label.textContent.includes('Stop');
            
            try {
                const response = await fetch(`/api/camera/${cameraId}/detection`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: !isEnabled })
                });
                
                if (response.ok) {
                    if (isEnabled) {
                        label.textContent = '▶ Start Detect';
                        btn.style.background = '#6b7280';
                    } else {
                        label.textContent = '⏸ Stop Detect';
                        btn.style.background = '#10b981';
                    }
                    addLogEntry(`Camera ${cameraId} detection ${isEnabled ? 'disabled' : 'enabled'}`);
                }
            } catch (error) {
                console.error('Error toggling detection:', error);
            }
        }
        
        // Toggle recording for a camera - NOW SAVES TO MP4
        async function toggleRecording(cameraId) {
            const btn = document.getElementById(`recordBtn${cameraId}`);
            const label = document.getElementById(`recordLabel${cameraId}`);
            const isRecording = label.textContent.includes('Stop');
            
            try {
                const response = await fetch(`/api/camera/${cameraId}/recording`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ recording: !isRecording })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    if (isRecording) {
                        // Stopped recording
                        label.textContent = '● Start Rec';
                        btn.style.background = '#3b82f6';
                        btn.classList.remove('recording-active');
                        const info = data.recording_info || {};
                        if (info.filename) {
                            addLogEntry(`Camera ${cameraId} recording SAVED: ${info.filename} (${info.frame_count} frames, ${(info.duration || 0).toFixed(1)}s)`);
                            showNotification('📹 Recording Saved', `${info.filename}`, 'success');
                        } else {
                            addLogEntry(`Camera ${cameraId} recording stopped`);
                        }
                    } else {
                        // Started recording
                        label.textContent = '■ Stop Rec';
                        btn.style.background = '#ef4444';
                        btn.classList.add('recording-active');
                        const info = data.recording_info || {};
                        addLogEntry(`Camera ${cameraId} recording STARTED: ${info.filename || 'recording...'}`);
                        showNotification('🔴 Recording Started', `Camera ${cameraId} now recording to MP4`, 'info');
                    }
                } else {
                    const errorData = await response.json();
                    addLogEntry(`Recording error: ${errorData.detail || 'Unknown error'}`);
                    showNotification('❌ Recording Error', errorData.detail || 'Failed to toggle recording', 'error');
                }
            } catch (error) {
                console.error('Error toggling recording:', error);
                addLogEntry(`Recording error: ${error.message}`);
            }
        }
        
        // Check tamper status periodically
        async function checkTamperStatus() {
            try {
                const response = await fetch('/api/tamper/status');
                if (response.ok) {
                    const data = await response.json();
                    for (const [cameraId, status] of Object.entries(data.tamper_status)) {
                        const card = document.querySelector(`#cameraImg${cameraId}`)?.closest('.camera-card');
                        if (status.is_tampered && status.warning_sent) {
                            // Add tamper warning class to camera card
                            if (card && !card.classList.contains('tamper-warning')) {
                                card.classList.add('tamper-warning');
                                // Show tamper warning notification only once
                                showNotification(
                                    `⚠️ TAMPER ALERT: Camera ${cameraId}`,
                                    `${status.camera_name}: ${status.tamper_type}`,
                                    'tamper'
                                );
                                addLogEntry(`⚠️ TAMPER: Camera ${cameraId} - ${status.tamper_type}`);
                            }
                        } else {
                            // Remove tamper warning class
                            if (card) card.classList.remove('tamper-warning');
                        }
                    }
                }
            } catch (error) {
                console.error('Error checking tamper status:', error);
            }
        }
        
        // Tamper status is checked as part of updateStats (every 2 sec) - no separate polling needed
        // The backend tamper detection runs in the camera capture thread and only alerts after 10+ seconds of blocking
        
        // Toggle zone drawing for a camera
        let activeZoneCanvas = null;
        let drawingRect = null;
        let cameraZones = {};  // Store zones per camera
        let selectedZone = null;  // Track selected zone for editing
        let hoveredZone = null;  // Track hovered zone for UI feedback
        let lastViolationCount = 0;  // Track violations for alerts
        let violationSound = null;
        let isDraggingZone = false;  // Track if dragging a zone
        let isResizingZone = false;  // Track if resizing a zone
        let dragStartX = 0;  // Drag start X
        let dragStartY = 0;  // Drag start Y
        let zoneStartRect = null;  // Original zone rect
        let resizeHandle = null;  // Which handle is being dragged (tl, tr, bl, br, t, b, l, r)
        
        // Initialize sound
        function initSound() {
            violationSound = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBzGH0fPTgjMGHm7A7+OZUQ0PVKzn7bdnGws+ltryxnMpBSuBzvLZizkIGGS56+OgTwwOUKXi77RkHAU4jtXxxXElBStty+/glUsNCVCn4++2Yh0FN4/V8sl3KwUse8jv3ZJBDA==');
        }
        
        function playSound(type) {
            if (type === 'violation' && violationSound) {
                violationSound.play().catch(e => console.log('Sound play failed:', e));
            } else if (type === 'success') {
                // Success sound - simple beep
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                gainNode.gain.value = 0.1;
                
                oscillator.start();
                oscillator.stop(audioContext.currentTime + 0.1);
            }
        }
        
        // Toggle touch-friendly mode
        function toggleTouchMode() {
            document.body.classList.toggle('touch-mode');
            const toggle = document.getElementById('touchModeToggle');
            const isTouch = document.body.classList.contains('touch-mode');
            
            // Update toggle appearance
            if (toggle) {
                toggle.classList.toggle('active', isTouch);
            }
            
            // Save preference to localStorage
            localStorage.setItem('vigilTouchMode', isTouch ? 'enabled' : 'disabled');
            
            // Show feedback
            addLogEntry(`Touch-friendly mode ${isTouch ? 'enabled' : 'disabled'}`);
        }
        
        // Initialize touch mode from localStorage
        function initTouchMode() {
            const savedMode = localStorage.getItem('vigilTouchMode');
            if (savedMode === 'enabled') {
                document.body.classList.add('touch-mode');
                const toggle = document.getElementById('touchModeToggle');
                if (toggle) {
                    toggle.classList.add('active');
                }
            }
        }
        
        // Display mode functions for different screen sizes
        function setDisplayMode(mode) {
            // Remove all display modes
            document.body.classList.remove('compact-mode', 'ultra-compact-mode');
            
            // Remove active class from all buttons
            document.getElementById('displayNormal').classList.remove('active');
            document.getElementById('displayCompact').classList.remove('active');
            document.getElementById('displayUltra').classList.remove('active');
            
            // Update settings dropdown if it exists
            const dropdown = document.getElementById('displayModeSelect');
            if (dropdown) {
                dropdown.value = mode;
            }
            
            if (mode === 'compact') {
                document.body.classList.add('compact-mode');
                document.getElementById('displayCompact').classList.add('active');
                addLogEntry('Switched to Compact mode (1280x720)');
            } else if (mode === 'ultra') {
                document.body.classList.add('ultra-compact-mode');
                document.getElementById('displayUltra').classList.add('active');
                addLogEntry('Switched to Ultra Compact mode (1024x600)');
            } else {
                document.getElementById('displayNormal').classList.add('active');
                addLogEntry('Switched to Normal display mode');
            }
            
            // Save preference
            localStorage.setItem('vigilDisplayMode', mode);
            localStorage.setItem('vigilAutoDisplayMode', 'false');
            
            // Uncheck auto mode checkbox
            const autoCheckbox = document.getElementById('autoDisplayMode');
            if (autoCheckbox) {
                autoCheckbox.checked = false;
            }
        }
        
        // Toggle auto display mode
        function toggleAutoDisplayMode(enabled) {
            localStorage.setItem('vigilAutoDisplayMode', enabled ? 'true' : 'false');
            if (enabled) {
                // Auto-detect and apply appropriate mode
                if (window.innerWidth <= 1024 || window.innerHeight <= 600) {
                    setDisplayMode('ultra');
                } else if (window.innerWidth <= 1280 || window.innerHeight <= 720) {
                    setDisplayMode('compact');
                } else {
                    setDisplayMode('normal');
                }
                // Re-check the checkbox since setDisplayMode unchecks it
                const autoCheckbox = document.getElementById('autoDisplayMode');
                if (autoCheckbox) {
                    autoCheckbox.checked = true;
                }
                localStorage.setItem('vigilAutoDisplayMode', 'true');
                addLogEntry('Auto display mode enabled - screen size detected');
            }
        }
        
        // Initialize display mode from localStorage
        function initDisplayMode() {
            const savedMode = localStorage.getItem('vigilDisplayMode');
            const autoMode = localStorage.getItem('vigilAutoDisplayMode');
            
            // Restore auto checkbox state
            const autoCheckbox = document.getElementById('autoDisplayMode');
            if (autoCheckbox && autoMode === 'true') {
                autoCheckbox.checked = true;
            }
            
            if (autoMode === 'true' || !savedMode) {
                // Auto-detect screen size and set appropriate mode
                if (window.innerWidth <= 1024 || window.innerHeight <= 600) {
                    setDisplayModeQuiet('ultra');
                } else if (window.innerWidth <= 1280 || window.innerHeight <= 720) {
                    setDisplayModeQuiet('compact');
                } else {
                    setDisplayModeQuiet('normal');
                }
            } else if (savedMode) {
                setDisplayModeQuiet(savedMode);
            } else {
                document.getElementById('displayNormal').classList.add('active');
            }
        }
        
        // Set display mode without logging (for init)
        function setDisplayModeQuiet(mode) {
            document.body.classList.remove('compact-mode', 'ultra-compact-mode');
            document.getElementById('displayNormal').classList.remove('active');
            document.getElementById('displayCompact').classList.remove('active');
            document.getElementById('displayUltra').classList.remove('active');
            
            const dropdown = document.getElementById('displayModeSelect');
            if (dropdown) {
                dropdown.value = mode;
            }
            
            if (mode === 'compact') {
                document.body.classList.add('compact-mode');
                document.getElementById('displayCompact').classList.add('active');
            } else if (mode === 'ultra') {
                document.body.classList.add('ultra-compact-mode');
                document.getElementById('displayUltra').classList.add('active');
            } else {
                document.getElementById('displayNormal').classList.add('active');
            }
        }
        
        function showNotification(title, message, type = 'info') {
            if ('Notification' in window && Notification.permission === 'granted') {
                new Notification(title, {
                    body: message,
                    icon: type === 'violation' ? '🚨' : 'ℹ️',
                    requireInteraction: true
                });
            }
        }
        
        function requestNotificationPermission() {
            if ('Notification' in window && Notification.permission === 'default') {
                Notification.requestPermission();
            }
        }
        
        async function sendEmailAlert(violation) {
            // Email alerts would be sent from backend
            try {
                await fetch('/api/violations/alert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(violation)
                });
            } catch (error) {
                console.error('Failed to send email alert:', error);
            }
        }
        
        function flashViolationAlert(cameraId) {
            const card = document.querySelector(`#cameraImg${cameraId}`).closest('.camera-card');
            if (card) {
                card.style.border = '4px solid #ef4444';
                card.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.6)';
                setTimeout(() => {
                    card.style.border = '';
                    card.style.boxShadow = '';
                }, 2000);
            }
        }
        
        async function loadCameraZones(cameraId) {
            try {
                const response = await fetch(`/api/camera/${cameraId}/zones`);
                const data = await response.json();
                cameraZones[cameraId] = data.zones || [];
            } catch (error) {
                console.error('Failed to load zones:', error);
            }
        }
        
        function toggleZoneDrawing(cameraId) {
            console.log('toggleZoneDrawing called for camera:', cameraId);
            const canvas = document.getElementById(`zoneCanvas${cameraId}`);
            
            if (!canvas) {
                console.error('Canvas not found for camera:', cameraId);
                alert('Canvas not found! Camera ID: ' + cameraId);
                return;
            }
            
            console.log('Canvas found:', canvas);
            
            if (activeZoneCanvas === cameraId) {
                // Hide canvas and reset states
                console.log('Hiding canvas');
                canvas.style.display = 'none';
                activeZoneCanvas = null;
                drawingRect = null;
                selectedZone = null;
                isDraggingZone = false;
                isResizingZone = false;
                resizeHandle = null;
            } else {
                // Show canvas and setup drawing
                if (activeZoneCanvas !== null) {
                    const prevCanvas = document.getElementById(`zoneCanvas${activeZoneCanvas}`);
                    if (prevCanvas) prevCanvas.style.display = 'none';
                }
                console.log('Showing canvas and setting up drawing');
                canvas.style.display = 'block';
                canvas.style.pointerEvents = 'auto';
                canvas.style.zIndex = '10';
                activeZoneCanvas = cameraId;
                setupZoneCanvas(canvas, cameraId);
            }
        }
        
        function setupZoneCanvas(canvas, cameraId) {
            console.log('setupZoneCanvas called for camera:', cameraId);
            const img = document.getElementById(`cameraImg${cameraId}`);
            
            if (!img) {
                console.error('Image not found for camera:', cameraId);
                return;
            }
            
            console.log('Image found, dimensions:', img.offsetWidth, 'x', img.offsetHeight);
            
            // Wait for image to be fully loaded
            if (img.complete && img.naturalWidth > 0) {
                console.log('Image already loaded, initializing canvas');
                initCanvas();
            } else {
                console.log('Waiting for image to load');
                img.onload = initCanvas;
            }
            
            function initCanvas() {
                console.log('initCanvas called');
                // Get the parent container dimensions
                const parent = canvas.parentElement;
                canvas.width = parent.offsetWidth;
                canvas.height = parent.offsetHeight;
                canvas.style.width = parent.offsetWidth + 'px';
                canvas.style.height = parent.offsetHeight + 'px';
                
                console.log('Canvas dimensions set to:', canvas.width, 'x', canvas.height);
                
                const ctx = canvas.getContext('2d');
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = 'high';
                drawingRect = null;
                
                // Local variables for new zone drawing
                let startX = 0, startY = 0, isDragging = false;
                
                // Clear any previous drawings
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Add semi-transparent background to show canvas is active
                ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                console.log('Canvas background drawn');
                
                // Draw existing zones
                drawExistingZones(ctx, cameraId);
                
                // Instruction text
                const text = '🖱️ DRAG TO DRAW ZONE';
                
                // Add instruction text with better visibility
                ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                ctx.shadowBlur = 8;
                ctx.shadowOffsetX = 2;
                ctx.shadowOffsetY = 2;
                ctx.fillStyle = '#00ff00';
                ctx.font = 'bold 20px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(text, canvas.width/2, 35);
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
                
                // Rectangle drag handlers with improved accuracy
                canvas.onmousedown = (e) => {
                    e.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const clickX = e.clientX - rect.left;
                    const clickY = e.clientY - rect.top;
                    
                    // Check if clicking on an existing zone
                    const zones = cameraZones[cameraId] || [];
                    let clickedZone = null;
                    let clickedAction = null;
                    
                    for (let i = zones.length - 1; i >= 0; i--) {
                        const zone = zones[i];
                        if (zone.rect && zone.active) {
                            const [x, y, w, h] = zone.rect;
                            
                            // Check if clicking delete button
                            if (clickX >= x + w - 25 && clickX <= x + w - 5 &&
                                clickY >= y + 5 && clickY <= y + 25) {
                                clickedZone = zone;
                                clickedAction = 'delete';
                                break;
                            }
                            
                            // Check if clicking edit button
                            if (clickX >= x + w - 50 && clickX <= x + w - 30 &&
                                clickY >= y + 5 && clickY <= y + 25) {
                                clickedZone = zone;
                                clickedAction = 'edit';
                                break;
                            }
                            
                            // Check resize handles on selected zone (8px hit area)
                            if (selectedZone && selectedZone.id === zone.id) {
                                const handleSize = 8;
                                // Top-left
                                if (Math.abs(clickX - x) <= handleSize && Math.abs(clickY - y) <= handleSize) {
                                    clickedZone = zone;
                                    clickedAction = 'resize-tl';
                                    break;
                                }
                                // Top-right
                                if (Math.abs(clickX - (x + w)) <= handleSize && Math.abs(clickY - y) <= handleSize) {
                                    clickedZone = zone;
                                    clickedAction = 'resize-tr';
                                    break;
                                }
                                // Bottom-left
                                if (Math.abs(clickX - x) <= handleSize && Math.abs(clickY - (y + h)) <= handleSize) {
                                    clickedZone = zone;
                                    clickedAction = 'resize-bl';
                                    break;
                                }
                                // Bottom-right
                                if (Math.abs(clickX - (x + w)) <= handleSize && Math.abs(clickY - (y + h)) <= handleSize) {
                                    clickedZone = zone;
                                    clickedAction = 'resize-br';
                                    break;
                                }
                            }
                            
                            // Check if clicking on zone body
                            if (clickX >= x && clickX <= x + w &&
                                clickY >= y && clickY <= y + h) {
                                clickedZone = zone;
                                clickedAction = 'select';
                                break;
                            }
                        }
                    }
                    
                    if (clickedZone) {
                        console.log('Clicked on zone:', clickedZone.name, 'Action:', clickedAction);
                        
                        if (clickedAction === 'delete') {
                            deleteZoneWithConfirm(cameraId, clickedZone);
                            return;
                        } else if (clickedAction === 'edit') {
                            editExistingZone(cameraId, clickedZone);
                            return;
                        } else if (clickedAction.startsWith('resize-')) {
                            // Start resizing
                            selectedZone = clickedZone;
                            isResizingZone = true;
                            resizeHandle = clickedAction.replace('resize-', '');
                            zoneStartRect = [...clickedZone.rect];
                            dragStartX = clickX;
                            dragStartY = clickY;
                            console.log('Started resizing zone:', resizeHandle);
                            return;
                        } else if (clickedAction === 'select') {
                            // Select zone and enable dragging
                            selectedZone = clickedZone;
                            isDraggingZone = true;
                            zoneStartRect = [...clickedZone.rect];
                            dragStartX = clickX;
                            dragStartY = clickY;
                            console.log('Selected zone for dragging');
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                            ctx.fillRect(0, 0, canvas.width, canvas.height);
                            drawExistingZones(ctx, cameraId);
                            return;
                        }
                    }
                    
                    // No zone clicked, start drawing new zone
                    selectedZone = null;
                    console.log('Mouse down event on canvas - drawing new zone');
                    startX = clickX;
                    startY = clickY;
                    isDragging = true;
                    console.log('Drawing started at:', startX, startY, 'Canvas rect:', rect);
                };
                
                canvas.onmousemove = (e) => {
                    e.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const mouseX = e.clientX - rect.left;
                    const mouseY = e.clientY - rect.top;
                    
                    // Handle zone dragging
                    if (isDraggingZone && selectedZone) {
                        const dx = mouseX - dragStartX;
                        const dy = mouseY - dragStartY;
                        selectedZone.rect = [
                            zoneStartRect[0] + dx,
                            zoneStartRect[1] + dy,
                            zoneStartRect[2],
                            zoneStartRect[3]
                        ];
                        
                        // Redraw
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId, mouseX, mouseY);
                        return;
                    }
                    
                    // Handle zone resizing
                    if (isResizingZone && selectedZone && resizeHandle) {
                        const dx = mouseX - dragStartX;
                        const dy = mouseY - dragStartY;
                        let [x, y, w, h] = zoneStartRect;
                        
                        if (resizeHandle === 'tl') {
                            x += dx; y += dy; w -= dx; h -= dy;
                        } else if (resizeHandle === 'tr') {
                            y += dy; w += dx; h -= dy;
                        } else if (resizeHandle === 'bl') {
                            x += dx; w -= dx; h += dy;
                        } else if (resizeHandle === 'br') {
                            w += dx; h += dy;
                        }
                        
                        // Ensure minimum size
                        if (w >= 30 && h >= 30) {
                            selectedZone.rect = [x, y, w, h];
                        }
                        
                        // Redraw
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId, mouseX, mouseY);
                        return;
                    }
                    
                    if (!isDragging) {
                        // Show hover effects on zones
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId, mouseX, mouseY);
                        
                        // Change cursor based on what's under mouse
                        let cursorStyle = 'crosshair';
                        const zones = cameraZones[cameraId] || [];
                        let overZone = false;
                        for (const zone of zones) {
                            if (zone.rect && zone.active) {
                                const [x, y, w, h] = zone.rect;
                                
                                // Check resize handles
                                if (selectedZone && selectedZone.id === zone.id) {
                                    const handleSize = 8;
                                    if (Math.abs(mouseX - x) <= handleSize && Math.abs(mouseY - y) <= handleSize) {
                                        cursorStyle = 'nwse-resize'; overZone = true; break;
                                    }
                                    if (Math.abs(mouseX - (x + w)) <= handleSize && Math.abs(mouseY - y) <= handleSize) {
                                        cursorStyle = 'nesw-resize'; overZone = true; break;
                                    }
                                    if (Math.abs(mouseX - x) <= handleSize && Math.abs(mouseY - (y + h)) <= handleSize) {
                                        cursorStyle = 'nesw-resize'; overZone = true; break;
                                    }
                                    if (Math.abs(mouseX - (x + w)) <= handleSize && Math.abs(mouseY - (y + h)) <= handleSize) {
                                        cursorStyle = 'nwse-resize'; overZone = true; break;
                                    }
                                }
                                
                                // Check buttons
                                if (mouseX >= x + w - 50 && mouseX <= x + w - 5 && mouseY >= y + 5 && mouseY <= y + 25) {
                                    cursorStyle = 'pointer'; overZone = true; break;
                                }
                                
                                // Check zone body
                                if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
                                    cursorStyle = 'move'; overZone = true; break;
                                }
                            }
                        }
                        canvas.style.cursor = cursorStyle;
                        
                        for (const zone of zones) {
                            if (zone.rect && zone.active) {
                                const [x, y, w, h] = zone.rect;
                                if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
                                    canvas.style.cursor = 'pointer';
                                    overZone = true;
                                    break;
                                }
                            }
                        }
                        canvas.style.cursor = cursorStyle;
                        
                        // Redraw instructions
                        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                        ctx.shadowBlur = 8;
                        ctx.shadowOffsetX = 2;
                        ctx.shadowOffsetY = 2;
                        ctx.fillStyle = '#00ff00';
                        ctx.font = 'bold 20px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText(text, canvas.width/2, 35);
                        ctx.shadowBlur = 0;
                        ctx.shadowOffsetX = 0;
                        ctx.shadowOffsetY = 0;
                        return;
                    }
                    
                    // Drawing new zone
                    const currentX = mouseX;
                    const currentY = mouseY;
                    
                    // Clear and redraw
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Redraw existing zones
                    drawExistingZones(ctx, cameraId);
                    
                    // Draw current rectangle with smooth lines
                    const width = currentX - startX;
                    const height = currentY - startY;
                    
                    // Draw filled rectangle
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.25)';
                    ctx.fillRect(startX, startY, width, height);
                    
                    // Draw border with glow effect
                    ctx.strokeStyle = '#00ff00';
                    ctx.lineWidth = 3;
                    ctx.shadowColor = '#00ff00';
                    ctx.shadowBlur = 10;
                    ctx.strokeRect(startX, startY, width, height);
                    ctx.shadowBlur = 0;
                    
                    // Draw corner markers for better visibility
                    ctx.fillStyle = '#00ff00';
                    const markerSize = 8;
                    ctx.fillRect(startX - markerSize/2, startY - markerSize/2, markerSize, markerSize);
                    ctx.fillRect(startX + width - markerSize/2, startY - markerSize/2, markerSize, markerSize);
                    ctx.fillRect(startX - markerSize/2, startY + height - markerSize/2, markerSize, markerSize);
                    ctx.fillRect(startX + width - markerSize/2, startY + height - markerSize/2, markerSize, markerSize);
                    
                    // Show dimensions
                    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                    ctx.shadowBlur = 4;
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 14px Arial';
                    ctx.textAlign = 'center';
                    const dims = `${Math.abs(width).toFixed(0)} × ${Math.abs(height).toFixed(0)}`;
                    ctx.fillText(dims, startX + width/2, startY + height/2);
                    ctx.shadowBlur = 0;
                    
                    // Redraw instructions
                    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                    ctx.shadowBlur = 8;
                    ctx.shadowOffsetX = 2;
                    ctx.shadowOffsetY = 2;
                    ctx.fillStyle = '#00ff00';
                    ctx.font = 'bold 20px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(text, canvas.width/2, 35);
                    ctx.shadowBlur = 0;
                    ctx.shadowOffsetX = 0;
                    ctx.shadowOffsetY = 0;
                };
                
                canvas.onmouseup = async (e) => {
                    e.preventDefault();
                    
                    // Handle zone drag end - save position
                    if (isDraggingZone && selectedZone) {
                        isDraggingZone = false;
                        console.log('Saving moved zone:', selectedZone.name, selectedZone.rect);
                        
                        // Update zone via API
                        await updateZoneRect(cameraId, selectedZone.id, selectedZone.rect);
                        
                        // Reload zones
                        await loadCameraZones(cameraId);
                        
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                        return;
                    }
                    
                    // Handle zone resize end - save size
                    if (isResizingZone && selectedZone) {
                        isResizingZone = false;
                        resizeHandle = null;
                        console.log('Saving resized zone:', selectedZone.name, selectedZone.rect);
                        
                        // Update zone via API
                        await updateZoneRect(cameraId, selectedZone.id, selectedZone.rect);
                        
                        // Reload zones
                        await loadCameraZones(cameraId);
                        
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                        return;
                    }
                    
                    if (!isDragging) return;
                    isDragging = false;
                    
                    const rect = canvas.getBoundingClientRect();
                    const endX = e.clientX - rect.left;
                    const endY = e.clientY - rect.top;
                    
                    const width = endX - startX;
                    const height = endY - startY;
                    
                    // Check minimum size (30px minimum for better usability)
                    if (Math.abs(width) < 30 || Math.abs(height) < 30) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                        
                        // Show error message
                        ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
                        ctx.shadowBlur = 8;
                        ctx.fillStyle = '#ff0000';
                        ctx.font = 'bold 18px Arial';
                        ctx.textAlign = 'center';
                        ctx.fillText('❌ Zone too small! Minimum 30x30 pixels', canvas.width/2, canvas.height/2);
                        setTimeout(() => initCanvas(), 2000);
                        return;
                    }
                    
                    // Normalize rectangle (handle negative width/height)
                    const x = Math.round(width < 0 ? endX : startX);
                    const y = Math.round(height < 0 ? endY : startY);
                    const w = Math.round(Math.abs(width));
                    const h = Math.round(Math.abs(height));
                    
                    drawingRect = [x, y, w, h];
                    console.log('Zone drawn:', drawingRect);
                    
                    // Show zone configuration dialog
                    await showZoneDialog(cameraId, drawingRect);
                };
                
                canvas.onmouseleave = () => {
                    // Cancel any ongoing operations
                    if (isDragging) {
                        isDragging = false;
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                    }
                    if (isDraggingZone) {
                        isDraggingZone = false;
                        // Revert to original position
                        if (selectedZone && zoneStartRect) {
                            selectedZone.rect = zoneStartRect;
                        }
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                    }
                    if (isResizingZone) {
                        isResizingZone = false;
                        resizeHandle = null;
                        // Revert to original size
                        if (selectedZone && zoneStartRect) {
                            selectedZone.rect = zoneStartRect;
                        }
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                    }
                };
            }
        }
        
        function drawExistingZones(ctx, cameraId, mouseX = null, mouseY = null) {
            const zones = cameraZones[cameraId] || [];
            zones.forEach((zone, idx) => {
                if (zone.rect && zone.rect.length === 4 && zone.active) {
                    const [x, y, w, h] = zone.rect;
                    
                    // Color by type
                    let color = zone.type === 'restricted' ? '#ff0000' : 
                                zone.type === 'warning' ? '#ffff00' : '#00ff00';
                    
                    // Check if mouse is hovering over this zone
                    const isHovered = mouseX !== null && mouseY !== null &&
                                     mouseX >= x && mouseX <= x + w &&
                                     mouseY >= y && mouseY <= y + h;
                    
                    // Check if this zone is selected
                    const isSelected = selectedZone && selectedZone.id === zone.id;
                    
                    // Draw zone with enhanced visuals for hover/select
                    ctx.strokeStyle = color;
                    ctx.lineWidth = isSelected ? 4 : (isHovered ? 3 : 2);
                    ctx.setLineDash(isSelected ? [8, 4] : []);
                    
                    // Fill with more opacity if hovered or selected
                    const opacity = isSelected ? 0.35 : (isHovered ? 0.25 : 0.15);
                    ctx.fillStyle = color + Math.round(opacity * 255).toString(16).padStart(2, '0');
                    ctx.fillRect(x, y, w, h);
                    
                    // Draw border with glow if selected/hovered
                    if (isSelected || isHovered) {
                        ctx.shadowColor = color;
                        ctx.shadowBlur = 10;
                    }
                    ctx.strokeRect(x, y, w, h);
                    ctx.shadowBlur = 0;
                    ctx.setLineDash([]);
                    
                    // Draw label with background
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
                    ctx.fillRect(x + 2, y + 2, ctx.measureText(zone.name).width + 10, 20);
                    ctx.fillStyle = '#ffffff';
                    ctx.font = 'bold 12px Arial';
                    ctx.fillText(zone.name, x + 7, y + 16);
                    
                    // Draw resize handles if selected
                    if (isSelected) {
                        ctx.fillStyle = '#ffffff';
                        ctx.strokeStyle = color;
                        ctx.lineWidth = 2;
                        const handleSize = 8;
                        
                        // Top-left
                        ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize);
                        // Top-right
                        ctx.fillRect(x + w - handleSize/2, y - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(x + w - handleSize/2, y - handleSize/2, handleSize, handleSize);
                        // Bottom-left
                        ctx.fillRect(x - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(x - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                        // Bottom-right
                        ctx.fillRect(x + w - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                        ctx.strokeRect(x + w - handleSize/2, y + h - handleSize/2, handleSize, handleSize);
                    }
                    
                    // Draw edit/delete icons if hovered or selected
                    if (isHovered || isSelected) {
                        // Edit icon (pencil)
                        ctx.fillStyle = '#3b82f6';
                        ctx.fillRect(x + w - 50, y + 5, 20, 20);
                        ctx.fillStyle = '#ffffff';
                        ctx.font = 'bold 14px Arial';
                        ctx.fillText('✏️', x + w - 47, y + 19);
                        
                        // Delete icon (X)
                        ctx.fillStyle = '#ef4444';
                        ctx.fillRect(x + w - 25, y + 5, 20, 20);
                        ctx.fillStyle = '#ffffff';
                        ctx.fillText('🗑️', x + w - 22, y + 19);
                    }
                    
                    // Store zone info for click detection
                    zone._renderBounds = { x, y, w, h, idx };
                }
            });
        }
        
        async function editExistingZone(cameraId, zone) {
            console.log('Editing zone:', zone);
            
            // Show edit dialog
            const result = await showEditZoneDialog(cameraId, zone);
            
            if (result) {
                // Redraw zones
                const canvas = document.getElementById(`zoneCanvas${cameraId}`);
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                drawExistingZones(ctx, cameraId);
            }
        }
        
        async function showEditZoneDialog(cameraId, zone) {
            // Create modal dialog
            const dialog = document.createElement('div');
            dialog.style.cssText = `
                position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                background: white; padding: 25px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                z-index: 10000; min-width: 350px;
            `;
            
            const typeChecked = {
                restricted: zone.type === 'restricted' ? 'checked' : '',
                warning: zone.type === 'warning' ? 'checked' : '',
                safe: zone.type === 'safe' ? 'checked' : ''
            };
            
            dialog.innerHTML = `
                <h3 style="margin: 0 0 20px 0; color: #1f2937; font-size: 20px;">✏️ Edit Zone</h3>
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; color: #4b5563; font-weight: 600;">Zone Name:</label>
                    <input type="text" id="editZoneName" value="${zone.name}" style="width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px;">
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 8px; color: #4b5563; font-weight: 600;">Zone Type:</label>
                    <div style="display: flex; gap: 10px;">
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="editZoneType" value="restricted" ${typeChecked.restricted} style="margin-right: 5px;">
                            <span style="color: #dc2626;">🚫 Restricted</span>
                        </label>
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="editZoneType" value="warning" ${typeChecked.warning} style="margin-right: 5px;">
                            <span style="color: #f59e0b;">⚠️ Warning</span>
                        </label>
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="editZoneType" value="safe" ${typeChecked.safe} style="margin-right: 5px;">
                            <span style="color: #10b981;">✅ Safe</span>
                        </label>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; justify-content: space-between;">
                    <button id="deleteZoneBtn" style="padding: 8px 20px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">🗑️ Delete</button>
                    <div style="display: flex; gap: 10px;">
                        <button id="cancelEditZone" style="padding: 8px 20px; background: #6b7280; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Cancel</button>
                        <button id="saveEditZone" style="padding: 8px 20px; background: #10b981; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">💾 Save Changes</button>
                    </div>
                </div>
            `;
            
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9999;';
            
            document.body.appendChild(overlay);
            document.body.appendChild(dialog);
            
            return new Promise((resolve) => {
                document.getElementById('saveEditZone').onclick = async () => {
                    const name = document.getElementById('editZoneName').value;
                    const type = document.querySelector('input[name="editZoneType"]:checked').value;
                    
                    // Update zone via API
                    await updateZone(cameraId, zone.id, name, type);
                    
                    document.body.removeChild(overlay);
                    document.body.removeChild(dialog);
                    resolve(true);
                };
                
                document.getElementById('deleteZoneBtn').onclick = async () => {
                    if (confirm(`Delete zone "${zone.name}"?`)) {
                        await deleteZone(cameraId, zone.id);
                        document.body.removeChild(overlay);
                        document.body.removeChild(dialog);
                        resolve(true);
                    }
                };
                
                document.getElementById('cancelEditZone').onclick = () => {
                    document.body.removeChild(overlay);
                    document.body.removeChild(dialog);
                    resolve(false);
                };
            });
        }
        
        async function showZoneDialog(cameraId, rect) {
            // Create modal dialog
            const dialog = document.createElement('div');
            dialog.style.cssText = `
                position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                background: white; padding: 25px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                z-index: 10000; min-width: 350px;
            `;
            
            dialog.innerHTML = `
                <h3 style="margin: 0 0 20px 0; color: #1f2937; font-size: 20px;">Configure Zone</h3>
                <div style="margin-bottom: 15px;">
                    <label style="display: block; margin-bottom: 5px; color: #4b5563; font-weight: 600;">Zone Name:</label>
                    <input type="text" id="zoneName" value="Zone ${Date.now()}" style="width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 14px;">
                </div>
                <div style="margin-bottom: 20px;">
                    <label style="display: block; margin-bottom: 8px; color: #4b5563; font-weight: 600;">Zone Type:</label>
                    <div style="display: flex; gap: 10px;">
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="zoneType" value="restricted" checked style="margin-right: 5px;">
                            <span style="color: #dc2626;">🚫 Restricted</span>
                        </label>
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="zoneType" value="warning" style="margin-right: 5px;">
                            <span style="color: #f59e0b;">⚠️ Warning</span>
                        </label>
                        <label style="flex: 1; cursor: pointer;">
                            <input type="radio" name="zoneType" value="safe" style="margin-right: 5px;">
                            <span style="color: #10b981;">✅ Safe</span>
                        </label>
                    </div>
                </div>
                <div style="display: flex; gap: 10px; justify-content: flex-end;">
                    <button id="cancelZone" style="padding: 8px 20px; background: #6b7280; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Cancel</button>
                    <button id="saveZone" style="padding: 8px 20px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">Save Zone</button>
                </div>
            `;
            
            const overlay = document.createElement('div');
            overlay.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); z-index: 9999;';
            
            document.body.appendChild(overlay);
            document.body.appendChild(dialog);
            
            return new Promise((resolve) => {
                document.getElementById('saveZone').onclick = async () => {
                    const name = document.getElementById('zoneName').value;
                    const type = document.querySelector('input[name="zoneType"]:checked').value;
                    
                    await saveZone(cameraId, name, type, rect);
                    
                    document.body.removeChild(overlay);
                    document.body.removeChild(dialog);
                    
                    // Hide canvas
                    const canvas = document.getElementById(`zoneCanvas${cameraId}`);
                    canvas.style.display = 'none';
                    activeZoneCanvas = null;
                    
                    resolve(true);
                };
                
                document.getElementById('cancelZone').onclick = () => {
                    document.body.removeChild(overlay);
                    document.body.removeChild(dialog);
                    
                    // Clear canvas
                    const canvas = document.getElementById(`zoneCanvas${cameraId}`);
                    const ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    drawExistingZones(ctx, cameraId);
                    
                    resolve(false);
                };
            });
        }
        
        async function saveZone(cameraId, zoneName, zoneType, rect) {
            try {
                const response = await fetch(`/api/camera/${cameraId}/zones`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        camera_id: cameraId,
                        name: zoneName,
                        type: zoneType,
                        rect: rect,  // [x, y, width, height]
                        active: true
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Add to local cache
                    if (!cameraZones[cameraId]) cameraZones[cameraId] = [];
                    cameraZones[cameraId].push(data.zone);
                    
                    addLogEntry(`${zoneType.toUpperCase()} zone "${zoneName}" created for Camera ${cameraId}`);
                    
                    // Play sound notification
                    playSound('success');
                } else {
                    alert('Failed to save zone');
                }
            } catch (error) {
                console.error('Error saving zone:', error);
                alert('Error saving zone: ' + error.message);
            }
        }
        
        async function updateZoneRect(cameraId, zoneId, rect) {
            try {
                const response = await fetch(`/api/camera/${cameraId}/zones/${zoneId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        rect: rect,
                        active: true
                    })
                });
                
                if (response.ok) {
                    console.log('Zone position/size updated');
                    playSound('success');
                } else {
                    console.error('Failed to update zone position/size');
                }
            } catch (error) {
                console.error('Error updating zone rect:', error);
            }
        }
        
        async function updateZone(cameraId, zoneId, zoneName, zoneType) {
            try {
                const response = await fetch(`/api/camera/${cameraId}/zones/${zoneId}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: zoneName,
                        type: zoneType,
                        active: true
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    // Update local cache
                    const zones = cameraZones[cameraId] || [];
                    const idx = zones.findIndex(z => z.id === zoneId);
                    if (idx !== -1) {
                        zones[idx] = {...zones[idx], name: zoneName, type: zoneType};
                    }
                    
                    addLogEntry(`Zone "${zoneName}" updated successfully`);
                    playSound('success');
                    
                    // Reload zones
                    await loadCameraZones(cameraId);
                } else {
                    alert('Failed to update zone');
                }
            } catch (error) {
                console.error('Error updating zone:', error);
                alert('Error updating zone: ' + error.message);
            }
        }
        
        async function deleteZone(cameraId, zoneId) {
            try {
                const response = await fetch(`/api/camera/${cameraId}/zones/${zoneId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    // Remove from local cache
                    const zones = cameraZones[cameraId] || [];
                    const idx = zones.findIndex(z => z.id === zoneId);
                    if (idx !== -1) {
                        const zoneName = zones[idx].name;
                        zones.splice(idx, 1);
                        addLogEntry(`Zone "${zoneName}" deleted`);
                    }
                    
                    playSound('success');
                    
                    // Reload zones
                    await loadCameraZones(cameraId);
                    
                    // Redraw canvas if active
                    if (activeZoneCanvas === cameraId) {
                        const canvas = document.getElementById(`zoneCanvas${cameraId}`);
                        const ctx = canvas.getContext('2d');
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
                        ctx.fillRect(0, 0, canvas.width, canvas.height);
                        drawExistingZones(ctx, cameraId);
                    }
                } else {
                    alert('Failed to delete zone');
                }
            } catch (error) {
                console.error('Error deleting zone:', error);
                alert('Error deleting zone: ' + error.message);
            }
        }
        
        async function deleteZoneWithConfirm(cameraId, zone) {
            if (confirm(`Delete zone "${zone.name}"?`)) {
                await deleteZone(cameraId, zone.id);
            }
        }
        
        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');
            
            // Load tab-specific content
            if (tabName === 'cameras') loadCameraDetails();
            if (tabName === 'zones') loadZoneManagement();
            if (tabName === 'settings') loadSettings();
            if (tabName === 'events') loadEvents();
        }
        
        // Load camera details
        async function loadCameraDetails() {
            const container = document.getElementById('cameraDetailView');
            const response = await fetch('/api/cameras');
            const data = await response.json();
            
            container.innerHTML = data.cameras.map(cam => `
                <div class="camera-detail">
                    <div class="camera-detail-header">
                        <div>
                            <h3 style="color: #00d4ff; margin-bottom: 5px;">Camera ${cam.id} - ${cam.position}</h3>
                            <div style="color: #888; font-size: 12px;">Device: /dev/video${cam.device_index} | Resolution: ${cam.resolution}</div>
                        </div>
                        <div class="camera-controls">
                            <button class="camera-control-btn" onclick="toggleDetection(${cam.id})">
                                ${cam.detection_enabled ? 'Disable' : 'Enable'} Detection
                            </button>
                            <button class="camera-control-btn" onclick="takeSnapshot(${cam.id})">Snapshot</button>
                            <button class="camera-control-btn" onclick="toggleRecording(${cam.id})">
                                ${cam.recording ? 'Stop' : 'Start'} Recording
                            </button>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px;">
                        <div>
                            <img src="/api/camera/${cam.id}/stream" style="width: 100%; border-radius: 8px;">
                        </div>
                        <div>
                            <div class="panel">
                                <div class="panel-header">Statistics</div>
                                <div class="system-info-item">
                                    <span class="info-label">Total Detections</span>
                                    <span class="info-value">${cam.total_detections}</span>
                                </div>
                                <div class="system-info-item">
                                    <span class="info-label">Current Detections</span>
                                    <span class="info-value">${cam.detections}</span>
                                </div>
                                <div class="system-info-item">
                                    <span class="info-label">Violations</span>
                                    <span class="info-value">${cam.violations}</span>
                                </div>
                                <div class="system-info-item">
                                    <span class="info-label">FPS</span>
                                    <span class="info-value">${cam.fps}</span>
                                </div>
                            </div>
                            
                            <div class="chart-container" id="chart-${cam.id}">
                                <canvas id="chartCanvas-${cam.id}"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        // Load zone management
        async function loadZoneManagement() {
            const select = document.getElementById('zoneCamera');
            const response = await fetch('/api/cameras');
            const data = await response.json();
            
            select.innerHTML = '<option value="">Select a camera...</option>' + 
                data.cameras.map(cam => `<option value="${cam.id}">Camera ${cam.id} - ${cam.position}</option>`).join('');
        }
        
        async function loadZonesForCamera() {
            const cameraId = document.getElementById('zoneCamera').value;
            if (!cameraId) {
                document.getElementById('zoneManagement').style.display = 'none';
                return;
            }
            
            document.getElementById('zoneManagement').style.display = 'block';
            
            // Load zones
            const response = await fetch(`/api/camera/${cameraId}/zones`);
            const data = await response.json();
            
            const zoneList = document.getElementById('zoneList');
            zoneList.innerHTML = '<h4 style="color: #00d4ff; margin: 20px 0 10px;">Existing Zones:</h4>' +
                (data.zones.length === 0 ? '<p style="color: #888;">No zones defined</p>' :
                data.zones.map(zone => `
                    <div class="zone-item">
                        <div>
                            <div class="zone-item-name">${zone.name}</div>
                            <div class="zone-item-type">${zone.type}</div>
                        </div>
                        <div class="zone-item-actions">
                            <button class="zone-action-btn" onclick="editZone('${zone.id}')">Edit</button>
                            <button class="zone-action-btn" onclick="deleteZone(${cameraId}, '${zone.id}')">Delete</button>
                        </div>
                    </div>
                `).join(''));
            
            // Setup canvas for drawing (OLD - disabled to avoid conflicts)
            // setupZoneCanvasOld(cameraId);
        }
        
        function setupZoneCanvasOld(cameraId) {
            // This function is disabled - using new zone drawing on camera cards instead
            const img = document.getElementById('zonePreviewImage');
            const canvas = document.getElementById('zoneCanvas');
            
            if (img && canvas) {
                img.src = `/api/camera/${cameraId}/stream`;
                img.onload = function() {
                    canvas.width = img.width;
                    canvas.height = img.height;
                };
            }
        }
        
        let drawingMode = null;
        let drawingPoints = [];
        
        function startDrawing(mode) {
            drawingMode = mode;
            drawingPoints = [];
            document.querySelectorAll('.zone-tool-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            addLogEntry(`Started drawing ${mode}`);
        }
        
        function clearDrawing() {
            const canvas = document.getElementById('zoneCanvas');
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            drawingPoints = [];
            drawingMode = null;
            document.querySelectorAll('.zone-tool-btn').forEach(btn => btn.classList.remove('active'));
        }
        
        async function savePolygonZone() {
            const cameraId = document.getElementById('zoneCamera').value;
            const name = document.getElementById('zoneName').value;
            const type = document.getElementById('zoneType').value;
            
            if (!name || drawingPoints.length < 3) {
                alert('Please enter zone name and draw at least 3 points');
                return;
            }
            
            const zone = {
                id: 'zone_' + Date.now(),
                camera_id: parseInt(cameraId),
                name: name,
                type: type,
                points: drawingPoints,
                active: true
            };
            
            const response = await fetch(`/api/camera/${cameraId}/zones`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(zone)
            });
            
            if (response.ok) {
                addLogEntry(`Zone "${name}" created successfully`);
                clearDrawing();
                document.getElementById('zoneName').value = '';
                loadZonesForCamera();
            }
        }
        
        async function deleteZone(cameraId, zoneId) {
            if (confirm('Are you sure you want to delete this zone?')) {
                await fetch(`/api/camera/${cameraId}/zones/${zoneId}`, {method: 'DELETE'});
                loadZonesForCamera();
            }
        }
        
        // Load settings
        async function loadSettings() {
            const container = document.getElementById('cameraSettings');
            const response = await fetch('/api/cameras');
            const data = await response.json();
            
            container.innerHTML = data.cameras.map(cam => `
                <div style="padding: 15px; background: #0a0a0a; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="color: #00d4ff; margin-bottom: 15px;">Camera ${cam.id} - ${cam.position}</h4>
                    <div class="setting-item">
                        <label class="setting-label">
                            <input type="checkbox" class="setting-checkbox" ${cam.detection_enabled ? 'checked' : ''}> Enable Detection
                        </label>
                    </div>
                    <div class="setting-item">
                        <label class="setting-label">Brightness</label>
                        <input type="range" class="setting-slider" min="0" max="255" value="128">
                    </div>
                </div>
            `).join('');
        }
        
        function updateConfidence(value) {
            document.getElementById('confValue').textContent = value;
        }
        
        function updateTransparency(value) {
            document.getElementById('transValue').textContent = value;
        }
        
        function saveSettings() {
            addLogEntry('Settings saved successfully');
            alert('Settings saved!');
        }
        
        // Barrier Detection Functions
        async function toggleBarrierDetection(enabled) {
            try {
                const response = await fetch('/api/barrier/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({enabled: enabled})
                });
                
                if (response.ok) {
                    const statusEl = document.getElementById('barrierStatus');
                    if (enabled) {
                        statusEl.textContent = 'ACTIVE';
                        statusEl.style.background = '#10b981';
                    } else {
                        statusEl.textContent = 'DISABLED';
                        statusEl.style.background = '#6b7280';
                    }
                    addLogEntry(`Barrier detection ${enabled ? 'enabled' : 'disabled'}`);
                }
            } catch (error) {
                console.error('Error toggling barrier detection:', error);
            }
        }
        
        async function updateBarrierSetting(setting, value) {
            const settings = {};
            settings[setting] = parseInt(value);
            
            // Update display
            if (setting === 'min_area') {
                document.getElementById('barrierMinValue').textContent = value;
            }
            
            try {
                await fetch('/api/barrier/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(settings)
                });
            } catch (error) {
                console.error('Error updating barrier setting:', error);
            }
        }
        
        function setBarrierSensitivity(level) {
            let minArea, minWidth, minHeight, minSaturation, minColorCoverage;
            
            switch(level) {
                case 'low':
                    // Stricter - fewer false positives, might miss some barriers
                    minArea = 5000;
                    minWidth = 100;
                    minHeight = 60;
                    minSaturation = 150;
                    minColorCoverage = 0.30;
                    break;
                case 'high':
                    // More sensitive - catches more barriers but may have false positives
                    minArea = 2000;
                    minWidth = 60;
                    minHeight = 30;
                    minSaturation = 100;
                    minColorCoverage = 0.15;
                    break;
                default: // medium - balanced
                    minArea = 3000;
                    minWidth = 80;
                    minHeight = 40;
                    minSaturation = 120;
                    minColorCoverage = 0.20;
            }
            
            // Update slider display
            document.getElementById('barrierMinArea').value = minArea;
            document.getElementById('barrierMinValue').textContent = minArea;
            
            // Send to server with all parameters
            fetch('/api/barrier/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    min_area: minArea,
                    min_width: minWidth,
                    min_height: minHeight,
                    min_saturation: minSaturation,
                    min_color_coverage: minColorCoverage
                })
            }).then(() => {
                addLogEntry(`Barrier sensitivity set to ${level}`);
            });
        }
        
        async function loadBarrierSettings() {
            try {
                const response = await fetch('/api/barrier/settings');
                const settings = await response.json();
                
                document.getElementById('barrierEnabled').checked = settings.enabled;
                document.getElementById('barrierMinArea').value = settings.min_area;
                document.getElementById('barrierMinValue').textContent = settings.min_area;
                
                const statusEl = document.getElementById('barrierStatus');
                if (settings.enabled) {
                    statusEl.textContent = 'ACTIVE';
                    statusEl.style.background = '#10b981';
                } else {
                    statusEl.textContent = 'DISABLED';
                    statusEl.style.background = '#6b7280';
                }
            } catch (error) {
                console.error('Error loading barrier settings:', error);
            }
        }
        
        // ============== AUDIO ALARM FUNCTIONS ==============
        
        // Audio context and sounds
        let audioContext = null;
        let audioEnabled = true;
        let audioVolume = 0.7;
        let lastAlarmTime = {};
        let audioCooldown = 10;
        
        function initAudioContext() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
            return audioContext;
        }
        
        function playAlarmSound(type = 'alert', volume = audioVolume) {
            if (!audioEnabled) return;
            
            const ctx = initAudioContext();
            const now = ctx.currentTime;
            
            const oscillator = ctx.createOscillator();
            const gainNode = ctx.createGain();
            oscillator.connect(gainNode);
            gainNode.connect(ctx.destination);
            
            gainNode.gain.value = volume;
            
            switch(type) {
                case 'siren':
                    // Siren sound - alternating frequencies
                    oscillator.type = 'sawtooth';
                    oscillator.frequency.setValueAtTime(800, now);
                    oscillator.frequency.linearRampToValueAtTime(1200, now + 0.3);
                    oscillator.frequency.linearRampToValueAtTime(800, now + 0.6);
                    oscillator.frequency.linearRampToValueAtTime(1200, now + 0.9);
                    gainNode.gain.setValueAtTime(volume, now);
                    gainNode.gain.linearRampToValueAtTime(0, now + 1.0);
                    oscillator.start(now);
                    oscillator.stop(now + 1.0);
                    break;
                    
                case 'beep':
                    // Simple beep
                    oscillator.type = 'sine';
                    oscillator.frequency.value = 880;
                    gainNode.gain.setValueAtTime(volume * 0.5, now);
                    gainNode.gain.linearRampToValueAtTime(0, now + 0.2);
                    oscillator.start(now);
                    oscillator.stop(now + 0.2);
                    break;
                    
                case 'chime':
                    // Gentle chime
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(523, now);  // C5
                    oscillator.frequency.setValueAtTime(659, now + 0.15);  // E5
                    oscillator.frequency.setValueAtTime(784, now + 0.3);  // G5
                    gainNode.gain.setValueAtTime(volume * 0.4, now);
                    gainNode.gain.linearRampToValueAtTime(0, now + 0.5);
                    oscillator.start(now);
                    oscillator.stop(now + 0.5);
                    break;
                    
                default: // 'alert'
                    // Alert tone - urgent beeps
                    oscillator.type = 'square';
                    oscillator.frequency.value = 1000;
                    gainNode.gain.setValueAtTime(volume * 0.6, now);
                    gainNode.gain.setValueAtTime(0, now + 0.1);
                    gainNode.gain.setValueAtTime(volume * 0.6, now + 0.2);
                    gainNode.gain.setValueAtTime(0, now + 0.3);
                    gainNode.gain.setValueAtTime(volume * 0.6, now + 0.4);
                    gainNode.gain.linearRampToValueAtTime(0, now + 0.6);
                    oscillator.start(now);
                    oscillator.stop(now + 0.6);
            }
        }
        
        function triggerViolationAlarm(cameraId, zoneName) {
            if (!audioEnabled) return;
            
            const key = `${cameraId}-${zoneName}`;
            const now = Date.now();
            
            // Check cooldown
            if (lastAlarmTime[key] && (now - lastAlarmTime[key]) < audioCooldown * 1000) {
                return; // Still in cooldown
            }
            
            lastAlarmTime[key] = now;
            const soundType = document.getElementById('audioSoundType')?.value || 'alert';
            playAlarmSound(soundType, audioVolume);
        }
        
        async function toggleAudioAlarm(enabled) {
            audioEnabled = enabled;
            
            try {
                await fetch('/api/audio/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({enabled: enabled})
                });
                
                const statusEl = document.getElementById('audioStatus');
                if (enabled) {
                    statusEl.textContent = 'ACTIVE';
                    statusEl.style.background = '#10b981';
                } else {
                    statusEl.textContent = 'DISABLED';
                    statusEl.style.background = '#6b7280';
                }
                addLogEntry(`Audio alarms ${enabled ? 'enabled' : 'disabled'}`);
            } catch (error) {
                console.error('Error updating audio settings:', error);
            }
        }
        
        function updateAudioVolume(value) {
            audioVolume = value / 100;
            document.getElementById('volumeValue').textContent = value;
            
            fetch('/api/audio/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({volume: audioVolume})
            });
        }
        
        function updateAudioSoundType(type) {
            fetch('/api/audio/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({sound_type: type})
            });
            addLogEntry(`Audio sound type set to ${type}`);
        }
        
        function updateAudioCooldown(value) {
            audioCooldown = parseInt(value);
            fetch('/api/audio/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cooldown: audioCooldown})
            });
        }
        
        function testAudioAlarm(type) {
            // Ensure audio context is initialized with user gesture
            initAudioContext();
            playAlarmSound(type, audioVolume);
            addLogEntry(`Test alarm played: ${type}`);
        }
        
        async function loadAudioSettings() {
            try {
                const response = await fetch('/api/audio/settings');
                const settings = await response.json();
                
                audioEnabled = settings.enabled;
                audioVolume = settings.volume;
                audioCooldown = settings.cooldown;
                
                document.getElementById('audioEnabled').checked = settings.enabled;
                document.getElementById('audioVolume').value = settings.volume * 100;
                document.getElementById('volumeValue').textContent = Math.round(settings.volume * 100);
                document.getElementById('audioSoundType').value = settings.sound_type;
                document.getElementById('audioCooldown').value = settings.cooldown;
                
                const statusEl = document.getElementById('audioStatus');
                if (settings.enabled) {
                    statusEl.textContent = 'ACTIVE';
                    statusEl.style.background = '#10b981';
                } else {
                    statusEl.textContent = 'DISABLED';
                    statusEl.style.background = '#6b7280';
                }
            } catch (error) {
                console.error('Error loading audio settings:', error);
            }
        }
        
        function showToast(message, type = 'info') {
            // Create toast notification
            const toast = document.createElement('div');
            toast.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 15px 25px;
                border-radius: 8px;
                color: white;
                font-weight: bold;
                z-index: 10000;
                animation: slideIn 0.3s ease;
                background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            `;
            toast.textContent = message;
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'fadeOut 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }
        
        // ============== PDF REPORT FUNCTIONS ==============
        
        async function generateReport(period) {
            addLogEntry(`Generating ${period} report...`);
            
            // Show loading indicator
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '⏳ Generating...';
            btn.disabled = true;
            
            try {
                const response = await fetch(`/api/reports/generate?period=${period}`);
                
                if (response.ok) {
                    // Download the PDF
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `vigil_report_${period}_${new Date().toISOString().slice(0,10)}.pdf`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    window.URL.revokeObjectURL(url);
                    
                    addLogEntry(`${period.charAt(0).toUpperCase() + period.slice(1)} report generated and downloaded`);
                    
                    // Refresh reports list
                    loadReportsList();
                } else {
                    const error = await response.json();
                    alert(`Failed to generate report: ${error.message}`);
                    addLogEntry(`Report generation failed: ${error.message}`);
                }
            } catch (error) {
                console.error('Error generating report:', error);
                alert('Failed to generate report. Check if reportlab is installed.');
            } finally {
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        }
        
        async function loadReportsList() {
            const container = document.getElementById('reportsList');
            
            try {
                const response = await fetch('/api/reports/list');
                const data = await response.json();
                
                if (data.reports.length === 0) {
                    container.innerHTML = `
                        <div style="color: #888; text-align: center; padding: 20px;">
                            No reports generated yet. Click a button above to create one.
                        </div>
                    `;
                    return;
                }
                
                container.innerHTML = data.reports.map(report => `
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px; border-bottom: 1px solid #333; background: rgba(255,255,255,0.02);">
                        <div>
                            <div style="color: #fff; font-weight: 500;">📄 ${report.filename}</div>
                            <div style="color: #888; font-size: 12px;">
                                ${new Date(report.created).toLocaleString()} • ${(report.size / 1024).toFixed(1)} KB
                            </div>
                        </div>
                        <a href="${report.download_url}" download 
                           style="padding: 6px 12px; background: #3b82f6; color: white; border-radius: 6px; text-decoration: none; font-size: 12px;">
                            ⬇️ Download
                        </a>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Error loading reports:', error);
                container.innerHTML = `
                    <div style="color: #ef4444; text-align: center; padding: 20px;">
                        Failed to load reports list
                    </div>
                `;
            }
        }
        
        // Load events
        async function loadEvents() {
            const container = document.getElementById('eventHistory');
            const response = await fetch('/api/events');
            const data = await response.json();
            
            container.innerHTML = data.events.map(event => `
                <div class="log-entry">
                    <div class="log-time">${new Date(event.timestamp).toLocaleString()}</div>
                    <div class="log-message">
                        <strong>Camera ${event.camera_id}</strong> - ${event.event_type}: ${event.description}
                        ${event.confidence ? ` (${(event.confidence * 100).toFixed(0)}%)` : ''}
                    </div>
                </div>
            `).join('');
        }
        
        // Camera controls
        async function toggleDetection(cameraId) {
            const response = await fetch(`/api/camera/${cameraId}/settings`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({detection_enabled: true})
            });
            addLogEntry(`Detection toggled for Camera ${cameraId}`);
            loadCameraDetails();
        }
        
        function takeSnapshot(cameraId) {
            addLogEntry(`Snapshot taken from Camera ${cameraId}`);
            alert(`Snapshot saved for Camera ${cameraId}`);
        }
        
        function toggleRecording(cameraId) {
            addLogEntry(`Recording toggled for Camera ${cameraId}`);
            alert(`Recording toggled for Camera ${cameraId}`);
        }
        
        // Initialize
        window.lastDetectionCount = 0;
        lastViolationCount = 0;
        window.addEventListener('DOMContentLoaded', async () => {
            // Initialize sound and notifications
            initSound();
            requestNotificationPermission();
            initTouchMode();  // Restore touch mode preference
            initDisplayMode();  // Restore display mode preference
            loadBarrierSettings();  // Load barrier detection settings
            loadAudioSettings();  // Load audio alarm settings
            loadReportsList();  // Load generated reports
            
            await initializeCameras();
            updateStats();
            initCharts();
            
            // Start update loops
            setInterval(updateStats, 2000);
            setInterval(updateSystemParams, 3000);
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VIGIL V6.0 Professional Complete Web Server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
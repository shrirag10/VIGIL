# VIGIL — Vehicle-Installed Guard for Injury Limitation

**AI-Powered Forklift Pedestrian Detection System with Web Dashboard**

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

VIGIL is a professional-grade AI-powered pedestrian detection system designed to be mounted on forklifts, featuring:

- 🎥 **Multi-Camera Support** — Up to 4 cameras with real-time streaming
- 🤖 **YOLO Detection** — Person detection with confidence scoring
- 🚧 **Barrier Detection** — Industrial safety barrier recognition (red/yellow striped)
- 🔴 **Zone Management** — Restricted, warning, and safe zones
- ⚠️ **Tamper Detection** — Camera blocking/disconnection alerts
- 📊 **Real-time Analytics** — Live statistics and event logging
- 📹 **Recording** — Per-camera video recording
- 📄 **PDF Reports** — Automated report generation
- 🔊 **Audio Alarms** — Browser-based alert sounds
- 🌐 **Web Dashboard** — Complete professional UI

## Requirements

### Python Packages

```bash
pip install fastapi uvicorn pydantic opencv-python numpy ultralytics reportlab psutil
```

### Optional Dependencies

- **YOLO Model** — Automatically downloaded on first run
- **Cameras** — USB or built-in cameras (tested with up to 4)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn pydantic opencv-python numpy ultralytics reportlab psutil
   ```

2. **Run the server:**
   ```bash
   python3 VIGIL.py
   ```

3. **Access the dashboard:**
   Open [http://localhost:8000](http://localhost:8000) in your browser

## Features

### Camera Management
- Auto-detection of available cameras
- Live MJPEG streaming
- Per-camera detection toggle
- Recording controls

### Zone System
- **Restricted Zones** (Red) - Triggers immediate alerts
- **Warning Zones** (Yellow) - Triggers warnings
- **Safe Zones** (Green) - Monitored areas
- Draw zones directly on camera feeds

### Detection Capabilities
- Person detection via YOLOv8
- Industrial barrier detection (red/yellow striped)
- Behind-barrier filtering (reduces false positives)
- Configurable confidence thresholds

### Tamper Detection
- Camera blocked/covered detection (darkness)
- Camera disconnection monitoring
- Automatic alerts and logging

### Alerts & Notifications
- Real-time violation logging
- Audio alarms (browser-based)
- Email alerts (configurable)

### Recording & Reports
- Per-camera video recording
- Violation history export (JSON/CSV)
- PDF report generation (daily/weekly/monthly)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web Dashboard |
| `/api/cameras` | GET | List all cameras |
| `/api/camera/{id}/stream` | GET | MJPEG video stream |
| `/api/camera/{id}/zones` | GET/POST | Zone management |
| `/api/system/stats` | GET | System statistics |
| `/api/violations` | GET | Violation history |
| `/api/reports/generate` | GET | Generate PDF report |
| `/api/tamper/status` | GET | Tamper detection status |

## Configuration

### Email Alerts
Edit `EMAIL_CONFIG` in VIGIL.py:
```python
EMAIL_CONFIG = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',
    'recipient_emails': ['alert@example.com']
}
```

### Barrier Detection
Adjust `barrier_settings` in VIGIL.py for your environment:
```python
barrier_settings = {
    'enabled': True,
    'min_area': 5000,
    'min_saturation': 150,
    # ... more options
}
```

## Directory Structure

```
VIGIL/
├── VIGIL.py           # Main application
├── README.md          # This file
├── zones.json         # Saved zone configurations
├── violations.json    # Violation history
├── recordings/        # Video recordings
└── reports/           # Generated PDF reports
```

## Troubleshooting

### Port Already in Use
```bash
lsof -ti:8000 | xargs kill -9
python3 VIGIL.py
```

### NumPy Compatibility Warning
If you see NumPy 2.x warnings with PyTorch:
```bash
pip install "numpy<2"
```

### No Cameras Detected
- Check USB connections
- Grant camera permissions in System Preferences (macOS)
- Try different camera indices

## License

MIT License - Feel free to use and modify for your projects.

## Author

VIGIL — Vehicle-Installed Guard for Injury Limitation

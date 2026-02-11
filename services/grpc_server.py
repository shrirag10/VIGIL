"""
VIGIL V7.0 — gRPC Server Implementation
Provides high-performance streaming for detection results,
violation events, tamper alerts, and system health.

Usage:
    python services/grpc_server.py          # standalone
    # or imported by VIGIL.py as a background thread
"""

import logging
import queue
import sys
import threading
import time
from concurrent import futures
from datetime import datetime
from pathlib import Path

import grpc
from google.protobuf import empty_pb2
from google.protobuf.timestamp_pb2 import Timestamp

# Add parent to path so we can import from VIGIL
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger('VIGIL_GRPC')

# ── These will be populated by generated proto stubs ──
# Generate with:
#   python -m grpc_tools.protoc -I proto --python_out=services/generated \
#          --grpc_python_out=services/generated proto/vigil.proto
try:
    from services.generated import vigil_pb2, vigil_pb2_grpc
    GRPC_STUBS_AVAILABLE = True
except ImportError:
    GRPC_STUBS_AVAILABLE = False
    logger.warning(
        "gRPC stubs not generated. Run:\n"
        "  python -m grpc_tools.protoc -I proto "
        "--python_out=services/generated "
        "--grpc_python_out=services/generated "
        "proto/vigil.proto"
    )


def _now_timestamp():
    """Create a protobuf Timestamp for now."""
    ts = Timestamp()
    ts.FromDatetime(datetime.utcnow())
    return ts


# ═══════════════════════════════════════════
# Event Bus (in-process) — bridges VIGIL core → gRPC streams
# ═══════════════════════════════════════════

class EventBus:
    """Simple pub/sub for relaying events to gRPC stream subscribers."""

    def __init__(self):
        self._subscribers: dict[str, list[queue.Queue]] = {
            'detection': [],
            'violation': [],
            'tamper': [],
            'health': [],
        }
        self._lock = threading.Lock()

    def subscribe(self, channel: str) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.setdefault(channel, []).append(q)
        return q

    def unsubscribe(self, channel: str, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers[channel].remove(q)
            except (KeyError, ValueError):
                pass

    def publish(self, channel: str, event):
        with self._lock:
            subs = list(self._subscribers.get(channel, []))
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                # Drop oldest if subscriber is slow
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except queue.Empty:
                    pass


# Global event bus — VIGIL.py publishes, gRPC streams consume
event_bus = EventBus()


# ═══════════════════════════════════════════
# gRPC Service Implementations
# ═══════════════════════════════════════════

if GRPC_STUBS_AVAILABLE:

    class VIGILDetectionServicer(vigil_pb2_grpc.VIGILDetectionServicer):
        """Handles frame streaming and event subscriptions."""

        def __init__(self, vigil_app=None):
            self.vigil_app = vigil_app

        def StreamFrames(self, request_iterator, context):
            """
            Bi-directional streaming:
            - Client sends CameraFrame messages (JPEG bytes)
            - Server returns DetectionResult messages
            """
            logger.info("gRPC StreamFrames: client connected")

            for frame_msg in request_iterator:
                if context.is_active():
                    # Process frame through VIGIL detection pipeline
                    result = self._process_frame(frame_msg)
                    if result:
                        yield result
                else:
                    break

            logger.info("gRPC StreamFrames: client disconnected")

        def _process_frame(self, frame_msg):
            """Process a single frame through the detection pipeline."""
            import cv2
            import numpy as np

            try:
                # Decode JPEG
                nparr = np.frombuffer(frame_msg.jpeg_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    return None

                # Import dynamically to avoid circular imports
                if self.vigil_app:
                    from VIGIL import process_frame_with_detection
                    processed, detections, violations = process_frame_with_detection(
                        frame, frame_msg.camera_id
                    )

                    result = vigil_pb2.DetectionResult(
                        camera_id=frame_msg.camera_id,
                        timestamp=_now_timestamp(),
                        person_count=detections,
                        violation_count=violations,
                    )

                    # Publish to event bus for other subscribers
                    event_bus.publish('detection', result)

                    return result

            except Exception as e:
                logger.error(f"gRPC frame processing error: {e}")
                return None

        def StreamViolations(self, request, context):
            """Server-side streaming: push violation events to subscribers."""
            logger.info("gRPC StreamViolations: subscriber connected")
            sub = event_bus.subscribe('violation')

            try:
                while context.is_active():
                    try:
                        event = sub.get(timeout=1.0)
                        # Events from VIGIL.py arrive as plain dicts — convert
                        if isinstance(event, dict):
                            cam_id = event.get('camera_id', 0)
                            zone_type = event.get('zone_type', '')
                            # Apply filters
                            if request.camera_id and cam_id != request.camera_id:
                                continue
                            if request.zone_type and zone_type != request.zone_type:
                                continue
                            yield vigil_pb2.ViolationEvent(
                                camera_id=cam_id,
                                zone_name=event.get('zone_name', ''),
                                zone_type=zone_type,
                                timestamp=_now_timestamp(),
                            )
                        else:
                            # Already a protobuf message
                            if request.camera_id and event.camera_id != request.camera_id:
                                continue
                            if request.zone_type and event.zone_type != request.zone_type:
                                continue
                            yield event
                    except queue.Empty:
                        continue
            finally:
                event_bus.unsubscribe('violation', sub)
                logger.info("gRPC StreamViolations: subscriber disconnected")

        def StreamTamperAlerts(self, request, context):
            """Server-side streaming: push tamper alert events."""
            logger.info("gRPC StreamTamperAlerts: subscriber connected")
            sub = event_bus.subscribe('tamper')

            try:
                while context.is_active():
                    try:
                        event = sub.get(timeout=1.0)
                        yield event
                    except queue.Empty:
                        continue
            finally:
                event_bus.unsubscribe('tamper', sub)

        def StreamHealth(self, request, context):
            """Server-side streaming: periodic health status updates."""
            logger.info("gRPC StreamHealth: subscriber connected")

            while context.is_active():
                try:
                    health = self._get_health()
                    yield health
                    time.sleep(2.0)
                except Exception as e:
                    logger.error(f"Health stream error: {e}")
                    time.sleep(5.0)

            logger.info("gRPC StreamHealth: subscriber disconnected")

        def _get_health(self):
            """Build a HealthStatus message from current system state."""
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                try:
                    temps = psutil.sensors_temperatures()
                    temp = temps.get('coretemp', [{}])[0].current if 'coretemp' in temps else 30 + cpu * 0.6
                except Exception:
                    temp = 30 + cpu * 0.6
            except ImportError:
                cpu, mem, temp = 0, 0, 0

            return vigil_pb2.HealthStatus(
                timestamp=_now_timestamp(),
                status='operational',
                cpu_percent=cpu,
                memory_percent=mem,
                temperature_c=temp,
                yolo_available=self._check_yolo(),
            )

        def _check_yolo(self):
            try:
                from VIGIL import YOLO_AVAILABLE, detection_engine
                return YOLO_AVAILABLE and detection_engine is not None
            except Exception:
                return False

    class VIGILManagementServicer(vigil_pb2_grpc.VIGILManagementServicer):
        """Handles camera, zone, and system management RPCs."""

        def __init__(self, vigil_app=None):
            self.vigil_app = vigil_app

        def ListCameras(self, request, context):
            """List all cameras with current status."""
            cameras_list = vigil_pb2.CameraList()
            try:
                from VIGIL import camera_settings, camera_stats, cameras
                for cam_id, cam_info in cameras.items():
                    stats = camera_stats.get(cam_id, {})
                    settings = camera_settings.get(cam_id, {})
                    cameras_list.cameras.append(vigil_pb2.CameraStatus(
                        camera_id=cam_id,
                        name=cam_info.get('position', f'Camera {cam_id}'),
                        active=cam_info.get('active', False),
                        recording=cam_info.get('recording', False),
                        detection_enabled=settings.get('detection_enabled', True),
                        fps=stats.get('fps', 0),
                        current_detections=stats.get('detections', 0),
                        total_detections=stats.get('total_detections', 0),
                    ))
            except Exception as e:
                logger.error(f"ListCameras error: {e}")
            return cameras_list

        def GetCameraStatus(self, request, context):
            """Get single camera status."""
            cameras_list = self.ListCameras(empty_pb2.Empty(), context)
            for cam in cameras_list.cameras:
                if cam.camera_id == request.camera_id:
                    return cam
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Camera {request.camera_id} not found")
            return vigil_pb2.CameraStatus()

        def ToggleDetection(self, request, context):
            """Toggle detection for a camera."""
            try:
                from VIGIL import camera_settings
                camera_settings[request.camera_id]['detection_enabled'] = request.enabled
                return vigil_pb2.OperationResult(
                    success=True,
                    message=f"Detection {'enabled' if request.enabled else 'disabled'} for camera {request.camera_id}"
                )
            except Exception as e:
                return vigil_pb2.OperationResult(success=False, message=str(e))

        def ToggleRecording(self, request, context):
            """Toggle recording for a camera."""
            try:
                from VIGIL import cameras, start_recording, stop_recording
                if request.start:
                    start_recording(request.camera_id)
                    cameras[request.camera_id]['recording'] = True
                else:
                    stop_recording(request.camera_id)
                    cameras[request.camera_id]['recording'] = False
                return vigil_pb2.OperationResult(
                    success=True,
                    message=f"Recording {'started' if request.start else 'stopped'}"
                )
            except Exception as e:
                return vigil_pb2.OperationResult(success=False, message=str(e))

        def ListZones(self, request, context):
            """List zones for a camera."""
            try:
                from VIGIL import camera_zones
                zone_list = vigil_pb2.ZoneList()
                for z in camera_zones.get(request.camera_id, []):
                    rect = z.get('rect', [0, 0, 0, 0])
                    zone_list.zones.append(vigil_pb2.ZoneConfig(
                        id=z.get('id', ''),
                        camera_id=request.camera_id,
                        name=z.get('name', ''),
                        type=z.get('type', 'safe'),
                        rect=vigil_pb2.BoundingBox(
                            x1=rect[0], y1=rect[1], x2=rect[2], y2=rect[3]
                        ),
                        active=z.get('active', True),
                    ))
                return zone_list
            except Exception as e:
                logger.error(f"ListZones error: {e}")
                return vigil_pb2.ZoneList()

        def GetHealth(self, request, context):
            """Get current system health."""
            return VIGILDetectionServicer()._get_health()

        def SetSystemPower(self, request, context):
            """Toggle system power."""
            try:
                from VIGIL import camera_settings, system_stats
                system_stats['system_enabled'] = request.enabled
                for cam_id in camera_settings:
                    camera_settings[cam_id]['detection_enabled'] = request.enabled
                return vigil_pb2.OperationResult(
                    success=True,
                    message=f"System {'enabled' if request.enabled else 'disabled'}"
                )
            except Exception as e:
                return vigil_pb2.OperationResult(success=False, message=str(e))

        def Shutdown(self, request, context):
            """Shutdown the system."""
            import os
            import signal
            logger.info("gRPC Shutdown requested")
            os.kill(os.getpid(), signal.SIGTERM)
            return vigil_pb2.OperationResult(success=True, message="Shutting down")


# ═══════════════════════════════════════════
# Server Lifecycle
# ═══════════════════════════════════════════

_grpc_server = None


def start_grpc_server(port: int = 50051, vigil_app=None) -> grpc.Server:
    """Start the gRPC server in a background thread."""
    global _grpc_server

    if not GRPC_STUBS_AVAILABLE:
        logger.warning("gRPC stubs not available — server not started")
        return None

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 10 * 1024 * 1024),     # 10MB
            ('grpc.max_receive_message_length', 10 * 1024 * 1024),
            ('grpc.keepalive_time_ms', 30000),
            ('grpc.keepalive_timeout_ms', 10000),
        ]
    )

    vigil_pb2_grpc.add_VIGILDetectionServicer_to_server(
        VIGILDetectionServicer(vigil_app), server
    )
    vigil_pb2_grpc.add_VIGILManagementServicer_to_server(
        VIGILManagementServicer(vigil_app), server
    )

    server.add_insecure_port(f'[::]:{port}')
    server.start()
    _grpc_server = server

    logger.info(f"gRPC server started on port {port}")
    return server


def stop_grpc_server():
    """Stop the gRPC server gracefully."""
    global _grpc_server
    if _grpc_server:
        _grpc_server.stop(grace=5)
        logger.info("gRPC server stopped")
        _grpc_server = None


# ── Standalone entry point ──
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    server = start_grpc_server(port=50051)
    if server:
        logger.info("gRPC server running. Press Ctrl+C to stop.")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            stop_grpc_server()
    else:
        logger.error("Failed to start gRPC server. Generate stubs first.")

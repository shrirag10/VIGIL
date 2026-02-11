"""
VIGIL V7.0 — Kafka Event Producer
Publishes detection events, violations, tamper alerts, and system metrics
to Kafka topics for decoupled consumption by alerting, recording,
and analytics services.

Topics:
    vigil.detections          — Every detection cycle result
    vigil.violations          — Zone violations
    vigil.tamper-alerts       — Camera tamper events
    vigil.system-metrics      — Periodic system health
    vigil.recording-commands  — Start/stop recording triggers

Usage:
    producer = VIGILKafkaProducer(bootstrap_servers='localhost:9092')
    producer.publish_violation({...})
"""

import json
import logging
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger('VIGIL_KAFKA')

# ── Topic names ──
TOPIC_DETECTIONS = 'vigil.detections'
TOPIC_VIOLATIONS = 'vigil.violations'
TOPIC_TAMPER     = 'vigil.tamper-alerts'
TOPIC_METRICS    = 'vigil.system-metrics'
TOPIC_RECORDING  = 'vigil.recording-commands'

# Try to import Kafka
try:
    from kafka import KafkaAdminClient, KafkaConsumer, KafkaProducer
    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.info(
        "kafka-python not installed. Install with: pip install kafka-python\n"
        "Kafka integration will be disabled."
    )


def _json_serializer(data: Any) -> bytes:
    """Serialize data to JSON bytes for Kafka."""
    return json.dumps(data, default=str).encode('utf-8')


class VIGILKafkaProducer:
    """
    Kafka producer for VIGIL events.
    Falls back to no-op if Kafka is unavailable or unreachable.
    """

    def __init__(
        self,
        bootstrap_servers: str = 'localhost:9092',
        enabled: bool = True,
    ):
        self.enabled = enabled and KAFKA_AVAILABLE
        self.producer: Any | None = None
        self._connected = False

        if not self.enabled:
            logger.info("Kafka producer disabled (library not available or disabled)")
            return

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=_json_serializer,
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                max_block_ms=5000,
                linger_ms=10,       # Batch small messages
                batch_size=16384,
            )
            self._connected = True
            logger.info(f"Kafka producer connected to {bootstrap_servers}")
            self._ensure_topics(bootstrap_servers)
        except Exception as e:
            logger.warning(f"Kafka producer failed to connect: {e}. Running without Kafka.")
            self.enabled = False

    def _ensure_topics(self, bootstrap_servers: str):
        """Create topics if they don't exist."""
        try:
            admin = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
            topics = [
                NewTopic(name=TOPIC_DETECTIONS, num_partitions=4, replication_factor=1),
                NewTopic(name=TOPIC_VIOLATIONS, num_partitions=2, replication_factor=1),
                NewTopic(name=TOPIC_TAMPER, num_partitions=1, replication_factor=1),
                NewTopic(name=TOPIC_METRICS, num_partitions=1, replication_factor=1),
                NewTopic(name=TOPIC_RECORDING, num_partitions=1, replication_factor=1),
            ]
            admin.create_topics(new_topics=topics, validate_only=False)
            logger.info("Kafka topics created/verified")
        except TopicAlreadyExistsError:
            logger.debug("Kafka topics already exist")
        except Exception as e:
            logger.debug(f"Topic creation note: {e}")

    def _publish(self, topic: str, key: str, data: dict[str, Any]):
        """Internal publish with error handling."""
        if not self.enabled or not self.producer:
            return

        try:
            future = self.producer.send(topic, key=key, value=data)
            # Don't block — fire and forget for performance
        except Exception as e:
            logger.error(f"Kafka publish error on {topic}: {e}")

    # ── Public API ──

    def publish_detection(self, camera_id: int, detections: int, violations: int,
                          barriers: int = 0, fps: float = 0):
        """Publish a detection cycle result."""
        self._publish(TOPIC_DETECTIONS, f'cam-{camera_id}', {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'person_count': detections,
            'violation_count': violations,
            'barrier_count': barriers,
            'fps': fps,
        })

    def publish_violation(self, violation: dict[str, Any]):
        """Publish a zone violation event."""
        camera_id = violation.get('camera_id', 0)
        self._publish(TOPIC_VIOLATIONS, f'cam-{camera_id}', {
            'timestamp': violation.get('timestamp', datetime.now().isoformat()),
            'camera_id': camera_id,
            'camera_name': violation.get('camera_name', ''),
            'zone_id': violation.get('zone_id', ''),
            'zone_name': violation.get('zone_name', ''),
            'zone_type': violation.get('zone_type', ''),
            'person_box': violation.get('person_box'),
        })

    def publish_tamper_alert(self, camera_id: int, tamper_type: str, description: str):
        """Publish a camera tamper alert."""
        self._publish(TOPIC_TAMPER, f'cam-{camera_id}', {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'tamper_type': tamper_type,
            'description': description,
        })

    def publish_system_metrics(self, metrics: dict[str, Any]):
        """Publish periodic system health metrics."""
        self._publish(TOPIC_METRICS, 'system', {
            'timestamp': datetime.now().isoformat(),
            **metrics,
        })

    def publish_recording_command(self, camera_id: int, start: bool):
        """Publish a recording start/stop command."""
        self._publish(TOPIC_RECORDING, f'cam-{camera_id}', {
            'timestamp': datetime.now().isoformat(),
            'camera_id': camera_id,
            'action': 'start' if start else 'stop',
        })

    def flush(self):
        """Flush pending messages."""
        if self.producer:
            self.producer.flush(timeout=5)

    def close(self):
        """Close the producer."""
        if self.producer:
            self.producer.flush(timeout=5)
            self.producer.close()
            logger.info("Kafka producer closed")


class VIGILKafkaConsumer:
    """
    Kafka consumer for VIGIL events.
    Runs in a background thread, calls registered handlers for each topic.
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str = 'vigil-consumer',
        bootstrap_servers: str = 'localhost:9092',
    ):
        self.handlers: dict[str, list] = {}
        self.consumer: Any | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        if not KAFKA_AVAILABLE:
            logger.info("Kafka consumer disabled (library not available)")
            return

        try:
            self.consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000,
            )
            logger.info(f"Kafka consumer connected, subscribed to: {topics}")
        except Exception as e:
            logger.warning(f"Kafka consumer failed to connect: {e}")

    def on(self, topic: str, handler):
        """Register a handler for a topic."""
        self.handlers.setdefault(topic, []).append(handler)

    def start(self):
        """Start consuming in a background thread."""
        if not self.consumer:
            return

        self._running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("Kafka consumer started")

    def _consume_loop(self):
        """Main consumer loop."""
        while self._running:
            try:
                for message in self.consumer:
                    if not self._running:
                        break

                    topic = message.topic
                    handlers = self.handlers.get(topic, [])
                    for handler in handlers:
                        try:
                            handler(message.value)
                        except Exception as e:
                            logger.error(f"Kafka handler error on {topic}: {e}")
            except Exception as e:
                if self._running:
                    logger.error(f"Kafka consumer loop error: {e}")

    def stop(self):
        """Stop the consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.consumer:
            self.consumer.close()
            logger.info("Kafka consumer stopped")


# ═══════════════════════════════════════════
# Pre-configured Consumer Services
# ═══════════════════════════════════════════

class AlertConsumer:
    """
    Consumes violation and tamper events, dispatches alerts
    (email, buzzer, push notifications).
    """

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.consumer = VIGILKafkaConsumer(
            topics=[TOPIC_VIOLATIONS, TOPIC_TAMPER],
            group_id='vigil-alerts',
            bootstrap_servers=bootstrap_servers,
        )
        self.consumer.on(TOPIC_VIOLATIONS, self._handle_violation)
        self.consumer.on(TOPIC_TAMPER, self._handle_tamper)

    def _handle_violation(self, event: dict):
        """Handle violation event — send alerts."""
        zone_type = event.get('zone_type', '')
        camera_name = event.get('camera_name', '')
        logger.info(f"ALERT: {zone_type.upper()} violation on {camera_name}")
        # TODO: Integrate with email, ESP32 buzzer, push notifications

    def _handle_tamper(self, event: dict):
        """Handle tamper event — send urgent alerts."""
        tamper_type = event.get('tamper_type', '')
        camera_id = event.get('camera_id', '')
        logger.warning(f"TAMPER ALERT: Camera {camera_id} - {tamper_type}")
        # TODO: Immediate alert dispatch

    def start(self):
        self.consumer.start()

    def stop(self):
        self.consumer.stop()


class AnalyticsConsumer:
    """
    Consumes detection events for real-time analytics aggregation.
    """

    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.consumer = VIGILKafkaConsumer(
            topics=[TOPIC_DETECTIONS, TOPIC_METRICS],
            group_id='vigil-analytics',
            bootstrap_servers=bootstrap_servers,
        )
        self.consumer.on(TOPIC_DETECTIONS, self._handle_detection)
        self.consumer.on(TOPIC_METRICS, self._handle_metrics)

        # Aggregation state
        self.detection_counts: dict[int, int] = {}
        self.total_processed = 0

    def _handle_detection(self, event: dict):
        """Aggregate detection stats."""
        cam_id = event.get('camera_id', 0)
        self.detection_counts[cam_id] = self.detection_counts.get(cam_id, 0) + event.get('person_count', 0)
        self.total_processed += 1

    def _handle_metrics(self, event: dict):
        """Process system metrics."""
        # Could push to time-series DB (TimescaleDB, InfluxDB)
        pass

    def start(self):
        self.consumer.start()

    def stop(self):
        self.consumer.stop()


# ── Standalone test ──
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing Kafka integration...")
    print(f"kafka-python available: {KAFKA_AVAILABLE}")

    if KAFKA_AVAILABLE:
        # Test producer
        producer = VIGILKafkaProducer(bootstrap_servers='localhost:9092')
        if producer._connected:
            producer.publish_detection(0, detections=2, violations=1, barriers=0, fps=15.0)
            producer.publish_violation({
                'camera_id': 0,
                'camera_name': 'Camera 0',
                'zone_type': 'restricted',
                'zone_name': 'Test Zone',
            })
            producer.flush()
            print("Published test events")
        else:
            print("Producer not connected (is Kafka running?)")
        producer.close()
    else:
        print("Install kafka-python: pip install kafka-python")

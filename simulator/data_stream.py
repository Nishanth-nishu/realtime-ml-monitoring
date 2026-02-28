"""
Real-time sensor data stream simulator.
Sends synthetic IoT readings to the inference API to generate live metrics.

Run: python simulator/data_stream.py --mode mixed --rps 10
"""

import sys
import argparse
import time
import random
import json
import requests
import numpy as np
import logging
from threading import Thread

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8001"
SENSOR_IDS = [f"sensor_{i:03d}" for i in range(1, 11)]


def normal_reading(sensor_id: str) -> dict:
    """Generate a normal sensor reading."""
    return {
        "sensor_id": sensor_id,
        "temperature": float(np.random.normal(75, 4)),
        "vibration": float(np.random.normal(0.2, 0.04)),
        "pressure": float(np.random.normal(101, 1.5)),
        "current": float(np.random.normal(5.0, 0.25)),
        "voltage": float(np.random.normal(220, 2.5)),
        "frequency": float(np.random.normal(50, 0.8)),
        "power_factor": float(np.clip(np.random.normal(0.9, 0.015), 0.7, 1.0)),
        "humidity": float(np.random.normal(30, 2)),
    }


def anomaly_reading(sensor_id: str, pattern: str = "overheat") -> dict:
    """Generate an anomalous sensor reading."""
    reading = normal_reading(sensor_id)

    if pattern == "overheat":
        reading["temperature"] = float(np.random.normal(115, 10))
        reading["current"] = float(np.random.normal(8.5, 1.0))
    elif pattern == "vibration_spike":
        reading["vibration"] = float(np.random.normal(0.9, 0.2))
        reading["pressure"] = float(np.random.normal(88, 8))
    elif pattern == "power_fault":
        reading["voltage"] = float(np.random.normal(195, 12))
        reading["power_factor"] = float(np.random.normal(0.65, 0.08))
        reading["frequency"] = float(np.random.normal(47, 2))
    elif pattern == "all":
        reading["temperature"] = float(np.random.normal(120, 15))
        reading["vibration"] = float(np.random.normal(1.2, 0.3))
        reading["pressure"] = float(np.random.normal(80, 12))

    return reading


def send_prediction(reading: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}/predict", json=reading, timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API error: {e}")
    return None


def run_stream(mode: str = "mixed", rps: float = 5.0, duration: int = 0):
    """
    Stream sensor data to the API.

    Args:
        mode: "normal" | "anomaly" | "mixed"
        rps: requests per second
        duration: seconds to run (0 = infinite)
    """
    patterns = ["overheat", "vibration_spike", "power_fault", "all"]
    delay = 1.0 / max(rps, 0.1)
    start = time.time()
    n_sent = 0
    n_anomaly = 0

    logger.info(f"Starting stream: mode={mode}, rps={rps}")
    logger.info(f"Target: {API_URL}/predict")

    while True:
        sensor_id = random.choice(SENSOR_IDS)

        if mode == "normal":
            reading = normal_reading(sensor_id)
        elif mode == "anomaly":
            reading = anomaly_reading(sensor_id, random.choice(patterns))
        else:  # mixed: ~10% anomaly
            if random.random() < 0.10:
                reading = anomaly_reading(sensor_id, random.choice(patterns))
            else:
                reading = normal_reading(sensor_id)

        result = send_prediction(reading)
        n_sent += 1

        if result:
            if result.get("is_anomaly"):
                n_anomaly += 1
            if n_sent % 50 == 0:
                elapsed = time.time() - start
                actual_rps = n_sent / elapsed
                anomaly_rate = n_anomaly / n_sent
                logger.info(
                    f"Sent: {n_sent:5d} | "
                    f"Actual RPS: {actual_rps:.1f} | "
                    f"Anomaly rate: {anomaly_rate:.1%} | "
                    f"Last alert: {result.get('alert_level', 'N/A')}"
                )

        if duration > 0 and (time.time() - start) >= duration:
            break

        time.sleep(delay)

    logger.info(f"\nStream complete. Sent {n_sent} readings, {n_anomaly} anomalies ({n_anomaly/n_sent:.1%})")


def main():
    parser = argparse.ArgumentParser("IoT Sensor Data Stream Simulator")
    parser.add_argument("--mode", choices=["normal", "anomaly", "mixed"], default="mixed")
    parser.add_argument("--rps", type=float, default=5.0, help="Requests per second")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=infinite)")
    parser.add_argument("--url", type=str, default="http://localhost:8001")
    args = parser.parse_args()

    global API_URL
    API_URL = args.url

    try:
        run_stream(mode=args.mode, rps=args.rps, duration=args.duration)
    except KeyboardInterrupt:
        logger.info("Stream stopped by user.")


if __name__ == "__main__":
    main()

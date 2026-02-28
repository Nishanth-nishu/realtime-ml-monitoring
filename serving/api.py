"""
Real-time ML inference server with Prometheus metrics.
Exposes anomaly detection predictions + monitoring telemetry.

GCP equivalent: Cloud Run service + Cloud Monitoring
"""

import sys
import os
import json
import time
import pickle
import logging
import asyncio
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
import threading

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.train_anomaly import AnomalyAutoencoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "model" / "artifacts"

# ── Metrics Store (Prometheus-compatible) ────────────────────────────── #

class MetricsStore:
    """In-memory metrics store with Prometheus text format export."""

    def __init__(self):
        self._lock = threading.Lock()
        self.request_count = 0
        self.anomaly_count = 0
        self.normal_count = 0
        self.total_latency_ms = 0.0
        self.last_100_latencies: List[float] = []
        self.last_100_scores: List[float] = []
        self.drift_alerts = 0
        self._start_time = time.time()

    def record_prediction(self, score: float, is_anomaly: bool, latency_ms: float):
        with self._lock:
            self.request_count += 1
            self.total_latency_ms += latency_ms
            if is_anomaly:
                self.anomaly_count += 1
            else:
                self.normal_count += 1

            self.last_100_latencies.append(latency_ms)
            self.last_100_scores.append(score)
            if len(self.last_100_latencies) > 100:
                self.last_100_latencies.pop(0)
            if len(self.last_100_scores) > 100:
                self.last_100_scores.pop(0)

            # Data drift detection: if anomaly rate > 2x expected (5%), trigger alert
            if self.request_count >= 50:
                current_anomaly_rate = self.anomaly_count / self.request_count
                if current_anomaly_rate > 0.15:  # 3x threshold
                    self.drift_alerts += 1

    def to_prometheus(self) -> str:
        with self._lock:
            uptime = time.time() - self._start_time
            avg_latency = (self.total_latency_ms / max(self.request_count, 1))
            p95_latency = float(np.percentile(self.last_100_latencies, 95)) if self.last_100_latencies else 0.0
            anomaly_rate = (self.anomaly_count / max(self.request_count, 1))
            avg_score = float(np.mean(self.last_100_scores)) if self.last_100_scores else 0.0

            lines = [
                "# HELP ml_requests_total Total number of inference requests",
                "# TYPE ml_requests_total counter",
                f"ml_requests_total {self.request_count}",
                "",
                "# HELP ml_anomalies_total Total anomaly predictions",
                "# TYPE ml_anomalies_total counter",
                f"ml_anomalies_total {self.anomaly_count}",
                "",
                "# HELP ml_anomaly_rate Current anomaly rate (0.0-1.0)",
                "# TYPE ml_anomaly_rate gauge",
                f"ml_anomaly_rate {anomaly_rate:.4f}",
                "",
                "# HELP ml_latency_ms_avg Average inference latency in ms",
                "# TYPE ml_latency_ms_avg gauge",
                f"ml_latency_ms_avg {avg_latency:.3f}",
                "",
                "# HELP ml_latency_ms_p95 P95 inference latency in ms",
                "# TYPE ml_latency_ms_p95 gauge",
                f"ml_latency_ms_p95 {p95_latency:.3f}",
                "",
                "# HELP ml_avg_anomaly_score Average anomaly score (0.0-1.0)",
                "# TYPE ml_avg_anomaly_score gauge",
                f"ml_avg_anomaly_score {avg_score:.4f}",
                "",
                "# HELP ml_drift_alerts_total Total data drift alerts triggered",
                "# TYPE ml_drift_alerts_total counter",
                f"ml_drift_alerts_total {self.drift_alerts}",
                "",
                "# HELP ml_uptime_seconds Server uptime in seconds",
                "# TYPE ml_uptime_seconds gauge",
                f"ml_uptime_seconds {uptime:.1f}",
            ]
            return "\n".join(lines)

    def to_dict(self) -> dict:
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "request_count": self.request_count,
                "anomaly_count": self.anomaly_count,
                "normal_count": self.normal_count,
                "anomaly_rate": round(self.anomaly_count / max(self.request_count, 1), 4),
                "avg_latency_ms": round(self.total_latency_ms / max(self.request_count, 1), 3),
                "p95_latency_ms": round(float(np.percentile(self.last_100_latencies, 95)) if self.last_100_latencies else 0, 3),
                "avg_score": round(float(np.mean(self.last_100_scores)) if self.last_100_scores else 0, 4),
                "drift_alerts": self.drift_alerts,
                "uptime_s": round(uptime, 1),
            }


metrics = MetricsStore()


# ── Model State ──────────────────────────────────────────────────────── #

class ModelState:
    ae: Optional[AnomalyAutoencoder] = None
    iso_forest = None
    scaler = None
    threshold: float = 0.05
    feature_cols: List[str] = []
    model_metadata: dict = {}


state = ModelState()

# ── FastAPI App ──────────────────────────────────────────────────────── #

app = FastAPI(
    title="Real-Time ML Anomaly Detection",
    description="Production inference service with Prometheus monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    logger.info("Loading anomaly detection model...")

    meta_path = ARTIFACTS_DIR / "metadata.json"
    if not meta_path.exists():
        logger.warning(f"Model artifacts not found at {ARTIFACTS_DIR}. Run: python model/train_anomaly.py")
        return

    with open(meta_path) as f:
        state.model_metadata = json.load(f)

    state.threshold = state.model_metadata["threshold"]
    state.feature_cols = state.model_metadata["feature_cols"]
    n_features = state.model_metadata["n_features"]

    state.ae = AnomalyAutoencoder(input_dim=n_features)
    state.ae.load_state_dict(torch.load(ARTIFACTS_DIR / "autoencoder.pt", map_location="cpu"))
    state.ae.eval()

    with open(ARTIFACTS_DIR / "iso_forest.pkl", "rb") as f:
        state.iso_forest = pickle.load(f)
    with open(ARTIFACTS_DIR / "scaler.pkl", "rb") as f:
        state.scaler = pickle.load(f)

    logger.info(f"Model loaded! Threshold={state.threshold:.6f}")


# ── Schemas ───────────────────────────────────────────────────────────── #

class SensorReading(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    current: float
    voltage: float
    frequency: float
    power_factor: float
    humidity: float
    sensor_id: str = "sensor_001"

class BatchRequest(BaseModel):
    readings: List[SensorReading]

class PredictionResult(BaseModel):
    sensor_id: str
    is_anomaly: bool
    anomaly_score: float
    ae_score: float
    iso_score: float
    latency_ms: float
    alert_level: str  # "normal", "warning", "critical"


# ── Prediction Logic ─────────────────────────────────────────────────── #

def _predict_single(reading: SensorReading) -> PredictionResult:
    t0 = time.time()

    features = np.array([[
        reading.temperature, reading.vibration, reading.pressure,
        reading.current, reading.voltage, reading.frequency,
        reading.power_factor, reading.humidity,
    ]])

    scaled = state.scaler.transform(features)

    # Autoencoder score
    with torch.no_grad():
        x_t = torch.tensor(scaled, dtype=torch.float32)
        ae_score = float(state.ae.reconstruction_error(x_t).numpy()[0])

    # Isolation Forest score
    iso_raw = -float(state.iso_forest.decision_function(scaled)[0])
    iso_norm = max(0.0, min(1.0, (iso_raw + 0.5)))  # rough normalization

    # Threshold on AE score
    ae_norm = min(1.0, ae_score / (state.threshold * 3 + 1e-8))
    ensemble_score = 0.6 * ae_norm + 0.4 * iso_norm

    is_anomaly = ae_score > state.threshold
    latency_ms = (time.time() - t0) * 1000

    # Alert level
    if ensemble_score > 0.8:
        alert = "critical"
    elif ensemble_score > 0.5:
        alert = "warning"
    else:
        alert = "normal"

    metrics.record_prediction(ensemble_score, is_anomaly, latency_ms)

    return PredictionResult(
        sensor_id=reading.sensor_id,
        is_anomaly=is_anomaly,
        anomaly_score=round(ensemble_score, 4),
        ae_score=round(ae_score, 6),
        iso_score=round(iso_norm, 4),
        latency_ms=round(latency_ms, 3),
        alert_level=alert,
    )


# ── Endpoints ─────────────────────────────────────────────────────────── #

@app.get("/health")
def health():
    return {
        "status": "healthy" if state.ae else "degraded",
        "model_loaded": state.ae is not None,
        "threshold": state.threshold,
    }


@app.post("/predict", response_model=PredictionResult)
def predict(reading: SensorReading):
    if state.ae is None:
        raise HTTPException(503, "Model not loaded. Run: python model/train_anomaly.py")
    return _predict_single(reading)


@app.post("/predict/batch", response_model=List[PredictionResult])
def predict_batch(req: BatchRequest):
    if state.ae is None:
        raise HTTPException(503, "Model not loaded.")
    return [_predict_single(r) for r in req.readings]


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    """Prometheus-compatible /metrics endpoint."""
    return metrics.to_prometheus()


@app.get("/metrics/json")
def metrics_json():
    """JSON metrics for the custom dashboard."""
    return metrics.to_dict()


@app.get("/dashboard", response_class=HTMLResponse)
def live_dashboard():
    """Self-contained live monitoring dashboard."""
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ML Monitoring Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; }
        .header { background: linear-gradient(135deg, #1a237e, #283593); padding: 24px 32px; }
        .header h1 { font-size: 1.6rem; color: #fff; }
        .header p { color: #90caf9; font-size: 0.9rem; margin-top: 4px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; padding: 24px; }
        .card { background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; text-align: center; }
        .card .value { font-size: 2rem; font-weight: 700; }
        .card .label { font-size: 0.8rem; color: #8b949e; margin-top: 6px; }
        .normal { color: #3fb950; }
        .warning { color: #e3b341; }
        .critical { color: #f85149; }
        .info { color: #58a6ff; }
        .section { padding: 0 24px 24px; }
        .section h2 { font-size: 1rem; margin-bottom: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
        .footer { text-align: center; padding: 16px; color: #484f58; font-size: 0.8rem; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
        .badge-green { background: #0d4429; color: #3fb950; }
        .badge-yellow { background: #3d2c01; color: #e3b341; }
        .badge-red { background: #3d0404; color: #f85149; }
    </style>
    <script>
        async function fetchMetrics() {
            const r = await fetch('/metrics/json');
            const d = await r.json();
            document.getElementById('req').textContent = d.request_count.toLocaleString();
            document.getElementById('anoms').textContent = d.anomaly_count.toLocaleString();
            document.getElementById('rate').textContent = (d.anomaly_rate * 100).toFixed(2) + '%';
            document.getElementById('latency').textContent = d.avg_latency_ms.toFixed(2) + ' ms';
            document.getElementById('p95').textContent = d.p95_latency_ms.toFixed(2) + ' ms';
            document.getElementById('score').textContent = d.avg_score.toFixed(4);
            document.getElementById('drift').textContent = d.drift_alerts;
            document.getElementById('uptime').textContent = (d.uptime_s / 60).toFixed(1) + ' min';

            const rateEl = document.getElementById('rate');
            const anomRate = d.anomaly_rate;
            rateEl.className = 'value ' + (anomRate > 0.15 ? 'critical' : anomRate > 0.08 ? 'warning' : 'normal');
        }
        fetchMetrics();
        setInterval(fetchMetrics, 3000);
    </script>
</head>
<body>
    <div class="header">
        <h1>🤖 Real-Time ML Monitoring Dashboard</h1>
        <p>IoT Anomaly Detection | Prometheus metrics | Auto-refreshes every 3s</p>
    </div>
    <div class="grid">
        <div class="card"><div id="req" class="value info">-</div><div class="label">Total Requests</div></div>
        <div class="card"><div id="anoms" class="value warning">-</div><div class="label">Anomalies Detected</div></div>
        <div class="card"><div id="rate" class="value">-</div><div class="label">Anomaly Rate</div></div>
        <div class="card"><div id="latency" class="value normal">-</div><div class="label">Avg Latency</div></div>
        <div class="card"><div id="p95" class="value normal">-</div><div class="label">P95 Latency</div></div>
        <div class="card"><div id="score" class="value info">-</div><div class="label">Avg Anomaly Score</div></div>
        <div class="card"><div id="drift" class="value critical">-</div><div class="label">Drift Alerts</div></div>
        <div class="card"><div id="uptime" class="value normal">-</div><div class="label">Uptime</div></div>
    </div>
    <div class="section">
        <h2>Quick Test</h2>
        <p style="color:#58a6ff; font-size:0.85rem;">
            Normal: <code>curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{"temperature":75,"vibration":0.2,"pressure":101,"current":5.0,"voltage":220,"frequency":50,"power_factor":0.9,"humidity":30}'</code>
        </p>
    </div>
    <div class="footer">Refresh interval: 3s | Prometheus endpoint: <a href="/metrics" style="color:#58a6ff">/metrics</a></div>
</body>
</html>
"""
    return HTMLResponse(content=html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

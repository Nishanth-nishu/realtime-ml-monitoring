# Real-Time ML Inference & Cloud Monitoring Dashboard
> **ATS Keywords:** `cloud computing` · `real-time ML inference` · `monitoring` · `multi-tenant infrastructure` · `data visualization` · `Prometheus` · `Grafana` · `Docker` · `Python` · `anomaly detection`

A **production-ready real-time ML inference pipeline** mimicking Google Cloud's ML operations infrastructure — with anomaly detection, live telemetry, Prometheus metrics scraping, Grafana dashboards, and Docker Compose deployment.

---

## 🏗️ Architecture

```
IoT Sensor Stream
      │
      ▼
┌─────────────────────────────────────────────────┐
│         FastAPI ML Inference Service            │
│  ┌─────────────────────────────────────────┐   │
│  │  Autoencoder + Isolation Forest Ensemble │   │
│  │  Predict → score → alert_level          │   │
│  └─────────────────────────────────────────┘   │
│                                                  │
│  /predict        → inference endpoint           │
│  /metrics        → Prometheus scrape target     │
│  /dashboard      → live HTML monitoring UI      │
└──────────────────────┬──────────────────────────┘
                       │ metrics scrape (15s)
                       ▼
              ┌─────────────────┐
              │   Prometheus    │  ← time-series storage
              │   :9090         │
              └────────┬────────┘
                       │ data source
                       ▼
              ┌─────────────────┐
              │    Grafana      │  ← dashboards, alerts
              │    :3000        │
              └─────────────────┘
```

---

## 📊 What Gets Monitored

| Metric | Description | Alert Level |
|---|---|---|
| `ml_requests_total` | Total inference requests | — |
| `ml_anomaly_rate` | Fraction of anomalous predictions | > 15% = critical |
| `ml_latency_ms_avg` | Average inference latency | > 100ms = warning |
| `ml_latency_ms_p95` | 95th percentile latency | — |
| `ml_avg_anomaly_score` | Mean anomaly confidence score | — |
| `ml_drift_alerts_total` | Detected data distribution shifts | > 0 = alert |

---

## 🚀 Quick Start

### Option A: Without Docker (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train anomaly detection model
python model/train_anomaly.py

# 3. Start inference API
uvicorn serving.api:app --host 0.0.0.0 --port 8001

# 4. Start sensor simulator (in a new terminal)
python simulator/data_stream.py --mode mixed --rps 10

# 5. Open live dashboard
open http://localhost:8001/dashboard

# 6. View Prometheus metrics
curl http://localhost:8001/metrics
```

### Option B: Full Docker Stack (Prometheus + Grafana)

```bash
# 1. Build and start all services
docker-compose up --build

# Services:
# - ML API:     http://localhost:8001
# - Dashboard:  http://localhost:8001/dashboard
# - Prometheus: http://localhost:9090
# - Grafana:    http://localhost:3000 (admin/admin)
```

---

## 🧠 Model: Anomaly Detection Ensemble

| Component | Algorithm | Role |
|---|---|---|
| Autoencoder | PyTorch (8→4→8 latent) | Reconstruction-based scoring |
| Isolation Forest | sklearn | Tree-based outlier scoring |
| Ensemble | Weighted average (0.6 AE + 0.4 IF) | Final anomaly score |

**Input features**: temperature, vibration, pressure, current, voltage, frequency, power factor, humidity

**Result**:
- `anomaly_score`: 0.0–1.0 (higher = more anomalous)
- `alert_level`: "normal" / "warning" / "critical"

---

## 🔌 API Usage

### Single sensor prediction
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "machine_01",
    "temperature": 118.5,
    "vibration": 0.9,
    "pressure": 88.0,
    "current": 8.5,
    "voltage": 198.0,
    "frequency": 48.5,
    "power_factor": 0.72,
    "humidity": 55.0
  }'
```

Response:
```json
{
  "sensor_id": "machine_01",
  "is_anomaly": true,
  "anomaly_score": 0.8731,
  "alert_level": "critical",
  "latency_ms": 1.842
}
```

---

## 📁 Project Structure

```
realtime-ml-monitoring/
├── model/
│   ├── train_anomaly.py       # Train AE + Isolation Forest ensemble
│   └── artifacts/             # Saved model artifacts (auto-generated)
├── serving/
│   └── api.py                 # FastAPI server: /predict, /metrics, /dashboard
├── simulator/
│   └── data_stream.py         # IoT sensor stream simulator
├── monitoring/
│   ├── prometheus.yml         # Prometheus scrape config
│   └── grafana/
│       └── datasources.yml    # Auto-provisioned Prometheus datasource
├── docker-compose.yml         # Full stack: API + Prometheus + Grafana
├── Dockerfile.api             # ML API container
├── Dockerfile.simulator       # Simulator container
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Component | Technology | GCP Equivalent |
|---|---|---|
| ML Model | PyTorch Autoencoder + sklearn | Vertex AI Anomaly Detection |
| Inference API | FastAPI + Uvicorn | Cloud Run |
| Metrics | Prometheus-compatible /metrics | Cloud Monitoring |
| Dashboard | Grafana | Cloud Monitoring Dashboards |
| Orchestration | Docker Compose | Cloud Run + Cloud Tasks |
| Data Stream | Custom simulator | Pub/Sub |

Feel Free To reach out

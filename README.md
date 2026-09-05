# RiskGraph

RiskGraph is a temporal, graph-aware merchant abuse-risk console. It wraps the completed notebook artifacts with a FastAPI service and a focused React operations UI. Events are facts, signals are derived observations, and risk is a calibrated model estimate; shared infrastructure is never treated as proof of identity or intent.

## Source-of-truth ML system

The saved package is kept unchanged in `riskgraph_download/riskgraph_final_model`. The notebook establishes 97 ordered features: temporal behavioral aggregates, point-in-time NetworkX features, saved 32-dimensional temporal GraphSAGE embeddings, Isolation Forest anomaly score, XGBoost, sigmoid calibration, and SHAP. Relationships use `first_seen <= prediction_time`; `last_seen`, `event_count`, labels, and future transactions are excluded from point-in-time construction.

The verified decision policy is probability bands LOW `<0.20`, MEDIUM `<0.50`, HIGH `<0.80`, CRITICAL otherwise. The selected cost threshold is `0.05`, with FP cost ₹20 and FN cost ₹1,000. Expected loss is calibrated abuse probability × historical refund-plus-discount exposure. Intervention logic is preserved from the notebook.

## Run locally

From `RiskGraph`:

```powershell
$env:PYTHONPATH="backend"
uvicorn app.main:app --reload --port 8000
```

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The API is at http://localhost:8000/docs. Use `.env.example` to configure paths. Docker Compose is also included.

## API

`GET /api/v1/health`, `GET /api/v1/model/info`, `GET /api/v1/risk/queue`, `GET /api/v1/risk/customer/{customer_id}`, `POST /api/v1/risk/score`, `GET /api/v1/network/customer/{customer_id}`, `GET /api/v1/analytics/overview`, `GET /api/v1/analytics/trends`, and `GET /api/v1/customers/search`.

## Limitations and demo flow

The supplied notebook output contains customer-level test-snapshot predictions, not a complete serialized feature builder or time-series aggregate table. The app therefore serves those saved, real predictions for the queue and uses raw data for point-in-time behavioral/network evidence. It reports when the native XGBoost/PyTorch runtime cannot load; it never substitutes a fake model. SHAP contributions and live arbitrary-time GraphSAGE inference require the exact native runtime and a complete feature-builder execution path.

Demo: Overview → Risk Queue → select a high-loss account → review probability, expected loss, behavior, network evidence, and proportionate action → use the caveat that shared infrastructure is a signal, not proof.

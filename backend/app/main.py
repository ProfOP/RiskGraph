from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .ml.inference import engine, FEATURES, CONFIG
from .config import OUTPUT_DIR
from .schemas import ScoreRequest
app=FastAPI(title="RiskGraph API",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
@app.get("/api/v1/health")
def health():return {"status":"ok","model_ready":engine.ready,"model_error":engine.model_error}
@app.get("/api/v1/model/info")
def model_info():
    evaluation={}
    try:
        import pandas as pd
        cost=pd.read_csv(OUTPUT_DIR/"cost_analysis.csv")
        best=cost.loc[cost.total_cost.idxmin()].to_dict(); evaluation={"validation_cost_analysis":best,"available_metrics":["precision","recall","false positives","false negatives","expected cost"]}
    except Exception: evaluation={"available_metrics":[],"note":"Evaluation CSV unavailable"}
    return {"config":CONFIG,"feature_count":len(FEATURES),"features":FEATURES,"model_ready":engine.ready,"model_error":engine.model_error,"evaluation":evaluation,"temporal_split":{"train":"2025-08-01","validation":["2025-09-01","2025-10-01"],"test":["2025-11-01","2025-12-01"]}}
@app.get("/api/v1/risk/queue")
def queue(search:str="",band:str="",intervention:str="",limit:int=50,offset:int=0):return engine.queue(search,band,intervention,max(1,min(limit,200)),max(offset,0))
@app.get("/api/v1/risk/customer/{customer_id}")
def customer(customer_id:str):
    try:return engine.score(customer_id)
    except KeyError:raise HTTPException(404,"Customer not found in saved prediction artifact")
@app.post("/api/v1/risk/score")
def score(req:ScoreRequest):
    try:return engine.score(req.customer_id,req.prediction_time)
    except KeyError:raise HTTPException(404,"Customer not found")
@app.get("/api/v1/network/customer/{customer_id}")
def network(customer_id:str):return engine.network(customer_id)
@app.get("/api/v1/analytics/overview")
def overview():
    d=engine.predictions; return {"customers_monitored":len(d),"high_risk":int(d.risk_band.isin(["HIGH","CRITICAL"]).sum()),"critical":int((d.risk_band=="CRITICAL").sum()),"expected_loss":float(d.expected_loss.sum()),"historical_exposure":float(d.historical_exposure.sum()),"risk_bands":d.risk_band.value_counts().to_dict(),"interventions":d.recommended_action.value_counts().to_dict(),"source":"saved notebook output artifact"}
@app.get("/api/v1/analytics/trends")
def trends():return {"items":[],"note":"No time-series aggregate artifact was saved; values are intentionally omitted."}
@app.get("/api/v1/customers/search")
def search(q:str="",limit:int=20):return {"items":engine.predictions[engine.predictions.customer_id.str.contains(q,case=False)].head(limit).customer_id.tolist()}

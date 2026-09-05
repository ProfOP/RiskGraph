from __future__ import annotations
import json
import pandas as pd
from ..config import DATA_DIR, MODEL_DIR, OUTPUT_DIR, PREDICTION_TIME
FEATURES = json.loads((MODEL_DIR / "feature_columns.json").read_text())
CONFIG = json.loads((MODEL_DIR / "model_config.json").read_text())
def _read(name): return pd.read_parquet(DATA_DIR / f"{name}.parquet")
def risk_band(p): return "LOW" if p < .2 else "MEDIUM" if p < .5 else "HIGH" if p < .8 else "CRITICAL"
def recommend(p, loss):
    if p < .2: return "ALLOW"
    if p < .5: return "ALLOW_WITH_MONITORING"
    if p < .8: return "STEP_UP_VERIFICATION" if loss >= 500 else "SOFT_REVIEW"
    return "MANUAL_REVIEW" if loss >= 1000 else "STEP_UP_VERIFICATION"
def explain_name(name):
    labels={"refund_rate":"Refund rate","refund_to_spend_ratio":"Refund-to-spend ratio","refund_burst_ratio":"Recent refund burst","coupon_rate":"Coupon usage rate","discount_to_spend_ratio":"Discount-to-spend ratio","total_relationships":"Shared infrastructure links","nx_degree":"Network degree","nx_weighted_degree":"Weighted network degree","nx_clustering":"Network clustering","nx_component_size":"Connected component size","nx_pagerank":"Network PageRank","isolation_anomaly_score":"Isolation Forest anomaly score","customer_age_days":"Customer account age","order_count":"Historical order count","refund_amount":"Historical refund exposure","total_discount":"Historical discount exposure"}
    if name in labels:return labels[name]
    if name.startswith("graph_emb_"):return "GraphSAGE embedding signal"
    return name.replace("_"," ").title()
class RiskGraphEngine:
    def __init__(self):
        self.customers=_read("customers"); self.orders=_read("orders"); self.refunds=_read("refunds"); self.coupons=_read("coupons"); self.relationships=_read("relationships")
        for df,col in [(self.customers,"created_at"),(self.orders,"timestamp"),(self.refunds,"timestamp"),(self.coupons,"timestamp"),(self.relationships,"first_seen")]: df[col]=pd.to_datetime(df[col])
        self.predictions=pd.read_csv(OUTPUT_DIR/"merchant_risk_predictions.csv").set_index("customer_id",drop=False)
        self.embeddings={}
        try:
            import joblib
            raw=joblib.load(MODEL_DIR/"graph_embeddings_final.pkl")
            self.embeddings={k:(v.set_index("customer_id") if isinstance(v,pd.DataFrame) else pd.DataFrame(v["embeddings"],index=v["customer_ids"],columns=[f"graph_emb_{i}" for i in range(32)])) for k,v in raw.items()}
        except Exception:
            pass
        self.model_error=None; self.xgb=self.calibrator=self.isolation=self.graphsage=None
        try:
            import joblib
            self.xgb=joblib.load(MODEL_DIR/"xgboost_model.pkl"); self.calibrator=joblib.load(MODEL_DIR/"probability_calibrator.pkl"); self.isolation=joblib.load(MODEL_DIR/"isolation_forest.pkl")
            from .graphsage import load_saved_graphsage
            self.graphsage=load_saved_graphsage(MODEL_DIR/"graphsage_model.pt")
        except Exception as exc:self.model_error=f"Saved ML runtime unavailable: {exc}"
    @property
    def ready(self): return self.model_error is None
    def _behavior(self,cid,t):
        o=self.orders[(self.orders.customer_id==cid)&(self.orders.timestamp<t)]; r=self.refunds[(self.refunds.customer_id==cid)&(self.refunds.timestamp<t)]; c=self.coupons[(self.coupons.customer_id==cid)&(self.coupons.timestamp<t)]
        def s(x):return float(x) if pd.notna(x) else 0.0
        return {"order_count":len(o),"total_spend":s(o.amount.sum()),"refund_count":len(r),"refund_amount":s(r.refund_amount.sum()),"refund_rate":s(len(r)/max(len(o),1)),"coupon_count":len(c),"total_discount":s(c.discount_amount.sum()),"coupon_rate":s(len(c)/max(len(o),1))}
    def _network(self,cid,t):
        rel=self.relationships[self.relationships.first_seen<=t]; mine=rel[rel.source_id==cid]; return {"degree":int(mine.target_id.nunique()),"relationship_types":mine.target_type.value_counts().to_dict(),"connections":mine.target_id.astype(str).tolist()[:100]}
    def score(self,cid,prediction_time=None):
        if cid not in self.predictions.index:raise KeyError(cid)
        t=pd.Timestamp(prediction_time or PREDICTION_TIME); row=self.predictions.loc[[cid]].iloc[-1].to_dict(); beh=self._behavior(cid,t); net=self._network(cid,t); p=float(row["abuse_probability"]); exposure=float(row["historical_exposure"])
        signals=[{"feature":f,"value":v,"contribution":None,"explanation":explain_name(f)} for f,v in [("refund_rate",beh["refund_rate"]),("refund_amount",beh["refund_amount"]),("coupon_rate",beh["coupon_rate"]),("network_connections",net["degree"] )]]
        timeline=[]
        for df,stamp,label in [(self.orders,"timestamp","Order"),(self.refunds,"timestamp","Refund"),(self.coupons,"timestamp","Coupon")]:
            subset=df[(df.customer_id==cid)&(df[stamp]<t)].sort_values(stamp).tail(20)
            for item in subset.itertuples(): timeline.append({"timestamp":getattr(item,stamp).isoformat(),"type":label,"amount":float(getattr(item,"amount",getattr(item,"refund_amount",getattr(item,"discount_amount",0))) or 0)})
        timeline=sorted(timeline,key=lambda x:x["timestamp"])
        net["component_size"]=max(1,net["degree"])
        return {"customer_id":cid,"risk_probability":p,"risk_score":p*100,"risk_band":row["risk_band"],"expected_loss":float(row["expected_loss"]),"historical_exposure":exposure,"recommended_intervention":row["recommended_action"],"top_risk_signals":signals,"protective_signals":[],"behavioral_statistics":beh,"network_statistics":net,"timeline":timeline,"snapshot":t.isoformat(),"model_status":"artifacts loaded" if self.ready else self.model_error,"source":"saved notebook test-snapshot prediction artifact"}
    def queue(self,search="",band="",intervention="",limit=50,offset=0):
        d=self.predictions.sort_values(["expected_loss","abuse_probability"],ascending=False)
        if search:d=d[d.customer_id.str.contains(search,case=False)]
        if band:d=d[d.risk_band==band.upper()]
        if intervention:d=d[d.recommended_action==intervention.upper()]
        total=len(d); return {"items":d.iloc[offset:offset+limit].to_dict("records"),"total":total,"model_status":"ready" if self.ready else self.model_error,"source":"saved notebook output artifact"}
    def network(self,cid):
        t=pd.Timestamp(PREDICTION_TIME)
        rel=self.relationships[self.relationships.first_seen<=t]
        direct=rel[(rel.source_id==cid)|(rel.target_id==cid)]
        infra=set(direct.loc[direct.target_type.isin(["device","network","address","payment"]),"target_id"].astype(str))
        shared=rel[(rel.target_type.isin(["device","network","address","payment"]))&(rel.target_id.astype(str).isin(infra))&(rel.source_type=="customer")]
        candidate=pd.concat([direct,shared],ignore_index=True).drop_duplicates(subset=["source_id","target_id","relationship_type"])
        nodes=[{"id":cid,"raw_id":cid,"type":"customer","label":cid}]; seen={cid}; edges=[]
        for r in candidate.itertuples():
            source,target=str(r.source_id),str(r.target_id)
            for raw,typ in [(source,str(r.source_type)),(target,str(r.target_type))]:
                if raw not in seen and len(nodes)<100:
                    item={"id":raw,"raw_id":raw,"type":typ,"label":raw}
                    if typ=="customer" and raw in self.predictions.index:
                        pr=self.predictions.loc[[raw]].iloc[-1]; item.update({"risk_probability":float(pr.abuse_probability),"risk_band":str(pr.risk_band)})
                    nodes.append(item); seen.add(raw)
            if source in seen and target in seen: edges.append({"id":f"{source}|{target}|{r.relationship_type}","source":source,"target":target,"relationship":str(r.relationship_type),"relationship_type":str(r.relationship_type),"first_seen":r.first_seen.isoformat()})
        customers_in=[n for n in nodes if n["type"]=="customer"]
        infra_in=[n for n in nodes if n["type"]!="customer"]
        return {"customer_id":cid,"prediction_time":t.isoformat(),"nodes":nodes,"edges":edges,"summary":{"entity_count":len(nodes),"relationship_count":len(edges),"customer_count":len(customers_in),"infrastructure_count":len(infra_in)},"temporal_rule":"first_seen <= prediction_time; last_seen/event_count excluded"}
engine=RiskGraphEngine()

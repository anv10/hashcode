import os, json, time, subprocess, requests, pandas as pd
from datetime import datetime

PROM_URL = "http://localhost:9090"   # adjust if you used 9091
EXPORT_INTERVAL = 60                 # seconds between pulls

def get_jaeger_http():
    try:
        host_port = subprocess.check_output(
            ["docker", "port", "jaeger", "16686/tcp"], text=True
        ).strip()
        return "http://" + host_port
    except Exception:
        return "http://localhost:16686"

def query_prom(q):
    r = requests.get(f"{PROM_URL}/api/v1/query", params={"query": q}, timeout=10)
    return r.json()["data"]["result"]

def get_metrics():
    q = "rate(http_server_request_duration_seconds_count[1m])"
    res = query_prom(q)
    now = datetime.utcnow().isoformat()
    rows=[]
    for d in res:
        svc = d["metric"].get("service_name") or d["metric"].get("service") or d["metric"].get("job") or "unknown"
        val = float(d["value"][1])
        rows.append({"timestamp": now, "node": svc, "req_rate": val})
    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ no metrics")
    return df

def get_causal_graph():
    jb = get_jaeger_http()
    try:
        services = requests.get(f"{jb}/api/services", timeout=10).json()["data"]
    except:
        services = []
    edges=set(); nodes=set(services)
    for svc in services[:10]:
        traces=requests.get(f"{jb}/api/traces", params={"service":svc,"limit":10}, timeout=10).json().get("data",[])
        for tr in traces:
            procs=tr.get("processes",{}); spans=tr.get("spans",[])
            p2s={pid:(info.get("serviceName") or "unknown") for pid,info in procs.items()}
            id2={s["spanID"]:s for s in spans}
            for s in spans:
                child=p2s.get(s.get("processID"),"unknown")
                for ref in s.get("references",[]):
                    if ref.get("refType") in ("CHILD_OF","FOLLOWS_FROM"):
                        parent=id2.get(ref.get("spanID"))
                        if parent:
                            par=p2s.get(parent.get("processID"),"unknown")
                            if par!=child: edges.add((par,child)); nodes.update([par,child])
    cg={"nodes":[{"id":n,"label":n} for n in sorted(nodes)],
        "edges":[{"source":s,"target":t,"confidence":1.0} for s,t in sorted(edges)],
        "metadata":{"total_nodes":len(nodes),"total_edges":len(edges)}}
    with open("causal_graph.json","w") as f: json.dump(cg,f,indent=2)

def label_from_req_rate(df):
    if df.empty: return pd.DataFrame()
    q95=df["req_rate"].quantile(0.95)
    df["label"]=(df["req_rate"]>q95).astype(int)
    return df[["timestamp","node","label"]]

def loop():
    while True:
        metrics=get_metrics()
        if not metrics.empty:
            labels=label_from_req_rate(metrics)
            # append or create
            if not os.path.exists("node_metrics.csv"):
                metrics.to_csv("node_metrics.csv",index=False)
            else:
                metrics.to_csv("node_metrics.csv",mode="a",header=False,index=False)
            labels.to_csv("labels.csv",mode="a",header=not os.path.exists("labels.csv"),index=False)
            get_causal_graph()
            print(f"[{datetime.now().isoformat()}] ✅ exported {len(metrics)} metrics")
        time.sleep(EXPORT_INTERVAL)

if __name__=="__main__":
    loop()

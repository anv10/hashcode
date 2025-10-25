import pandas as pd
import numpy as np
import networkx as nx
import json, time, itertools
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from datetime import datetime

# ============================================================
#     AION Live Causal Engine - Compatible with OTel Export
# ============================================================

def enforce_dag(G: nx.DiGraph) -> nx.DiGraph:
    """Ensure the graph is a DAG by removing cycles."""
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[0][0], cycle[0][1])
        except nx.NetworkXNoCycle:
            break
    return G


def discover_causal_graph(df: pd.DataFrame):
    """Lightweight correlation + regression-based causal discovery."""
    # Drop completely empty columns, forward/backward fill, and keep only numeric
    df = df.dropna(axis=1, how="all").ffill().bfill()

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in node_metrics.csv")

    # Normalize numeric data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)


    G = nx.DiGraph()
    G.add_nodes_from(X.columns)
    confidence = {}

    cols = list(X.columns)
    for a, b in itertools.permutations(cols, 2):
        try:
            corr, _ = pearsonr(X[a], X[b])
            if abs(corr) < 0.6:
                continue
            x, y = X[a].values.reshape(-1, 1), X[b].values
            model = LinearRegression().fit(x, y)
            resid1 = np.std(y - model.predict(x))
            model_rev = LinearRegression().fit(y.reshape(-1, 1), X[a])
            resid2 = np.std(X[a] - model_rev.predict(y.reshape(-1, 1)))
            if resid1 < resid2:
                G.add_edge(a, b)
                confidence[(a, b)] = round(abs(corr), 2)
            else:
                G.add_edge(b, a)
                confidence[(b, a)] = round(abs(corr), 2)
        except Exception:
            continue

    G = enforce_dag(G)
    return G, confidence


def save_causal_graph(G: nx.DiGraph, confidence: dict, out_path="causal_graph.json"):
    """Save graph to JSON."""
    edges = []
    for u, v in G.edges():
        edges.append([u, v, float(confidence.get((u, v), 0.5))])
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "edges": edges}, f, indent=2)
    print(f"✅ Causal graph saved → {out_path}  ({len(edges)} edges)")


def run_live_causal_engine(csv_path="node_metrics.csv", out_json="causal_graph.json",
                           interval=30, iterations=10):
    """Run causal discovery repeatedly for live OTel data."""
    print("\n" + "="*80)
    print(" 🧠  AION LIVE CAUSAL ENGINE - OpenTelemetry Integration")
    print("="*80)
    print(f"Watching: {csv_path}")
    print(f"Will refresh every {interval}s for {iterations} cycles\n")

    for i in range(iterations):
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
            if "timestamp" in df.columns:
                df = df.drop(columns=["timestamp"])
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle {i+1}/{iterations}")
            print(f"  → Loaded {df.shape[0]} rows × {df.shape[1]} metrics")

            G, conf = discover_causal_graph(df)
            print(f"  → Discovered {len(G.nodes())} nodes, {len(G.edges())} edges")
            if conf:
                avg_conf = np.mean(list(conf.values()))
                print(f"  → Avg. confidence: {avg_conf:.2f}")

                # Show top causal edges
                sorted_edges = sorted(conf.items(), key=lambda x: x[1], reverse=True)
                print("  → Top correlated pairs:")
                for (a, b), score in sorted_edges[:10]:
                    print(f"     {a}  →  {b}   (conf={score:.2f})")

                # Save causal edges as CSV
                pd.DataFrame(
                    [{"source": a, "target": b, "confidence": score} for (a, b), score in sorted_edges]
                ).to_csv("causal_edges.csv", index=False)
                print("  → Exported edge list → causal_edges.csv")

            else:
                print("  → No strong edges found")

            save_causal_graph(G, conf, out_json)

            save_causal_graph(G, conf, out_json)
        except Exception as e:
            print(f"⚠️  Error in cycle {i+1}: {e}")

        if i < iterations - 1:
            time.sleep(interval)

    print("\n✅ Completed all cycles. Exiting causal engine.")


if __name__ == "__main__":
    # Customize how many refreshes you want here ↓
    run_live_causal_engine(
        csv_path="node_metrics.csv",
        out_json="causal_graph.json",
        interval=30,   # seconds between refreshes
        iterations=10  # number of refresh cycles
    )

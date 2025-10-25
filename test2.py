import numpy as np
import pandas as pd
import networkx as nx
from causallearn.search.ScoreBased.GES import ges
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict

# ============================================================
#     AION Causal Engine - Sock Shop Hackathon Demo
# ============================================================

class SockShopDataGenerator:
    """
    Generates realistic Sock Shop microservices telemetry data
    with embedded causal failure scenarios
    """
    
    @staticmethod
    def generate_normal_operations(time_windows: int = 200) -> pd.DataFrame:
        """Generate baseline normal operations data"""
        np.random.seed(42)
        
        data = pd.DataFrame({
            # Frontend Service
            'frontend_cpu': np.random.normal(35, 5, time_windows),
            'frontend_memory': np.random.normal(45, 6, time_windows),
            'frontend_response_time': np.random.normal(150, 20, time_windows),
            'frontend_error_rate': np.random.poisson(2, time_windows),
            
            # Catalogue Service
            'catalogue_cpu': np.random.normal(40, 6, time_windows),
            'catalogue_latency': np.random.normal(80, 12, time_windows),
            'catalogue_errors': np.random.poisson(1, time_windows),
            
            # Catalogue DB
            'catalogue_db_cpu': np.random.normal(50, 8, time_windows),
            'catalogue_db_connections': np.random.normal(20, 3, time_windows),
            'catalogue_db_query_time': np.random.normal(30, 5, time_windows),
            
            # Cart Service
            'cart_cpu': np.random.normal(30, 5, time_windows),
            'cart_memory': np.random.normal(55, 7, time_windows),
            'cart_session_count': np.random.normal(100, 15, time_windows),
            
            # Orders Service
            'orders_cpu': np.random.normal(38, 6, time_windows),
            'orders_queue_depth': np.random.normal(10, 3, time_windows),
            'orders_processing_time': np.random.normal(200, 25, time_windows),
            
            # Payment Service
            'payment_cpu': np.random.normal(25, 4, time_windows),
            'payment_latency': np.random.normal(120, 18, time_windows),
            'payment_failures': np.random.poisson(1, time_windows),
            
            # Shipping Service
            'shipping_cpu': np.random.normal(20, 4, time_windows),
            'shipping_queue_depth': np.random.normal(5, 2, time_windows),
            
            # User Service
            'user_cpu': np.random.normal(28, 5, time_windows),
            'user_db_latency': np.random.normal(40, 8, time_windows),
        })
        
        return data
    
    @staticmethod
    def inject_database_bottleneck_scenario(data: pd.DataFrame, 
                                            start_idx: int = 80, 
                                            duration: int = 40) -> pd.DataFrame:
        """
        SCENARIO 1: Catalogue DB bottleneck cascades through the system
        Root Cause: Database CPU spike
        """
        end_idx = start_idx + duration - 1  # Fix: subtract 1 to match duration
        actual_duration = end_idx - start_idx + 1
        
        # Root cause: DB CPU spike
        data.loc[start_idx:end_idx, 'catalogue_db_cpu'] += np.random.normal(35, 5, actual_duration)
        
        # Immediate effect: DB query time increases
        db_cpu_values = data.loc[start_idx:end_idx, 'catalogue_db_cpu'].values
        data.loc[start_idx:end_idx, 'catalogue_db_query_time'] += 0.8 * db_cpu_values + np.random.normal(40, 8, actual_duration)
        
        # Cascade: DB connections pool exhausted
        query_time_values = data.loc[start_idx:end_idx, 'catalogue_db_query_time'].values
        data.loc[start_idx:end_idx, 'catalogue_db_connections'] += 0.4 * query_time_values + np.random.normal(25, 5, actual_duration)
        
        # Catalogue service impacted
        data.loc[start_idx:end_idx, 'catalogue_latency'] += 0.7 * query_time_values + np.random.normal(50, 10, actual_duration)
        
        catalogue_latency_values = data.loc[start_idx:end_idx, 'catalogue_latency'].values
        data.loc[start_idx:end_idx, 'catalogue_cpu'] += 0.3 * catalogue_latency_values + np.random.normal(15, 3, actual_duration)
        
        # Frontend impacted (waits for catalogue)
        data.loc[start_idx:end_idx, 'frontend_response_time'] += 0.6 * catalogue_latency_values + np.random.normal(80, 15, actual_duration)
        data.loc[start_idx:end_idx, 'frontend_error_rate'] += np.random.poisson(5, actual_duration)
        
        return data
    
    @staticmethod
    def inject_payment_failure_scenario(data: pd.DataFrame,
                                       start_idx: int = 140,
                                       duration: int = 30) -> pd.DataFrame:
        """
        SCENARIO 2: Payment service failures cause order processing backlog
        Root Cause: Payment service failures
        """
        end_idx = start_idx + duration - 1  # Fix: subtract 1 to match duration
        actual_duration = end_idx - start_idx + 1
        
        # Root cause: Payment failures spike
        data.loc[start_idx:end_idx, 'payment_failures'] += np.random.poisson(8, actual_duration)
        
        # Payment latency increases due to retries
        payment_failures_values = data.loc[start_idx:end_idx, 'payment_failures'].values
        data.loc[start_idx:end_idx, 'payment_latency'] += 0.9 * payment_failures_values * 20 + np.random.normal(100, 20, actual_duration)
        
        # Payment CPU increases (retry logic)
        payment_latency_values = data.loc[start_idx:end_idx, 'payment_latency'].values
        data.loc[start_idx:end_idx, 'payment_cpu'] += 0.5 * payment_latency_values + np.random.normal(20, 5, actual_duration)
        
        # Orders queue backs up
        data.loc[start_idx:end_idx, 'orders_queue_depth'] += 0.8 * payment_failures_values + np.random.normal(20, 5, actual_duration)
        
        orders_queue_values = data.loc[start_idx:end_idx, 'orders_queue_depth'].values
        data.loc[start_idx:end_idx, 'orders_processing_time'] += 0.6 * orders_queue_values + np.random.normal(100, 20, actual_duration)
        
        # Orders CPU increases
        data.loc[start_idx:end_idx, 'orders_cpu'] += 0.4 * orders_queue_values + np.random.normal(25, 5, actual_duration)
        
        # Shipping delayed
        data.loc[start_idx:end_idx, 'shipping_queue_depth'] += 0.5 * orders_queue_values + np.random.normal(10, 3, actual_duration)
        
        return data
    
    @staticmethod
    def inject_memory_leak_scenario(data: pd.DataFrame,
                                    start_idx: int = 50,
                                    duration: int = 60) -> pd.DataFrame:
        """
        SCENARIO 3: Cart service memory leak causes cascading failures
        Root Cause: Cart memory leak
        """
        end_idx = start_idx + duration - 1  # Fix: subtract 1 to match duration
        actual_duration = end_idx - start_idx + 1
        
        # Root cause: Gradual memory leak in cart service
        memory_increase = np.linspace(0, 50, actual_duration)
        data.loc[start_idx:end_idx, 'cart_memory'] += memory_increase + np.random.normal(0, 3, actual_duration)
        
        # Cart CPU increases (GC pressure)
        cart_mem_diff = data.loc[start_idx:end_idx, 'cart_memory'].values - 55
        data.loc[start_idx:end_idx, 'cart_cpu'] += 0.7 * cart_mem_diff + np.random.normal(10, 3, actual_duration)
        
        # Session count drops (pods restart)
        data.loc[start_idx:end_idx, 'cart_session_count'] -= 0.3 * cart_mem_diff + np.random.normal(10, 5, actual_duration)
        
        # Frontend errors increase (lost sessions)
        data.loc[start_idx:end_idx, 'frontend_error_rate'] += np.random.poisson(3, actual_duration)
        
        return data
    
    @classmethod
    def generate_hackathon_demo_data(cls, time_windows: int = 200) -> pd.DataFrame:
        """
        Generate complete Sock Shop demo data with multiple failure scenarios
        """
        print("\n[Data Generation] Creating Sock Shop telemetry with failure scenarios...")
        
        # Start with normal operations
        data = cls.generate_normal_operations(time_windows)
        
        # Inject realistic failure scenarios
        data = cls.inject_memory_leak_scenario(data)
        data = cls.inject_database_bottleneck_scenario(data)
        data = cls.inject_payment_failure_scenario(data)
        
        print(f"  ✓ Generated {time_windows} time windows")
        print(f"  ✓ Injected 3 failure scenarios:")
        print(f"    - Memory leak in Cart service (t=50-110)")
        print(f"    - Database bottleneck (t=80-120)")
        print(f"    - Payment failures (t=140-170)")
        
        return data


class SockShopAlertGenerator:
    """Generate production-ready alerts for Sock Shop incidents"""
    
    def __init__(self, causal_dag: nx.DiGraph):
        self.causal_dag = causal_dag
        self.service_map = {
            'frontend': ['frontend_cpu', 'frontend_memory', 'frontend_response_time', 'frontend_error_rate'],
            'catalogue': ['catalogue_cpu', 'catalogue_latency', 'catalogue_errors'],
            'catalogue_db': ['catalogue_db_cpu', 'catalogue_db_connections', 'catalogue_db_query_time'],
            'cart': ['cart_cpu', 'cart_memory', 'cart_session_count'],
            'orders': ['orders_cpu', 'orders_queue_depth', 'orders_processing_time'],
            'payment': ['payment_cpu', 'payment_latency', 'payment_failures'],
            'shipping': ['shipping_cpu', 'shipping_queue_depth'],
            'user': ['user_cpu', 'user_db_latency']
        }
    
    def identify_affected_services(self, anomalous_metrics: list) -> dict:
        """Map anomalous metrics to affected services"""
        affected = defaultdict(list)
        
        for metric in anomalous_metrics:
            for service, metrics in self.service_map.items():
                if metric in metrics:
                    affected[service].append(metric)
        
        return dict(affected)
    
    def trace_to_root_cause(self, anomalous_metrics: list) -> list:
        """Trace anomalies back to root causes using causal DAG"""
        root_causes = []
        
        for metric in anomalous_metrics:
            if metric not in self.causal_dag:
                continue
            
            ancestors = nx.ancestors(self.causal_dag, metric)
            
            if not ancestors:  # This is a root cause
                descendants = list(nx.descendants(self.causal_dag, metric))
                root_causes.append({
                    'root_cause_metric': metric,
                    'root_cause_service': self._get_service_name(metric),
                    'affected_metrics': descendants,
                    'affected_services': list(set([self._get_service_name(m) for m in descendants])),
                    'severity': 'CRITICAL' if len(descendants) > 5 else 'HIGH'
                })
            else:
                # Find root nodes among ancestors
                root_ancestors = [a for a in ancestors if self.causal_dag.in_degree(a) == 0]
                for root in root_ancestors:
                    path = nx.shortest_path(self.causal_dag, root, metric)
                    root_causes.append({
                        'root_cause_metric': root,
                        'root_cause_service': self._get_service_name(root),
                        'affected_metric': metric,
                        'affected_service': self._get_service_name(metric),
                        'causal_chain': ' → '.join(path),
                        'severity': 'MEDIUM'
                    })
        
        # Deduplicate by root cause
        seen_roots = set()
        unique_causes = []
        for rc in root_causes:
            if rc['root_cause_metric'] not in seen_roots:
                seen_roots.add(rc['root_cause_metric'])
                unique_causes.append(rc)
        
        return unique_causes
    
    def _get_service_name(self, metric: str) -> str:
        """Extract service name from metric"""
        for service, metrics in self.service_map.items():
            if metric in metrics:
                return service
        return 'unknown'
    
    def generate_runbook_recommendation(self, root_cause_metric: str) -> dict:
        """Generate actionable runbook recommendations"""
        runbooks = {
            'catalogue_db_cpu': {
                'immediate_action': 'Scale up Catalogue-DB replicas',
                'investigation': 'Check for slow queries and missing indexes',
                'commands': [
                    'kubectl scale deployment catalogue-db --replicas=3',
                    'kubectl exec -it catalogue-db -- mysql -e "SHOW PROCESSLIST"'
                ]
            },
            'payment_failures': {
                'immediate_action': 'Check external payment gateway status',
                'investigation': 'Review payment service logs for API errors',
                'commands': [
                    'kubectl logs -l name=payment --tail=100',
                    'curl -X GET http://payment/health'
                ]
            },
            'cart_memory': {
                'immediate_action': 'Restart cart pods to clear memory leak',
                'investigation': 'Check for session store memory issues',
                'commands': [
                    'kubectl rollout restart deployment cart',
                    'kubectl top pods -l name=cart'
                ]
            },
            'frontend_response_time': {
                'immediate_action': 'Check downstream service health',
                'investigation': 'Trace requests to identify bottleneck',
                'commands': [
                    'kubectl get pods -l name=frontend',
                    'kubectl logs -l name=frontend --tail=50'
                ]
            }
        }
        
        # Match runbook to metric
        for key, runbook in runbooks.items():
            if key in root_cause_metric:
                return runbook
        
        return {
            'immediate_action': f'Investigate {root_cause_metric} anomaly',
            'investigation': 'Review service logs and metrics',
            'commands': [f'kubectl logs -l name={self._get_service_name(root_cause_metric)}']
        }
    
    def generate_incident_report(self, root_causes: list, confidence_scores: dict) -> dict:
        """Generate comprehensive incident report for Sock Shop"""
        if not root_causes:
            return {
                'status': 'HEALTHY',
                'timestamp': datetime.now().isoformat(),
                'message': 'All Sock Shop services operating normally'
            }
        
        incidents = []
        for rc in root_causes:
            runbook = self.generate_runbook_recommendation(rc['root_cause_metric'])
            
            incident = {
                'incident_id': f"SOCKSHOP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'severity': rc['severity'],
                'root_cause': {
                    'service': rc['root_cause_service'],
                    'metric': rc['root_cause_metric'],
                    'confidence': confidence_scores.get(
                        (rc['root_cause_metric'], rc.get('affected_metric', '')), 0.85
                    )
                },
                'impact': {
                    'affected_services': rc.get('affected_services', [rc.get('affected_service')]),
                    'affected_metrics': rc.get('affected_metrics', [rc.get('affected_metric')])
                },
                'causal_chain': rc.get('causal_chain', f"Root: {rc['root_cause_metric']}"),
                'runbook': runbook
            }
            
            incidents.append(incident)
        
        return {
            'status': 'INCIDENT',
            'incident_count': len(incidents),
            'timestamp': datetime.now().isoformat(),
            'incidents': incidents
        }


def get_adj_matrix_custom(cg):
    """Extract adjacency matrix from causal-learn GES result"""
    if isinstance(cg, dict):
        if 'G' in cg:
            G_part = cg['G']
            if hasattr(G_part, 'graph') and isinstance(G_part.graph, np.ndarray):
                return G_part.graph
            raise RuntimeError("GES result 'G' exists but no valid adjacency matrix found.")
        else:
            raise RuntimeError("GES result dict does not contain key 'G'.")
    else:
        if hasattr(cg, 'graph'):
            return cg.graph
        elif hasattr(cg, 'G') and hasattr(cg.G, 'graph'):
            return cg.G.graph
        raise RuntimeError("Unrecognized GES result structure.")


def enforce_dag(G: nx.DiGraph) -> nx.DiGraph:
    """Ensure the graph is a Directed Acyclic Graph (DAG)"""
    if nx.is_directed_acyclic_graph(G):
        return G
    
    print("\n[DAG Enforcement] Cycles detected. Removing edges to create DAG...")
    
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation='original')
            G.remove_edge(cycle[0][0], cycle[0][1])
            print(f"  Removed edge: {cycle[0][0]} -> {cycle[0][1]}")
        except nx.NetworkXNoCycle:
            break
    
    print("[DAG Enforcement] DAG structure ensured.\n")
    return G


def estimate_edge_confidence(data: pd.DataFrame, n_bootstrap: int = 20) -> dict:
    """Estimate confidence scores for causal edges using bootstrap"""
    print(f"\n[Confidence Estimation] Running {n_bootstrap} bootstrap iterations...")
    edge_counts = defaultdict(int)
    
    for i in range(n_bootstrap):
        if i % 5 == 0:
            print(f"  Progress: {i}/{n_bootstrap}")
        
        sample = data.sample(frac=1.0, replace=True)
        
        try:
            cg = ges(sample.values)
            adj_matrix = get_adj_matrix_custom(cg)
            
            G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
            G = nx.relabel_nodes(G, dict(enumerate(data.columns)))
            
            for edge in G.edges():
                edge_counts[edge] += 1
        except:
            continue
    
    edge_confidence = {edge: count / n_bootstrap 
                      for edge, count in edge_counts.items()}
    
    print(f"  ✓ Completed confidence estimation")
    return edge_confidence


def discover_sock_shop_causal_dag(data: pd.DataFrame) -> tuple:
    """Discover Causal DAG for Sock Shop microservices"""
    print("\n" + "="*80)
    print("                  CAUSAL DISCOVERY - SOCK SHOP")
    print("="*80)
    print(f"\n[GES Algorithm] Analyzing {len(data)} time windows across {len(data.columns)} metrics...")
    
    # Run GES
    cg = ges(data.values)
    adj_matrix = get_adj_matrix_custom(cg)
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, dict(enumerate(data.columns)))
    
    # Enforce DAG structure
    G = enforce_dag(G)
    
    # Estimate confidence scores
    confidence_scores = {}
    if len(G.edges()) > 0:
        confidence_scores = estimate_edge_confidence(data, n_bootstrap=20)
    
    # Print causal relationships
    if G.edges:
        print("\n" + "="*80)
        print("                  DISCOVERED CAUSAL RELATIONSHIPS")
        print("="*80)
        print(f"{'CAUSE METRIC':<35} -> {'EFFECT METRIC':<35} {'CONF.':<8}")
        print("-"*80)
        
        sorted_edges = sorted(G.edges(), key=lambda e: confidence_scores.get(e, 0), reverse=True)
        for u, v in sorted_edges:
            confidence = confidence_scores.get((u, v), 0.0)
            confidence_str = f"{confidence:.0%}" if confidence > 0 else "N/A"
            print(f"{u:<35} -> {v:<35} {confidence_str:<8}")
        print("="*80)
    else:
        print("⚠ No causal edges detected.")
    
    return G, confidence_scores


def visualize_sock_shop_dag(causal_dag: nx.DiGraph, confidence_scores: dict,
                            save_path: str = 'sock_shop_causal_dag.png'):
    """Visualize Sock Shop Causal DAG with service grouping"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    if len(causal_dag.nodes()) == 0:
        print("No nodes to visualize.")
        return
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Use hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(causal_dag, prog='dot', args='-Grankdir=LR')
    except:
        pos = nx.spring_layout(causal_dag, k=4, iterations=50, seed=42)
    
    # Color nodes by service
    service_colors = {
        'frontend': '#E74C3C',
        'catalogue': '#3498DB',
        'cart': '#F39C12',
        'orders': '#9B59B6',
        'payment': '#1ABC9C',
        'shipping': '#34495E',
        'user': '#E67E22',
        'catalogue_db': '#2C3E50'
    }
    
    node_colors = []
    for node in causal_dag.nodes():
        color = '#95A5A6'  # Default gray
        for service, svc_color in service_colors.items():
            if node.startswith(service):
                color = svc_color
                break
        node_colors.append(color)
    
    # Draw nodes
    nx.draw_networkx_nodes(causal_dag, pos,
                          node_color=node_colors,
                          node_size=4000,
                          alpha=0.9,
                          ax=ax)
    
    # Draw edges with confidence-based thickness
    for edge in causal_dag.edges():
        confidence = confidence_scores.get(edge, 0.5)
        width = 1 + (confidence * 5)
        alpha = 0.3 + (confidence * 0.6)
        
        nx.draw_networkx_edges(causal_dag, pos,
                              edgelist=[edge],
                              width=width,
                              alpha=alpha,
                              edge_color='#2C3E50',
                              arrows=True,
                              arrowsize=25,
                              arrowstyle='->',
                              connectionstyle='arc3,rad=0.1',
                              ax=ax)
    
    # Draw labels
    labels = {node: '\n'.join(node.split('_')) for node in causal_dag.nodes()}
    nx.draw_networkx_labels(causal_dag, pos, labels,
                           font_size=8,
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # Add confidence scores
    if confidence_scores:
        edge_labels = {edge: f"{conf:.0%}"
                      for edge, conf in confidence_scores.items()
                      if conf > 0}
        nx.draw_networkx_edge_labels(causal_dag, pos, edge_labels,
                                     font_size=7,
                                     font_color='#7F8C8D',
                                     ax=ax)
    
    # Create legend
    legend_elements = [mpatches.Patch(color=color, label=service.upper())
                      for service, color in service_colors.items()]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10, ncol=2)
    
    ax.set_title('AION Sock Shop Causal Discovery\nMicroservices Failure Propagation Analysis',
                fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n[Visualization] ✓ Saved to: {save_path}")
    
    plt.show()


def run_sock_shop_hackathon_demo():
    """
    Main demo function for hackathon presentation
    """
    print("\n" + "="*80)
    print("  █████╗ ██╗ ██████╗ ███╗   ██╗ ")
    print(" ██╔══██╗██║██╔═══██╗████╗  ██║ ")
    print(" ███████║██║██║   ██║██╔██╗ ██║ ")
    print(" ██╔══██║██║██║   ██║██║╚██╗██║ ")
    print(" ██║  ██║██║╚██████╔╝██║ ╚████║ ")
    print(" ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ")
    print("           Intelligent Root Cause Analysis for Microservices")
    print("="*80)
    
    # Generate realistic Sock Shop data with failure scenarios
    sock_shop_data = SockShopDataGenerator.generate_hackathon_demo_data(time_windows=200)
    
    print(f"\n[Dataset] Shape: {sock_shop_data.shape}")
    print(f"[Dataset] Services monitored: Frontend, Catalogue, Cart, Orders, Payment, Shipping, User")
    print(f"[Dataset] Total metrics: {len(sock_shop_data.columns)}")
    
    # Discover causal relationships
    causal_dag, confidence_scores = discover_sock_shop_causal_dag(sock_shop_data)
    
    # Analyze causal structure
    print("\n" + "="*80)
    print("                     ROOT CAUSE ANALYSIS")
    print("="*80)
    
    root_causes = [node for node in causal_dag.nodes() if causal_dag.in_degree(node) == 0]
    final_effects = [node for node in causal_dag.nodes() if causal_dag.out_degree(node) == 0]
    
    print(f"\n[Root Causes Identified] {len(root_causes)} metrics:")
    for rc in root_causes:
        descendants = list(nx.descendants(causal_dag, rc))
        print(f"  • {rc}")
        print(f"    └─ Affects {len(descendants)} downstream metrics")
    
    print(f"\n[Final Effects] {len(final_effects)} metrics:")
    for fe in final_effects:
        print(f"  • {fe}")
    
    # Generate incident alerts
    alert_gen = SockShopAlertGenerator(causal_dag)
    
    # Simulate detected anomalies
    anomalous_metrics = [
        'frontend_error_rate',
        'catalogue_latency',
        'payment_failures',
        'cart_memory'
    ]
    
    print("\n" + "="*80)
    print("                   INCIDENT DETECTION & ALERTS")
    print("="*80)
    print(f"\n[Anomalies Detected] {len(anomalous_metrics)} metrics exceeding thresholds:")
    for metric in anomalous_metrics:
        print(f"  ⚠ {metric}")
    
    # Trace to root causes
    traced_root_causes = alert_gen.trace_to_root_cause(anomalous_metrics)
    incident_report = alert_gen.generate_incident_report(traced_root_causes, confidence_scores)
    
    print("\n[Incident Report]")
    print(json.dumps(incident_report, indent=2))
    
    # Visualize the causal DAG
    print("\n" + "="*80)
    print("                    VISUALIZATION")
    print("="*80)
    visualize_sock_shop_dag(causal_dag, confidence_scores)
    
    # Export for integrations
    print("\n" + "="*80)
    print("                  INTEGRATION EXPORTS")
    print("="*80)
    
    # Neo4j Cypher queries
    print("\n[Neo4j Export] Sample Cypher queries:")
    print("-"*80)
    for u, v in list(causal_dag.edges())[:3]:
        conf = confidence_scores.get((u, v), 0.85)
        print(f"CREATE ({u.replace('-', '_')}:Metric {{name: '{u}'}})")
        print(f"       -[:CAUSES {{confidence: {conf:.2f}}}]->")
        print(f"       ({v.replace('-', '_')}:Metric {{name: '{v}'}})")
        print()
    
    print("\n" + "="*80)
    print("         ✓ AION SOCK SHOP DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\n[Summary]")
    print(f"  • Analyzed {len(sock_shop_data)} time windows across {len(sock_shop_data.columns)} metrics")
    print(f"  • Discovered {len(causal_dag.edges())} causal relationships")
    print(f"  • Identified {len(root_causes)} root cause metrics")
    print(f"  • Generated {incident_report['incident_count']} actionable incident reports")
    print(f"  • Average edge confidence: {np.mean(list(confidence_scores.values())):.1%}" if confidence_scores else "")
    
    return causal_dag, sock_shop_data, confidence_scores, incident_report


if __name__ == "__main__":
    causal_dag, data, confidence, incidents = run_sock_shop_hackathon_demo()
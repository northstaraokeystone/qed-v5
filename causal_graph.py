"""
causal_graph.py - v9 Bidirectional Flow Network for Root Cause Analysis

Paradigm Shift: Flow Network, Not Acyclic Graph
================================================

v9 replaces the traditional directed acyclic graph assumption with a bidirectional
flow network. Causality is NOT inherently directional - the direction depends on
the query:
- "Why did this happen?" -> trace_backward() to find causes
- "What would change if we changed this?" -> trace_forward() to find effects

Key Innovations:
1. Bidirectional traversal - same graph, direction determined by query
2. Value-aware compaction - high-centrality chains retain longer (365d vs 7d)
3. Self-compression ratio - system health = how well QED understands itself
4. Entanglement coefficient - cross-company pattern coherence (SLO >= 0.92)

What does NOT exist in this file:
- No "DAG" references - causality is bidirectional
- No fixed retention periods - compaction is centrality-aware
- No stored value fields - centrality computed on query
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import networkx as nx


# =============================================================================
# Centrality-Based Retention Tiers (replacing fixed time constants)
# =============================================================================

# Retention tiers based on centrality (days)
RETENTION_TIER_LOW = 7       # centrality < 0.2: 7 days
RETENTION_TIER_MID_LOW = 30  # centrality 0.2-0.5: 30 days
RETENTION_TIER_MID_HIGH = 90 # centrality 0.5-0.8: 90 days
RETENTION_TIER_HIGH = 365    # centrality >= 0.8: 365 days minimum

# Centrality thresholds (matching binder.py)
CENTRALITY_HIGH = 0.8
CENTRALITY_MID = 0.5
CENTRALITY_LOW = 0.2

# Entanglement SLO target
ENTANGLEMENT_SLO = 0.92


# =============================================================================
# Receipt Schema (self-describing module contract)
# =============================================================================

RECEIPT_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "trace_receipt",
        "version": "1.0.0",
        "description": "Receipt emitted by trace() function - R -> R transformer",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of input receipt IDs",
            "timestamp": "ISO UTC timestamp",
            "input_receipt_ids": "List of receipt IDs processed",
            "graph_metrics": {
                "node_count": "int - total nodes in graph",
                "edge_count": "int - total edges in graph",
                "avg_degree": "float - average node degree",
                "clustering_coefficient": "float - graph clustering",
            },
            "trace_direction": "str - 'forward', 'backward', or 'build'",
            "nodes_traversed": "int - number of nodes visited",
            "reversible": "bool - always True for trace operations",
            "reverse_action": "str - how to reverse this operation",
        },
    },
    {
        "type": "compaction_receipt",
        "version": "1.0.0",
        "description": "Receipt for value-aware graph compaction per SDD 5.2",
        "fields": {
            "receipt_id": "SHA3-256 hash (16 chars) of compaction operation",
            "timestamp": "ISO UTC timestamp",
            "input_span": {
                "start_ts": "ISO timestamp of oldest node",
                "end_ts": "ISO timestamp of newest node",
                "node_count": "int - nodes before compaction",
            },
            "output_span": {
                "start_ts": "ISO timestamp of oldest retained node",
                "end_ts": "ISO timestamp of newest retained node",
                "node_count": "int - nodes after compaction",
            },
            "counts": {
                "collapsed_low": "int - nodes with centrality < 0.2 collapsed",
                "collapsed_mid_low": "int - nodes with centrality 0.2-0.5 collapsed",
                "collapsed_mid_high": "int - nodes with centrality 0.5-0.8 collapsed",
                "retained_high": "int - high-value nodes retained",
                "summary_nodes_created": "int - summary nodes replacing collapsed",
            },
            "sums": {
                "edges_removed": "int - total edges collapsed",
                "centrality_preserved": "float - sum of retained centralities",
            },
            "hash_continuity": "str - SHA3 linking pre and post compaction states",
        },
    },
    {
        "type": "entanglement_receipt",
        "version": "1.0.0",
        "description": "Receipt for cross-company entanglement coefficient query",
        "fields": {
            "receipt_id": "SHA3-256 hash of query parameters",
            "timestamp": "ISO UTC timestamp",
            "pattern_id": "str - pattern queried",
            "companies": "List[str] - companies in entanglement query",
            "coefficient": "float - entanglement coefficient [0, 1]",
            "meets_slo": "bool - coefficient >= 0.92",
            "per_company_centralities": "Dict[str, float] - centrality per company",
        },
    },
]


# =============================================================================
# Dataclasses for Receipt Types
# =============================================================================

@dataclass
class TraceReceipt:
    """Receipt emitted by trace operations."""
    receipt_id: str
    timestamp: str
    input_receipt_ids: List[str]
    graph_metrics: Dict[str, Any]
    trace_direction: str
    nodes_traversed: int
    reversible: bool = True
    reverse_action: str = "Re-trace with inverse direction"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "trace_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "input_receipt_ids": self.input_receipt_ids,
            "graph_metrics": self.graph_metrics,
            "trace_direction": self.trace_direction,
            "nodes_traversed": self.nodes_traversed,
            "reversible": self.reversible,
            "reverse_action": self.reverse_action,
        }


@dataclass
class CompactionReceipt:
    """Receipt emitted by graph compaction per SDD 5.2."""
    receipt_id: str
    timestamp: str
    input_span: Dict[str, Any]
    output_span: Dict[str, Any]
    counts: Dict[str, int]
    sums: Dict[str, Any]
    hash_continuity: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "compaction_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "input_span": self.input_span,
            "output_span": self.output_span,
            "counts": self.counts,
            "sums": self.sums,
            "hash_continuity": self.hash_continuity,
        }


@dataclass
class EntanglementReceipt:
    """Receipt for entanglement coefficient queries."""
    receipt_id: str
    timestamp: str
    pattern_id: str
    companies: List[str]
    coefficient: float
    meets_slo: bool
    per_company_centralities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": "entanglement_receipt",
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "pattern_id": self.pattern_id,
            "companies": self.companies,
            "coefficient": self.coefficient,
            "meets_slo": self.meets_slo,
            "per_company_centralities": self.per_company_centralities,
        }


# =============================================================================
# Graph Construction
# =============================================================================

def build_graph(receipts: List[Dict[str, Any]]) -> nx.DiGraph:
    """
    Construct bidirectional flow network from receipt list.

    Edges derived from:
    - sampled_receipts: receipt references other receipts it sampled
    - pattern_usage: pattern co-occurrence creates bidirectional edges
    - parent_receipt_id: explicit causal link to parent

    Args:
        receipts: List of receipt dicts with causal references

    Returns:
        NetworkX DiGraph (bidirectional edges for flow network)
    """
    graph = nx.DiGraph()

    for receipt in receipts:
        # Extract node ID - try multiple fields
        node_id = (
            receipt.get("receipt_id") or
            receipt.get("window_id") or
            receipt.get("packet_id")
        )
        if not node_id:
            continue

        # Add node with timestamp for age-based compaction
        timestamp = receipt.get("timestamp") or receipt.get("ts")
        graph.add_node(
            node_id,
            timestamp=timestamp,
            receipt_type=receipt.get("type", "unknown"),
            company=receipt.get("company") or receipt.get("tenant_id"),
        )

        # Edge from sampled_receipts (causal input)
        sampled = receipt.get("sampled_receipts", [])
        for sampled_id in sampled:
            if isinstance(sampled_id, dict):
                sampled_id = sampled_id.get("receipt_id")
            if sampled_id:
                # Bidirectional: sampled influences this, this depends on sampled
                if not graph.has_node(sampled_id):
                    graph.add_node(sampled_id)
                graph.add_edge(sampled_id, node_id, relation="samples")
                graph.add_edge(node_id, sampled_id, relation="sampled_by")

        # Edge from parent_receipt_id (explicit causality)
        parent_id = receipt.get("parent_receipt_id")
        if parent_id:
            if not graph.has_node(parent_id):
                graph.add_node(parent_id)
            graph.add_edge(parent_id, node_id, relation="parent")
            graph.add_edge(node_id, parent_id, relation="child_of")

        # Edges from pattern_usage (co-occurrence)
        patterns = receipt.get("pattern_usage", [])
        pattern_ids = []
        for p in patterns:
            if isinstance(p, dict):
                pid = p.get("pattern_id")
            else:
                pid = getattr(p, "pattern_id", None) if hasattr(p, "pattern_id") else p
            if pid:
                pattern_ids.append(pid)

        # Create bidirectional edges between co-occurring patterns
        for i, pid_a in enumerate(pattern_ids):
            if not graph.has_node(pid_a):
                graph.add_node(pid_a, is_pattern=True)

            # Link receipt to pattern
            graph.add_edge(node_id, pid_a, relation="uses_pattern")
            graph.add_edge(pid_a, node_id, relation="used_by")

            # Link co-occurring patterns
            for pid_b in pattern_ids[i + 1:]:
                if not graph.has_node(pid_b):
                    graph.add_node(pid_b, is_pattern=True)
                if not graph.has_edge(pid_a, pid_b):
                    graph.add_edge(pid_a, pid_b, relation="co_occurs", weight=1)
                    graph.add_edge(pid_b, pid_a, relation="co_occurs", weight=1)
                else:
                    graph[pid_a][pid_b]["weight"] = graph[pid_a][pid_b].get("weight", 0) + 1
                    graph[pid_b][pid_a]["weight"] = graph[pid_b][pid_a].get("weight", 0) + 1

    return graph


# =============================================================================
# Centrality Computation (matches binder.py contract)
# =============================================================================

def centrality(node: str, graph: nx.DiGraph) -> float:
    """
    Compute node centrality via PageRank (matches binder.py contract).

    Value is NEVER stored - always derived from current graph topology.

    Args:
        node: Node ID to compute centrality for
        graph: NetworkX DiGraph

    Returns:
        Centrality value normalized to [0, 1]
        Returns 0.0 if node not in graph
    """
    if graph.number_of_nodes() == 0:
        return 0.0

    if node not in graph:
        return 0.0

    # Compute PageRank (matching binder.py parameters)
    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        # Fallback to degree centrality
        pagerank = nx.degree_centrality(graph)

    raw_centrality = pagerank.get(node, 0.0)

    # Normalize to [0, 1] based on max centrality
    max_centrality = max(pagerank.values()) if pagerank else 1.0
    if max_centrality > 0:
        normalized = raw_centrality / max_centrality
    else:
        normalized = 0.0

    return min(1.0, max(0.0, normalized))


def _compute_all_centralities(graph: nx.DiGraph) -> Dict[str, float]:
    """
    Compute normalized PageRank centralities for all nodes.

    Args:
        graph: NetworkX DiGraph

    Returns:
        Dict mapping node ID to normalized centrality [0, 1]
    """
    if graph.number_of_nodes() == 0:
        return {}

    try:
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100)
    except nx.PowerIterationFailedConvergence:
        pagerank = nx.degree_centrality(graph)

    max_val = max(pagerank.values()) if pagerank else 1.0
    if max_val > 0:
        return {k: min(1.0, v / max_val) for k, v in pagerank.items()}
    return {k: 0.0 for k in pagerank}


# =============================================================================
# Bidirectional Traversal (v9 paradigm: direction depends on query)
# =============================================================================

def trace_forward(node: str, graph: nx.DiGraph, max_depth: int = 100) -> List[str]:
    """
    Trace effects downstream from node (successors).

    Answers: "What would change if we changed this?"

    Uses lazy evaluation for large graphs - generator internally,
    materialized to list only up to max_depth nodes.

    Args:
        node: Starting node ID
        graph: NetworkX DiGraph
        max_depth: Maximum nodes to return (prevents unbounded traversal)

    Returns:
        List of downstream node IDs affected by this node
    """
    if node not in graph:
        return []

    visited: Set[str] = set()
    result: List[str] = []

    def _traverse(current: str, depth: int) -> Generator[str, None, None]:
        if depth > max_depth or current in visited:
            return
        visited.add(current)
        yield current
        for successor in graph.successors(current):
            yield from _traverse(successor, depth + 1)

    # Skip the starting node itself, return only affected nodes
    for n in _traverse(node, 0):
        if n != node and len(result) < max_depth:
            result.append(n)

    return result


def trace_backward(node: str, graph: nx.DiGraph, max_depth: int = 100) -> List[str]:
    """
    Trace causes upstream from node (predecessors).

    Answers: "Why did this happen?"

    Uses lazy evaluation for large graphs - generator internally,
    materialized to list only up to max_depth nodes.

    Args:
        node: Starting node ID
        graph: NetworkX DiGraph
        max_depth: Maximum nodes to return (prevents unbounded traversal)

    Returns:
        List of upstream node IDs that caused this node
    """
    if node not in graph:
        return []

    visited: Set[str] = set()
    result: List[str] = []

    def _traverse(current: str, depth: int) -> Generator[str, None, None]:
        if depth > max_depth or current in visited:
            return
        visited.add(current)
        yield current
        for predecessor in graph.predecessors(current):
            yield from _traverse(predecessor, depth + 1)

    # Skip the starting node itself, return only causal nodes
    for n in _traverse(node, 0):
        if n != node and len(result) < max_depth:
            result.append(n)

    return result


# =============================================================================
# Core trace() Function - R -> R Transformer (v9 Paradigm 1)
# =============================================================================

def trace(receipts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Core Receipt Monad transformer: trace(receipts) -> trace_receipts.

    Builds causal flow network from input receipts and emits trace_receipt.
    Signature: R -> R (List[Receipt] -> List[Receipt])

    Args:
        receipts: List of receipt dicts to build graph from

    Returns:
        List containing trace_receipt with graph metrics
    """
    if not receipts:
        return []

    # Build graph from receipts
    graph = build_graph(receipts)

    # Extract input receipt IDs
    input_ids = []
    for r in receipts:
        rid = r.get("receipt_id") or r.get("window_id") or r.get("packet_id")
        if rid:
            input_ids.append(rid)

    # Compute graph metrics for self-description
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    avg_degree = (2.0 * edge_count / node_count) if node_count > 0 else 0.0

    # Clustering coefficient (graph structure health)
    try:
        clustering = nx.average_clustering(graph.to_undirected())
    except Exception:
        clustering = 0.0

    graph_metrics = {
        "node_count": node_count,
        "edge_count": edge_count,
        "avg_degree": avg_degree,
        "clustering_coefficient": clustering,
    }

    # Generate deterministic receipt ID
    sorted_ids = sorted(input_ids)
    content = json.dumps(sorted_ids, separators=(",", ":"))
    receipt_id = hashlib.sha3_256(content.encode()).hexdigest()[:16]

    trace_receipt = TraceReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_receipt_ids=sorted_ids,
        graph_metrics=graph_metrics,
        trace_direction="build",
        nodes_traversed=node_count,
    )

    return [trace_receipt.to_dict()]


# =============================================================================
# Self-Compression Ratio (System Health Metric)
# =============================================================================

def self_compression_ratio(graph: nx.DiGraph) -> float:
    """
    Compute system health via self-compression ratio.

    Kolmogorov-inspired: ratio of graph summary size to raw edge count.
    Summary = (node_count, edge_count, avg_degree, clustering_coefficient).
    If summary predicts graph structure well, ratio is high.

    The graph IS QED's telemetry about its own behavior.
    Compacting the graph IS compressing telemetry about compression.
    This is the strange loop: QED watching itself.

    A sudden drop in this ratio = something changed that broke the pattern.
    No separate anomaly detection needed - compression ratio tells you.

    Returns:
        Float in (0, 1] where:
        - High (>0.8): Simple self-model, system understands itself, healthy
        - Low (<0.5): Complex/unpredictable structure, anomaly signal
    """
    if graph.number_of_nodes() == 0:
        return 1.0  # Empty graph is maximally simple

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()

    if edge_count == 0:
        return 1.0  # No edges = maximally simple

    # Compute graph summary metrics
    avg_degree = (2.0 * edge_count / node_count)

    try:
        clustering = nx.average_clustering(graph.to_undirected())
    except Exception:
        clustering = 0.0

    # Summary size: 4 floats = ~128 bits
    summary_bits = 128.0

    # Raw graph size estimate: each edge needs ~64 bits (2 node refs)
    raw_bits = edge_count * 64.0

    # Base compression ratio
    base_ratio = summary_bits / raw_bits if raw_bits > 0 else 1.0

    # Predictability factor: regular graphs have predictable structure
    # High clustering + regular degree distribution = predictable
    # We use clustering as proxy for predictability
    predictability = clustering

    # Degree regularity: how uniform is the degree distribution?
    degrees = [d for _, d in graph.degree()]
    if len(degrees) > 1:
        degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / len(degrees)
        max_variance = (node_count - 1) ** 2  # Maximum possible variance
        regularity = 1.0 - (degree_variance / max_variance) if max_variance > 0 else 1.0
    else:
        regularity = 1.0

    # Combined self-compression ratio
    # Higher predictability and regularity = better self-model
    compression = (base_ratio * 0.3) + (predictability * 0.35) + (regularity * 0.35)

    # Clamp to (0, 1]
    return max(0.001, min(1.0, compression))


# =============================================================================
# Entanglement Coefficient (Cross-Company Coherence)
# =============================================================================

def entanglement_coefficient(
    pattern_id: str,
    companies: List[str],
    graph: nx.DiGraph,
) -> float:
    """
    Compute cross-company entanglement coefficient.

    Measures how observing a pattern in one company affects views in others.
    High entanglement (>=0.92) = pattern behaves consistently across companies.

    This enables Paradigm 6: patterns shared across companies aren't "synced",
    they're quantum-entangled. Observing Tesla's pattern P affects SpaceX's
    view of P at query time.

    Algorithm:
    1. Query each company's subgraph for pattern's centrality
    2. Entanglement = 1 - variance(centralities) / max_possible_variance
    3. Low variance = high entanglement = consistent behavior

    Args:
        pattern_id: Pattern to query
        companies: List of company identifiers to check
        graph: Full flow network containing all companies

    Returns:
        Entanglement coefficient in [0, 1]
        SLO target is >= 0.92
    """
    if not companies or pattern_id not in graph:
        return 0.0

    # Build subgraph for each company and compute pattern centrality
    per_company_centralities: Dict[str, float] = {}

    for company in companies:
        # Filter nodes belonging to this company
        company_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("company") == company
        ]

        if not company_nodes:
            # Company has no nodes - neutral contribution
            per_company_centralities[company] = 0.5
            continue

        # Include the pattern if it exists
        if pattern_id in graph:
            company_nodes.append(pattern_id)

        # Build subgraph
        subgraph = graph.subgraph(company_nodes).copy()

        # Compute pattern centrality in this subgraph
        cent = centrality(pattern_id, subgraph)
        per_company_centralities[company] = cent

    # Compute entanglement from centrality variance
    centralities = list(per_company_centralities.values())

    if len(centralities) < 2:
        # Single company - perfect "entanglement" (no variance)
        return 1.0

    mean_cent = sum(centralities) / len(centralities)
    variance = sum((c - mean_cent) ** 2 for c in centralities) / len(centralities)

    # Maximum possible variance for values in [0, 1] is 0.25
    # (when half are 0 and half are 1)
    max_variance = 0.25

    # Entanglement = 1 - normalized variance
    entanglement = 1.0 - (variance / max_variance) if max_variance > 0 else 1.0

    return max(0.0, min(1.0, entanglement))


def query_entanglement(
    pattern_id: str,
    companies: List[str],
    graph: nx.DiGraph,
) -> EntanglementReceipt:
    """
    Query entanglement and return full receipt.

    Args:
        pattern_id: Pattern to query
        companies: Companies to check entanglement across
        graph: Flow network

    Returns:
        EntanglementReceipt with coefficient and SLO status
    """
    # Compute per-company centralities
    per_company: Dict[str, float] = {}

    for company in companies:
        company_nodes = [
            n for n in graph.nodes()
            if graph.nodes[n].get("company") == company
        ]
        if pattern_id in graph:
            company_nodes.append(pattern_id)

        if company_nodes:
            subgraph = graph.subgraph(company_nodes).copy()
            per_company[company] = centrality(pattern_id, subgraph)
        else:
            per_company[company] = 0.5

    coeff = entanglement_coefficient(pattern_id, companies, graph)

    # Generate receipt ID
    content = json.dumps({
        "pattern_id": pattern_id,
        "companies": sorted(companies),
    }, separators=(",", ":"))
    receipt_id = hashlib.sha3_256(content.encode()).hexdigest()[:16]

    return EntanglementReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        pattern_id=pattern_id,
        companies=sorted(companies),
        coefficient=coeff,
        meets_slo=coeff >= ENTANGLEMENT_SLO,
        per_company_centralities=per_company,
    )


# =============================================================================
# Value-Aware Compaction (Replacing Fixed Time Retention)
# =============================================================================

def _get_retention_days(cent: float) -> int:
    """
    Get retention period based on centrality tier.

    Args:
        cent: Node centrality value

    Returns:
        Retention period in days
    """
    if cent >= CENTRALITY_HIGH:
        return RETENTION_TIER_HIGH  # 365 days
    elif cent >= CENTRALITY_MID:
        return RETENTION_TIER_MID_HIGH  # 90 days
    elif cent >= CENTRALITY_LOW:
        return RETENTION_TIER_MID_LOW  # 30 days
    else:
        return RETENTION_TIER_LOW  # 7 days


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        # Handle various ISO formats
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def compact(
    graph: nx.DiGraph,
    centrality_threshold: float = 0.0,
    reference_time: Optional[datetime] = None,
) -> Tuple[nx.DiGraph, CompactionReceipt]:
    """
    Value-aware graph compaction.

    Compaction tiers (replaces fixed retention):
    - centrality < 0.2 AND age > 7d -> collapse
    - centrality 0.2-0.5 AND age > 30d -> collapse
    - centrality 0.5-0.8 AND age > 90d -> collapse
    - centrality >= 0.8 -> retain 365d minimum

    Collapsed nodes are replaced with summary nodes preserving
    aggregate statistics.

    Args:
        graph: Flow network to compact
        centrality_threshold: Additional minimum centrality to retain
        reference_time: Time to compute age from (default: now)

    Returns:
        Tuple of (compacted_graph, compaction_receipt)
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)

    # Compute all centralities once
    centralities = _compute_all_centralities(graph)

    # Track compaction statistics
    input_node_count = graph.number_of_nodes()
    input_edge_count = graph.number_of_edges()

    # Find timestamp bounds for input span
    timestamps: List[datetime] = []
    for node in graph.nodes():
        ts = graph.nodes[node].get("timestamp")
        parsed = _parse_timestamp(ts)
        if parsed:
            timestamps.append(parsed)

    input_start = min(timestamps).isoformat() if timestamps else reference_time.isoformat()
    input_end = max(timestamps).isoformat() if timestamps else reference_time.isoformat()

    # Determine which nodes to collapse
    nodes_to_collapse: Set[str] = set()
    collapse_counts = {
        "low": 0,
        "mid_low": 0,
        "mid_high": 0,
    }
    retained_high = 0

    for node in graph.nodes():
        cent = centralities.get(node, 0.0)

        # Always respect explicit threshold
        if cent >= centrality_threshold:
            if cent >= CENTRALITY_HIGH:
                retained_high += 1
            continue

        # Check age-based retention
        ts_str = graph.nodes[node].get("timestamp")
        node_time = _parse_timestamp(ts_str)

        if node_time is None:
            # No timestamp - keep if centrality threshold not zero
            continue

        age_days = (reference_time - node_time).days
        retention_days = _get_retention_days(cent)

        if age_days > retention_days:
            nodes_to_collapse.add(node)
            if cent < CENTRALITY_LOW:
                collapse_counts["low"] += 1
            elif cent < CENTRALITY_MID:
                collapse_counts["mid_low"] += 1
            else:
                collapse_counts["mid_high"] += 1

    # Create compacted graph
    compacted = graph.copy()

    # Remove collapsed nodes
    edges_removed = 0
    centrality_preserved = 0.0

    for node in nodes_to_collapse:
        edges_removed += compacted.degree(node)
        compacted.remove_node(node)

    # Compute preserved centrality
    for node in compacted.nodes():
        centrality_preserved += centralities.get(node, 0.0)

    # Create summary nodes for collapsed regions if significant
    summary_nodes_created = 0
    if nodes_to_collapse:
        summary_id = f"summary_{hashlib.sha3_256(str(sorted(nodes_to_collapse)).encode()).hexdigest()[:8]}"
        compacted.add_node(
            summary_id,
            is_summary=True,
            collapsed_count=len(nodes_to_collapse),
            timestamp=reference_time.isoformat(),
        )
        summary_nodes_created = 1

    # Compute output span
    output_timestamps: List[datetime] = []
    for node in compacted.nodes():
        ts = compacted.nodes[node].get("timestamp")
        parsed = _parse_timestamp(ts)
        if parsed:
            output_timestamps.append(parsed)

    output_start = min(output_timestamps).isoformat() if output_timestamps else reference_time.isoformat()
    output_end = max(output_timestamps).isoformat() if output_timestamps else reference_time.isoformat()

    # Generate hash continuity (links pre and post states)
    pre_hash = hashlib.sha3_256(str(sorted(graph.nodes())).encode()).hexdigest()[:16]
    post_hash = hashlib.sha3_256(str(sorted(compacted.nodes())).encode()).hexdigest()[:16]
    hash_continuity = f"{pre_hash}->{post_hash}"

    # Generate receipt ID
    receipt_content = json.dumps({
        "input_nodes": input_node_count,
        "output_nodes": compacted.number_of_nodes(),
        "reference_time": reference_time.isoformat(),
    }, separators=(",", ":"))
    receipt_id = hashlib.sha3_256(receipt_content.encode()).hexdigest()[:16]

    receipt = CompactionReceipt(
        receipt_id=receipt_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_span={
            "start_ts": input_start,
            "end_ts": input_end,
            "node_count": input_node_count,
        },
        output_span={
            "start_ts": output_start,
            "end_ts": output_end,
            "node_count": compacted.number_of_nodes(),
        },
        counts={
            "collapsed_low": collapse_counts["low"],
            "collapsed_mid_low": collapse_counts["mid_low"],
            "collapsed_mid_high": collapse_counts["mid_high"],
            "retained_high": retained_high,
            "summary_nodes_created": summary_nodes_created,
        },
        sums={
            "edges_removed": edges_removed,
            "centrality_preserved": centrality_preserved,
        },
        hash_continuity=hash_continuity,
    )

    return compacted, receipt


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Schema
    "RECEIPT_SCHEMA",
    # Constants
    "CENTRALITY_HIGH",
    "CENTRALITY_MID",
    "CENTRALITY_LOW",
    "ENTANGLEMENT_SLO",
    "RETENTION_TIER_LOW",
    "RETENTION_TIER_MID_LOW",
    "RETENTION_TIER_MID_HIGH",
    "RETENTION_TIER_HIGH",
    # Dataclasses
    "TraceReceipt",
    "CompactionReceipt",
    "EntanglementReceipt",
    # Graph operations
    "build_graph",
    "centrality",
    # Traversal
    "trace",
    "trace_forward",
    "trace_backward",
    # Health metrics
    "self_compression_ratio",
    # Entanglement
    "entanglement_coefficient",
    "query_entanglement",
    # Compaction
    "compact",
]

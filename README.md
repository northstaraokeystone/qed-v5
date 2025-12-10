# QED: Receipts-Native Telemetry Compression

QED is a receipts-native telemetry compression system targeting 10-50x compression at ≥99.9% recall across safety-critical domains: Tesla, SpaceX, Starlink, Boring Company, Neuralink, and xAI.

## What is QED?

Every action emits a receipt. Receipts are the universal substrate—not logs, not metrics, receipts. The receipt is the territory.

QED starts from a simple observation: safety-critical systems emit far more telemetry than they need to keep. A Tesla vehicle generates petabytes annually, yet the actual safety-relevant data—anomalies, state transitions, thresholds—fits in terabytes at 10x compression with perfect recall. The same pattern holds for SpaceX launches, Starlink links, Boring tunnels, Neuralink implants, and xAI inference clusters.

The core principle: build a system where every decision—compression, anomaly detection, remediation, agent creation—leaves a receipt. These receipts become the audit trail, the training data, the fitness function, and the ground truth for the next generation of agents. Agents themselves are autocatalytic receipt patterns. The system bootstraps.

v5 delivers 10x compression, ~$900M/year value at Tesla scale. v6+ layers receipts, edge validation, and mesh visibility. v7 adds physics validation and cross-company anomaly sharing. v8 wraps deployment decisions in signed packets. v9 shifts from stored state to receipt-graph topology. v10 introduces autonomous healing (HUNTER/SHEPHERD agents). v11 adds immune system protection. v12 enables agent genesis—agents proposing and breeding new agents under human-in-the-loop gates.

## Value Proposition

QED baseline (v5) at Tesla scale: 10x compression on multi-billion vehicle-years → ~$900M/year savings in storage, bandwidth, and incident response. This compounds: SpaceX saves $331.7M/5yr (284% NPV), xAI at Grok scale saves $195.6B/5yr. The pattern is consistent: telemetry volume far exceeds safety relevance; compress 99.9% recall; capture both the savings and the audit trail.

Higher versions (v6-v12) add auditability (receipts, manifests), cross-company learning (anomaly library), autonomous agents (HUNTER/SHEPHERD), agent breeding (ARCHITECT), and immune system protection. Each layer is optional; v5 works standalone. Layers compound linearly on v5's baseline.

## Version Overview

| Version | Milestone | Key Capability |
|---------|-----------|---|
| **v5** | Baseline | Single engine (qed.py) + 5 hooks; 10x compression; $900M/yr at Tesla scale |
| **v6** | Receipts Native | QEDReceipt JSONL; edge_lab v1 validation; mesh_view v1 dashboard |
| **v7** | Physics Valid | ClarityClean; edge_lab v2 with NGSIM/SAE/NHTSA anomalies; shared_anomalies library |
| **v8** | TruthLink | DecisionPackets; qed_config.json; config merge rules (safety tightens only) |
| **v9** | Portfolio & Causality | PortfolioBinder; CausalGraph; EventStream; ROI gate; bidirectional causality |
| **v10** | Autonomous Agents | HUNTER (entropy anomaly detection); SHEPHERD (automated healing); unified_loop 8-phase cycle |
| **v11** | Immune System | Risk scoring; multi-dimensional fitness; autoimmune SELF protection; wound tracking |
| **v12** | Agent Genesis | ARCHITECT proposes new agents; blueprint validation; autocatalysis check; recombination |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         QED SYSTEM                               │
├─────────────────────────────────────────────────────────────────┤
│  TELEMETRY IN                                                    │
│       ↓                                                          │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                 │
│  │ qed.py  │───→│ receipts │───→│ mesh_view   │                 │
│  │ (core)  │    │ (JSONL)  │    │ (dashboard) │                 │
│  └─────────┘    └──────────┘    └─────────────┘                 │
│       ↓              ↓                                           │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                 │
│  │edge_lab │    │ manifest │───→│ TruthLink   │                 │
│  │(validate)│    │ (JSON)   │    │(DecisionPkt)│                 │
│  └─────────┘    └──────────┘    └─────────────┘                 │
│                      ↓                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │            unified_loop (v10+)            │                   │
│  │  SENSE → MEASURE → ANALYZE → REMEDIATE   │                   │
│  │  → HYPOTHESIZE → GATE → ACTUATE → EMIT   │                   │
│  └──────────────────────────────────────────┘                   │
│       ↓              ↓              ↓                            │
│  ┌─────────┐    ┌──────────┐    ┌─────────────┐                 │
│  │ HUNTER  │    │ SHEPHERD │    │  ARCHITECT  │                 │
│  │(detect) │    │ (heal)   │    │  (genesis)  │                 │
│  └─────────┘    └──────────┘    └─────────────┘                 │
│                      ↓                                           │
│  ┌──────────────────────────────────────────┐                   │
│  │         PortfolioBinder (v9+)            │                   │
│  │    CausalGraph │ EventStream │ ROI Gate  │                   │
│  └──────────────────────────────────────────┘                   │
│                      ↓                                           │
│  DECISIONS OUT (compressed telemetry + audit trail)             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Concepts

**Receipts.** Every action—compression decision, anomaly detection, healing action, agent birth—emits a receipt. Receipts are JSONL, append-only, Merkle-tree anchored per second. They form the audit trail, the training signal, and the ground truth. No silent failures; every receipt has a timestamp, tenant ID, and payload hash (SHA256:BLAKE3 dual-hash per CLAUDEME.md).

**Entropy.** Agent fitness is measured by entropy reduction. Shannon entropy H = -Σ p(x) log₂ p(x) quantifies system disorder. When an agent reduces system entropy (detects drift, fixes constraint violations), it gains fitness. `cycle_entropy_delta` tracks health: negative for 3+ cycles → degradation alert. Selection pressure uses Thompson sampling: agents with higher fitness-per-receipt have higher probability of reproduction and of being deployed to new domains.

**HITL (Human-in-the-Loop).** When confidence < 0.8 or risk ≥ 0.3, decisions escalate to humans. The system "becomes human" when uncertain. HITL decisions seed the wound tracking system (v11), which identifies recurring manual interventions and proposes automating them (ARCHITECT in v12).

**Agents as Receipt Patterns.** Agents are not code objects. They are autocatalytic receipt patterns—emergent from the receipt stream itself. HUNTER detects anomalies by watching entropy spikes. SHEPHERD heals via homeostasis actions. ARCHITECT synthesizes new agents from wound patterns. When an agent stops emitting receipts or loses coherence, it enters SUPERPOSITION (not deleted), available for future remixing.

**SELF Protection.** The immune system (v11) maintains a GERMLINE_PATTERNS set: {qed_core, hunter, shepherd, architect}. These cannot be killed by selection. `is_self(pattern)` checks origin; SELF patterns are immortal. Prevents the system from attacking its own core.

**Autoimmune Healing.** SHEPHERD runs a 6-step recovery taxonomy: rollback (undo change), reroute (send traffic elsewhere), isolate (quarantine faulty component), restart (reboot), failover (to backup), graceful_degradation (reduced capacity). Each recovery emits a receipt; success feeds back into fitness scoring.

**Fitness as Topology.** v9 deleted stored state. Value, mode, and behavior are computed at query time from receipt-graph topology using PageRank-style centrality. One new receipt can change the fitness of all agents. Entanglement replaces aggregation: observing Tesla's pattern affects SpaceX's view of it—instantaneously.

## Quick Start

### Install and Run Smoke Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke tests (fast validation)
bash tests/smoke/test_smoke_v10.sh

# Run full test suite
pytest tests/ -v
```

### Process Telemetry (v5 Basic)

```bash
# Single hook (Tesla example)
python qed.py --hook tesla --input telemetry.jsonl --output compressed.jsonl

# All hooks
for hook in tesla spacex boring neuralink xai; do
  python qed.py --hook $hook --input telemetry_$hook.jsonl
done
```

### View Dashboard (v6+)

```bash
# Mesh view aggregates receipts into per-company tables
python mesh_view.py --receipts data/receipts/ --output mesh_dashboard.json
```

### Deploy with Configuration (v8+)

```bash
# Validate config; merge rules ensure only tighter safety
python qed.py --config configs/qed_config.json --validate

# Run with merged config (global + deployment overrides)
python qed.py --config configs/qed_config.json --hook tesla --input telemetry.jsonl
```

## Configuration

qed_config.json example:
```json
{
  "hook": "tesla",
  "recall_floor": 0.9995,
  "max_fp_rate": 0.005,
  "compression_target": 10,
  "slo_latency_ms": 50,
  "enabled_patterns": ["drift", "degradation", "constraint_violation"]
}
```

**Merge rules (v8+):** Configs can only TIGHTEN safety and regulation, never loosen. Recall floors take the higher value; FP rate limits take the lower value; regulatory flags OR together; enabled patterns intersect.

## Directory Structure

```
qed/
├── qed.py                 # Core compression engine (v5+)
├── receipts.py            # Receipt emission (v6+)
├── manifest.py            # Run manifest generation (v6+)
├── edge_lab.py            # Edge scenario validation (v6+)
├── mesh_view.py           # Cross-company dashboard (v6+)
├── clarity_clean.py       # Receipt → text + audit (v7+)
├── truthlink.py           # DecisionPacket builder (v8+)
├── config.py              # Config validation + merge (v8+)
├── portfolio_binder.py    # Cross-company aggregation (v9+)
├── causal_graph.py        # Root cause tracing (v9+)
├── event_stream.py        # CQRS append-only log (v9+)
├── entropy.py             # Shannon entropy primitives (v10+)
├── integrity.py           # HUNTER anomaly detection (v10+)
├── remediate.py           # SHEPHERD healing (v10+)
├── unified_loop.py        # 8-phase metabolism (v10+)
├── risk.py                # Risk scoring (v11+)
├── fitness.py             # Multi-dimensional fitness (v11+)
├── autoimmune.py          # SELF/OTHER distinction (v11+)
├── wound.py               # Automation gap tracking (v11+)
├── meta.py                # Paradigm shift tracking (v11+)
├── data/
│   ├── receipts/          # Receipt JSONL storage
│   ├── manifests/         # Run manifest JSON
│   └── differentials/     # State cache (optional)
├── configs/
│   └── qed_config.json    # Runtime configuration
├── tests/
│   ├── test_qed_v6.py
│   ├── test_qed_v7.py
│   ├── test_qed_v8.py
│   ├── test_qed_v9.py
│   ├── test_qed_v10.py
│   ├── test_qed_v11.py
│   └── smoke/
│       ├── test_smoke_v6.sh
│       ├── test_smoke_v7.sh
│       ├── test_smoke_v8.py
│       ├── test_smoke_v9.sh
│       ├── test_smoke_v10.sh
│       └── results/
└── CLAUDEME.md            # Project standards
```

## Module Reference

| Module | Purpose | Version | Inputs | Outputs |
|--------|---------|---------|--------|---------|
| qed.py | Core compression engine | v5+ | Telemetry windows | Compression decisions |
| receipts.py | QEDReceipt emission | v6+ | Any action | JSONL receipt lines |
| manifest.py | Run manifest generation | v6+ | Receipt batch | qed_run_manifest.json |
| edge_lab.py | Edge scenario validation | v6+ | Receipts + scenarios | Hit/miss + latency metrics |
| mesh_view.py | Cross-company dashboard | v6+ | Receipts/manifests | Per-company tables |
| clarity_clean.py | Receipt → text audit | v7+ | Receipts | Text corpus + quality audit |
| truthlink.py | DecisionPacket builder | v8+ | Manifests + receipts | Signed deployment packets |
| config.py | Config validation + merge | v8+ | Config JSON | Merged + validated config |
| portfolio_binder.py | Cross-company aggregation | v9+ | DecisionPackets | Annual savings/company |
| causal_graph.py | Root cause tracing | v9+ | Receipt lineage | DAG + forward/backward traces |
| event_stream.py | CQRS append-only log | v9+ | Decisions | Indexed event stream |
| entropy.py | Shannon entropy primitives | v10+ | Receipts | H, agent_fitness, delta |
| integrity.py | HUNTER anomaly detection | v10+ | Event stream + graph | Anomaly_alert receipts |
| remediate.py | SHEPHERD healing | v10+ | Anomalies + graph | Recovery_action receipts |
| unified_loop.py | 8-phase metabolism | v10+ | Receipts | Loop cycle completion |
| risk.py | Risk scoring (inflammation) | v11+ | Pattern + confidence | Risk score (0.0-1.0) |
| fitness.py | Multi-dimensional fitness | v11+ | Pattern + metadata | Fitness score |
| autoimmune.py | SELF/OTHER distinction | v11+ | Pattern origin | Bool (is_self) |
| wound.py | Automation gap tracking | v11+ | Manual interventions | wound_receipt entries |
| meta.py | Paradigm shift tracking | v11+ | Receipts over time | Paradigm change signals |

## Agent Taxonomy (v10+)

| Agent | Module | Role | Detection | Autonomy | Immortal |
|-------|--------|------|-----------|----------|----------|
| HUNTER | integrity.py | Find problems | Entropy spike detection; drift, degradation, constraint_violation, pattern_deviation, emergent_anti_pattern | High (auto-alert) | Yes (SELF) |
| SHEPHERD | remediate.py | Fix problems | Confidence > 0.8 AND risk=low | High (auto-approve); escalate on low confidence | Yes (SELF) |
| ARCHITECT | unified_loop.py | Create patterns | Recurring wound patterns (manual interventions) | HITL gated | Yes (SELF) |

**Agents emerge from receipts.** Each agent is defined by its detection taxonomy, recovery taxonomy, and fitness gradient. New agents (v12) are proposed by ARCHITECT when wound patterns stabilize, validated by humans, and bred via recombination (crossover + mutation). Success rate goal: 80% deployment success, 50% wound reduction in 90 days.

## Testing

### Smoke Tests (Fast Validation)

```bash
# v10 complete 8-phase loop
bash tests/smoke/test_smoke_v10.sh

# Individual version tests
pytest tests/test_qed_v6.py -v  # Receipts + edge_lab
pytest tests/test_qed_v7.py -v  # Physics validation
pytest tests/test_qed_v8.py -v  # DecisionPackets
pytest tests/test_qed_v9.py -v  # Causal graph
pytest tests/test_qed_v10.py -v # Agents + unified_loop
pytest tests/test_qed_v11.py -v # Immune system
```

### Full Test Suite

```bash
pytest tests/ -v --tb=short
```

All tests follow CLAUDEME.md standard: every test has `assert` statements; every receipt is verified; no silent exceptions.

## Contributing

1. Read CLAUDEME.md for project standards.
2. Every function must emit receipts (via `emit_receipt()` in receipts.py).
3. All tests must include assertions (SLO checks at minimum).
4. Use dual-hash (SHA256:BLAKE3) for all hashing.
5. No silent exceptions—use stoprules on all error paths.
6. Conventional commits: `<type>(<scope>): <description>`
   - Types: `feat`, `fix`, `refactor`, `test`, `docs`
   - Example: `feat(integrity): add entropy spike detection for v10 HUNTER`

## Requirements

- Python 3.10+
- 8GB RAM minimum
- 10GB storage
- No GPU required
- Single laptop footprint
- <2GB memory for full pattern population

## License

Proprietary. See LICENSE.txt.

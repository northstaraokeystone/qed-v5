# Smoke Tests

## Purpose

Smoke tests validate core functionality across all QED versions (v6-v10). They ensure:
- Critical modules import correctly
- Key functions exist and are callable
- Required schemas and data files are present
- Basic functionality works without deep integration testing

## Naming Convention

- Test files: `test_smoke_v{N}.sh` or `test_smoke_v{N}.py`
- Results files: `results/v{N}_YYYYMMDD.txt`

## Execution

Run a smoke test and capture output:

```bash
bash tests/smoke/test_smoke_v6.sh 2>&1 | tee tests/smoke/results/v6_$(date +%Y%m%d).txt
```

Run all smoke tests:

```bash
bash tests/smoke/smoke_tests.sh
```

## Exit Codes

- `0` = All tests passed
- `1` = One or more tests failed

## Test Coverage

- **v6**: QEDReceipt emission, edge_lab_v1, mesh_view_v1, sympy_constraints
- **v7**: ClarityClean adapter, edge_lab_v2, shared_anomalies.jsonl
- **v8**: Comprehensive integration tests (pytest-based)
- **v9**: Portfolio binder, causal_graph, event_stream
- **v10**: Entropy-based system components (entropy, integrity, remediate, unified_loop)

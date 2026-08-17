# Null probability and alert lifecycle checks

The runtime passed **14 unit tests**. The mathematical checks enumerate independent binary reference/current samples across 54 combinations of sample sizes, Bernoulli probabilities, tail directions and significance allocations. Their exact null rejection probability is compared with the claimed Hoeffding upper bound; this uses direct enumeration rather than reimplementing the monitor's decision loop.

Other tests verify:

- Actual classification loss from an ID-based historical calibration join.
- Labels before/after predictions, no quality values before labels, and per-segment accounting.
- Idempotent duplicate events and conflicting records rejected before state/time mutation.
- Lifetime spending summability and no reset after firing, resolution or cooldown.
- A real hysteretic alert lifecycle, including a neutral band and suppressed cooldown openings.
- Pending-state/identity capacity failures, reference/configuration validation and monotone arrival clocks.
- Finite Prometheus exposition, bounded/hash-based traces, and snapshots that cannot mutate live state.
- Reproducible synthetic feature versus quality changes and the metrics CLI.

Commands executed:

```bash
python -m unittest discover -s tests -v
python -m drift_sentinel demo --output measurements/demo.json --metrics-output measurements/metrics.prom
python -m drift_sentinel evaluate --output measurements/evaluation.json
```

The tests passed. The separate 24-trial-per-scenario experiment measured stationary false openings and changepoint delay, with all generated labels eventually joined. Ruff checking/formatting was run on owned Python sources/tests. Root-owned verification and publishing scripts were not edited.

These checks do not establish real model quality, the truth of external missing-label assumptions, cluster-independent customer behavior, an alerting-service SLO or a universal drift detector. The independent-null enumeration validates 54 finite cases; the general repeated-testing argument is the stated weighted-Hoeffding plus union-bound derivation, not a claim of exhaustive distribution coverage. No generic mutation coverage percentage is claimed.

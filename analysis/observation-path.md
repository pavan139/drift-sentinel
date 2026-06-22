# From observation to alert state

```mermaid
sequenceDiagram
    participant C as Independent historical cohort
    participant M as Monitor
    participant P as Prediction stream
    participant L as Delayed labels
    participant A as Channel alert state
    participant O as Local exports
    C->>M: Calibrated per-segment means counts and fingerprint
    P->>M: Prediction feature and request identity
    M->>M: Observe covariate mean and retain pending prediction
    L->>M: Observed class before or after prediction
    M->>M: Idempotent join and actual classification loss
    M->>A: Completed block and next lifetime-alpha allocation
    A->>A: Test then hysteresis resolution or cooldown
    A->>O: JSON transitions and bounded trace
    M->>O: Prometheus counts means and state
```

`monitor.py` implements independent reference calibration, bounded input records, join identity, alpha spending, two-sample tests, alert state and exports. `experiments.py` uses separate calibration/event/delay random streams to measure stationary false openings and changed-stream delay. The CLI produces JSON and real Prometheus text; it never calls a monitoring service or a model.

Read [spending-and-delay.md](spending-and-delay.md) for exact assumptions and finite-epoch limitations, and [change-trajectories.md](change-trajectories.md) for executed denominators and delays.

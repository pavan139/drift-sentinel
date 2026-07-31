# Stationary and changed trajectories

```bash
python -m drift_sentinel demo --output measurements/demo.json --metrics-output measurements/metrics.prom
python -m drift_sentinel evaluate --output measurements/evaluation.json
```

The actual default evaluation uses **24 seeds per scenario**, **2,048 predictions per trajectory**, and a changepoint at request index **1,024**. Each segment's independent historical calibration has 512 labeled examples. Runtime blocks contain 128 observations, family alpha is 0.05 across six channels, and opening requires two positive blocks. Prediction, historical calibration and label-delay generators use separate pinned random streams; there is no threshold search over evaluation outcomes.

Stationary features are uniform on `[0.4,0.6]`, with simulated classification error probability 0.1. The covariate scenario raises feature means by 0.35 while preserving error probability. The quality scenario changes error probability to 0.5 while preserving feature distribution. Label delay is independently uniform over integer offsets 0–31, and final draining joins every label. Segment assignment is random and declared before monitoring.

| Scenario | Trajectory outcome | Delay among detected trajectories |
| --- | --- | --- |
| Stationary | 0 / 24 had any false opening | Not applicable |
| Covariate mean change | 24 / 24 detected | median 255, maximum 255 |
| Quality-loss increase | 24 / 24 detected | median 270, maximum 275 |

Delays use the **arrival request index minus the changepoint**, not the number of labels or an elapsed wall-clock claim. The postchange observation horizon is 1,055 arrival indices including final label draining. No trajectory was right-censored in this particular run. There were no prechange or wrong-metric openings in these change cohorts.

The stationary observed rate is zero, but its approximate Wilson 95% interval is `[0, 0.137976]`. This bounded experiment therefore does **not** empirically establish risk at most 0.05 or zero future false alerts. The nominal family statement rests on the stated sampling assumptions and spending proof. These synthetic results are not real-provider or real-data measurements.

The demo separately uses 896 predictions, a quality-loss increase at index 256, recovery at 576, and block size 64. Global quality opens at arrival index 400 and resolves at 718. Segment alerts open/resolve on their own sample schedules; not every segment must resolve before the finite demo ends. JSON and Prometheus files contain the actual observed state and counters.

Artifacts record exact stream fingerprints, reference fingerprints, configuration, seeds, scenario, sample counts, censoring and the measured denominators. They do not reuse a release-gate benchmark or silently replenish alpha after alerts. Full experiments are local CPU/stdlib only, with no paid services, pretrained models or network calls.

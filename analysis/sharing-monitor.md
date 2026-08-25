# Sharing a monitoring revision

Reproduce the stationary and changed trajectories, then inspect both outcomes and uncertainty:

```bash
python -m pip install -e . -r requirements-dev.txt
python -m drift_sentinel evaluate --output artifacts/trajectory-check.json
python -m drift_sentinel demo --output artifacts/recovery-check.json --metrics-output artifacts/monitor.prom
python tools/check_monitor.py --require-lint --wheel
```

The saved `measurements/` use fictional labels and independently generated delays. Keep that provenance when sharing them. If you publish a different run, retain its configuration, seeds, calibration fingerprint, stream fingerprint and missed/censored cases with the result. A quiet stationary run alone does not establish zero risk.

The source repository contains the monitor and tests, not a running alert service or a durable multi-epoch budget. Applications that instantiate it must decide how to handle restart, label availability and monitoring epochs. Explain those choices alongside any service integration.

A ZIP contains no Git metadata. Initialize and make your own reviewed first commit if needed, then connect the repository you created:

```bash
git remote add origin https://github.com/YOUR_USERNAME/drift-sentinel.git
git push -u origin HEAD
```

CI runs `tools/check_monitor.py` and stores its report. The workflow is configured in source; its remote execution happens after a push. Source hashes and executed counts are reported in [verification.json](../reports/verification.json).

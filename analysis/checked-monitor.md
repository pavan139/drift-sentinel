# Checked monitoring implementation

The [machine-readable verification report](../reports/verification.json) is the authority for the executed test count, skipped tests, installation checks and source digest. The report is refreshed after a layout or source change; use its recorded commands when comparing runs.

The behavioral details are in [null-and-lifecycle-checks.md](null-and-lifecycle-checks.md), and measured workload results are in [change-trajectories.md](change-trajectories.md). [spending-and-delay.md](spending-and-delay.md) states the assumptions those checks exercise. A passing local check records that run; the GitHub workflow has its own status after publishing.

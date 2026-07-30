# Alpha spending and delayed outcomes

## Inputs, reference calibration and estimands

A prediction contains a bounded request ID, request sequence, scalar feature in `[0,1]`, a predeclared segment and binary predicted class. A label contains that request ID and the observed binary class. Joining them computes `loss = 1[predicted != observed]`. Feature observations are available immediately; quality observations exist only after the join.

`calibrate` exactly joins an independent complete historical cohort and computes reference count/feature mean/loss mean globally and per segment. It rejects duplicate, unknown or missing cohort IDs. The runtime checks reference counts and weighted global/segment means for consistency. A source fingerprint records the supplied historical data; it does not prove that a collector supplied representative or independent data.

The monitored estimands are the mean scalar feature and observed classification-loss mean. A feature mean change is a covariate signal; a loss increase is an observed quality signal. Neither establishes causality. Global quality can change through segment composition even when within-segment quality is stable. Changes in variance or other distribution features with unchanged monitored means may remain undetected.

## Repeated-testing mathematics

For `s` declared segments there are `M = 2(s+1)` channels: feature/quality globally and for each segment. Channel lifetime allocation is `alpha_stream = family_alpha / M`. Its `j`th look spends

`alpha_j = alpha_stream * 6 / (pi² j²)`.

Thus `sum_j alpha_j <= alpha_stream`, and summing channel allocations is at most `family_alpha`. Allocation is spent at every completed look, including firing and cooldown. It never resets on an alert transition.

Let the independent historical reference have size `n0` and mean `R`; a current nonoverlapping block has size `n` and mean `C`. Under equal stationary means and independent bounded observations, weighted Hoeffding gives

`P(C - R >= t) <= exp(-2t² / (1/n0 + 1/n))`.

The one-sided loss boundary is

`t = sqrt(log(1/alpha_j) * (1/n0 + 1/n) / 2)`.

The two-sided feature boundary substitutes `log(2/alpha_j)`, using a union bound over both tails. A strict crossing is a positive look. Reference uncertainty is included; the historical empirical mean is not treated as a known population constant.

The same reference and overlapping global/segment observations make tests dependent. The union bound needs no independence between tests. It does require valid marginal test bounds: independent reference versus future observations and appropriately stationary, independently sampled observations within each null channel. The resulting family statement is unconditional over calibration and future sampling. It is not a conditional guarantee for every realized historical reference.

An opening requires at least one positive look, so the probability of any false opening in this bounded epoch is at most `family_alpha` under those null assumptions. Consecutive-positive hysteresis can reduce openings but is not used to multiply p-values or assume independence. No data-dependent replenishment, repeated-peeking reset or fresh-alpha-on-recovery shortcut is implemented.

## Delayed labels and state

Predictions and labels have separate idempotency maps and a joined-ID set. Labels may precede predictions; only a complete pair contributes quality once. Conflicting duplicate contents fail before mutation. Pending queues and lifetime identity ledgers have explicit capacities. Capacity failure preserves the earlier state rather than evicting unjoined observations or silently treating them as correct.

For population-quality interpretation, label availability and arrival order must be noninformative with respect to loss, or the application must model the selection mechanism. Otherwise the monitor describes the arriving labeled cohort and its nominal population guarantee need not hold. Runtime code cannot establish that missingness assumption. The experiment uses an independent label-delay random stream and eventually joins every label.

This version is deliberately in-memory. One `Monitor` object is one monitoring epoch. A restart loses spending counters, pending joins and alerts; reopening repeated fresh epochs without a higher-level error budget does not preserve one continuous lifetime guarantee. Persistent multi-epoch supervision is outside scope.

## Alert lifecycle

- **Clear:** accumulate consecutive positive looks; open after `open_after`.
- **Firing:** require `resolve_after` looks inside `hysteresis_ratio * boundary`. Feature exit uses absolute effect; quality exit allows improvement as well as recovery.
- **Cooldown:** continue tests/spending but suppress new notifications for the configured number of completed looks, then clear the positive streak.

The runtime exports opening and resolution events and current state. It does not send messages, page an operator or assert an automatic remediation. Resolution has no formal type-II/detection guarantee. Per-segment response times differ because their sample arrival rates differ.

## Observability and resources

Prometheus text and JSON export observation counts, cumulative means, per-segment correct/incorrect joined outcomes, completed looks, spending, alert state and transitions. Undefined means remain `null` in JSON and are not emitted as misleading zero-valued Prometheus mean series. Metric labels come only from the configured bounded segment set, never request IDs.

Trace entries use request hashes and bounded metadata; they are a bounded tail, with total/evicted counts exported. Hashing is not a general anonymization guarantee. The reference fingerprint and synthetic records are local; no telemetry leaves the process. Snapshot copies cannot mutate live monitor state.

At most eight segments, 10,000 predictions, bounded pending labels/predictions, and bounded trace storage are supported. Each event touches at most two channels, so updates are expected `O(1)` dictionary work plus constant-time block testing; joined identity storage is `O(N+pending)`. Alert events are bounded by completed looks, at most proportional to `4N/block_size`. Exports are linear in channel/alert/trace output. Calibration is linear in its bounded historical cohort. The synthetic generator's delay heap costs `O(log pending)` and is not a requirement of the monitor API.

Alpha spending makes thresholds more conservative at late looks, especially with a fixed finite reference sample. The code does not promise detection of arbitrary small/late changes, a maximum delay, arbitrary distribution shift, independent conditional p-values or a production service SLO.

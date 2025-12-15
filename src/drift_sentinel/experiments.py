"""Seeded stationary/change trajectories; no real model/provider measurements."""

from __future__ import annotations
from dataclasses import asdict
import hashlib
import heapq
import json
import random
from statistics import median
from .monitor import Config, Label, Monitor, Prediction, calibrate, digest


def baseline(seed: int, per_segment=512, segments=("a", "b")):
    rng = random.Random(seed)
    predictions, labels = [], []
    for segment in segments:
        for i in range(per_segment):
            request = f"cal-{segment}-{i}"
            predicted = rng.randrange(2)
            predictions.append(Prediction(request, i, 0.4 + 0.2 * rng.random(), segment, predicted))
            labels.append(Label(request, predicted ^ int(rng.random() < 0.1)))
    return calibrate(predictions, labels, segments)


def simulate(scenario="stationary", seed=17, samples=2048, change_at=1024, recovery_at=None, config=None):
    if scenario not in ("stationary", "covariate", "quality", "recovering") or type(seed) is not int:
        raise ValueError("Unknown trajectory scenario or invalid seed")
    if (
        type(samples) is not int
        or not 64 <= samples <= 10000
        or type(change_at) is not int
        or not 0 <= change_at < samples
    ):
        raise ValueError("Trajectory length/change index is outside supported bounds")
    if recovery_at is not None and (type(recovery_at) is not int or not change_at < recovery_at < samples):
        raise ValueError("Recovery must follow the change and precede the trajectory end")
    config = config or Config()
    if config.max_predictions < samples:
        raise ValueError("Trajectory exceeds configured identity ledger")
    reference = baseline(100000 + seed, segments=config.segments)
    monitor = Monitor(reference, config)
    rng, delays = random.Random(seed), random.Random(200000 + seed)
    due = []
    source = hashlib.sha256()

    def consume_label(sequence, label):
        monitor.label(label, at=sequence)
        source.update(json.dumps({"kind": "label", "at": sequence, **asdict(label)}, sort_keys=True).encode())

    for sequence in range(samples):
        segment = rng.choice(config.segments)
        changed = sequence >= change_at and (recovery_at is None or sequence < recovery_at)
        feature = 0.4 + 0.2 * rng.random() + (0.35 if scenario == "covariate" and changed else 0)
        predicted = rng.randrange(2)
        probability = (
            0.1 if not changed or scenario in ("stationary", "covariate") else 0.9 if scenario == "recovering" else 0.5
        )
        observed = predicted ^ int(rng.random() < probability)
        record = Prediction(f"req-{sequence}", sequence, feature, segment, predicted)
        monitor.prediction(record, at=sequence)
        source.update(json.dumps({"kind": "prediction", **asdict(record)}, sort_keys=True).encode())
        heapq.heappush(due, (sequence + delays.randrange(32), sequence, Label(record.request_id, observed)))
        while due and due[0][0] <= sequence:
            _, _, label = heapq.heappop(due)
            consume_label(sequence, label)
    while due:
        arrival, _, label = heapq.heappop(due)
        consume_label(arrival, label)
    snapshot = monitor.snapshot()
    return {
        "scenario": scenario,
        "seed": seed,
        "samples": samples,
        "change_at": change_at,
        "recovery_at": recovery_at,
        "stream_sha256": source.hexdigest(),
        "state": snapshot,
        "prometheus": monitor.prometheus(),
    }


def _wilson(successes, total, z=1.959963984540054):
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * ((proportion * (1 - proportion) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {
        "lower": 0.0 if successes == 0 else max(0.0, center - radius),
        "upper": 1.0 if successes == total else min(1.0, center + radius),
        "method": "Wilson95 approximate trajectory-level interval",
    }


def demo():
    result = simulate("recovering", seed=17, samples=896, change_at=256, recovery_at=576, config=Config(block_size=64))
    result["provenance"] = "Synthetic delayed classification labels; independent historical baseline."
    result["model_evaluation"] = False
    return result


def evaluate(trials=24, samples=2048, change_at=1024):
    if type(trials) is not int or not 1 <= trials <= 48:
        raise ValueError("Use one to48 independent seeded trajectories per scenario")
    config = Config()
    rows = []
    summaries = {}
    for scenario in ("stationary", "covariate", "quality"):
        cohort = []
        expected = "covariate_mean" if scenario == "covariate" else "quality_loss"
        for seed in range(trials):
            run = simulate(scenario, seed, samples, change_at, config=config)
            opens = [event for event in run["state"]["alert_events"] if event["transition"] == "opened"]
            post = [
                event["arrival_index"] - change_at
                for event in opens
                if event["metric"] == expected and event["arrival_index"] >= change_at
            ]
            row = {
                "scenario": scenario,
                "seed": seed,
                "stream_sha256": run["stream_sha256"],
                "any_alert": bool(opens),
                "opened_alerts": len(opens),
                "prechange_alert": any(event["arrival_index"] < change_at for event in opens),
                "expected_metric_detected": bool(post) if scenario != "stationary" else None,
                "delay": min(post) if post and scenario != "stationary" else None,
                "wrong_metric_alert": any(event["metric"] != expected for event in opens)
                if scenario != "stationary"
                else None,
                "joined": run["state"]["joined"],
                "spent_alpha": sum(channel["spent_alpha"] for channel in run["state"]["channels"]),
            }
            cohort.append(row)
            rows.append(row)
        if scenario == "stationary":
            false = sum(row["any_alert"] for row in cohort)
            summaries[scenario] = {
                "trajectories": trials,
                "any_false_alert": false,
                "trajectory_false_alert_rate": false / trials,
                "interval": _wilson(false, trials),
                "lifetime_family_alpha": config.family_alpha,
            }
        else:
            delays = [row["delay"] for row in cohort if row["delay"] is not None]
            summaries[scenario] = {
                "trajectories": trials,
                "detected": len(delays),
                "detection_fraction": len(delays) / trials,
                "median_delay_among_detected": median(delays) if delays else None,
                "maximum_delay_among_detected": max(delays) if delays else None,
                "right_censored": trials - len(delays),
                "observation_horizon_after_change": samples + 31 - change_at,
                "prechange_alert_trajectories": sum(row["prechange_alert"] for row in cohort),
                "wrong_metric_alert_trajectories": sum(row["wrong_metric_alert"] for row in cohort),
            }
    payload = {
        "version": "drift-trajectories-v1",
        "config": asdict(config),
        "samples": samples,
        "change_at": change_at,
        "trajectories": [(row["scenario"], row["seed"], row["stream_sha256"]) for row in rows],
    }
    return {
        "schema_version": 1,
        "dataset_sha256": digest(payload),
        "configuration": payload["config"],
        "provenance": "Synthetic independent trajectories; simulated classification outcomes and noninformative label delays.",
        "calibration": "512 historical labeled examples per segment from a separate random stream; no threshold tuning on evaluation seeds.",
        "samples_per_trajectory": samples,
        "change_at": change_at,
        "summaries": summaries,
        "trajectories": rows,
        "model_evaluation": False,
        "external_services_called": False,
        "limitations": [
            "Family guarantee assumes independent stationary bounded observations and independent historical calibration.",
            "Label-delay/availability must be noninformative for population-quality interpretation.",
            "Mean-shift monitoring does not detect all distribution changes; no guaranteed detection delay.",
            "The stationary empirical interval is broad at this bounded trial count; zero observed alerts is not zero risk.",
            "Alpha is never replenished after alerts; very late changes face increasingly conservative thresholds.",
        ],
    }

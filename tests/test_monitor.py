from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from drift_sentinel.monitor import (
    Baseline,
    Channel,
    Config,
    Label,
    Monitor,
    Prediction,
    Reference,
    alpha_for_look,
    calibrate,
    mean_boundary,
)
from drift_sentinel.experiments import evaluate, simulate


def known_baseline(feature=0.0, loss=0.0):
    return Baseline(
        {
            "global": Reference(10000, feature, loss),
            "a": Reference(5000, feature, loss),
            "b": Reference(5000, feature, loss),
        },
        "0" * 64,
    )


class MonitorTests(unittest.TestCase):
    def test_reference_calibration_joins_ids_and_computes_actual_classification_loss(self):
        predictions = [Prediction("one", 0, 0.2, "a", 0), Prediction("two", 1, 0.8, "b", 1)]
        labels = [Label("two", 0), Label("one", 0)]
        result = calibrate(predictions, labels)
        self.assertEqual(result.references["global"], Reference(2, 0.5, 0.5))
        self.assertEqual(result.references["a"].loss_mean, 0)
        self.assertEqual(result.references["b"].loss_mean, 1)
        self.assertEqual(result, calibrate(predictions, labels))
        for invalid in ([labels[0], labels[0]], [Label("unknown", 1), labels[1]]):
            with self.assertRaises(ValueError):
                calibrate(predictions, invalid)

    def test_labels_can_arrive_before_or_after_predictions(self):
        monitor = Monitor(known_baseline())
        monitor.label(Label("before", 1), at=0)
        self.assertEqual(monitor.snapshot()["pending_labels"], 1)
        monitor.prediction(Prediction("before", 1, 0.5, "a", 0))
        monitor.prediction(Prediction("after", 2, 0.5, "b", 1))
        self.assertEqual(monitor.snapshot()["joined"], 1)
        monitor.label(Label("after", 1), at=10)
        state = monitor.snapshot()
        self.assertEqual((state["joined"], state["pending_labels"], state["pending_predictions"]), (2, 0, 0))
        self.assertEqual(monitor.channels["quality_loss", "global"].count, 2)
        self.assertEqual(monitor.channels["quality_loss", "global"].total, 1)

    def test_unlabeled_predictions_never_become_quality_observations(self):
        config = Config(block_size=16, open_after=1)
        monitor = Monitor(known_baseline(), config)
        for i in range(16):
            monitor.prediction(Prediction(f"r{i}", i, 1.0, "a", 0))
        state = monitor.snapshot()
        self.assertTrue(any(event["metric"] == "covariate_mean" for event in state["alert_events"]))
        self.assertFalse(any(event["metric"] == "quality_loss" for event in state["alert_events"]))
        self.assertEqual(monitor.channels["quality_loss", "global"].count, 0)

    def test_duplicate_inputs_are_idempotent_and_conflicts_fail_before_mutation(self):
        monitor = Monitor(known_baseline())
        prediction = Prediction("one", 1, 0.5, "a", 0)
        label = Label("one", 1)
        monitor.prediction(prediction)
        monitor.label(label)
        monitor.prediction(prediction)
        monitor.label(label)
        self.assertEqual(monitor.snapshot()["duplicate_inputs"], 2)
        self.assertEqual(monitor.snapshot()["joined"], 1)
        self.assertEqual(monitor.channels["covariate_mean", "global"].count, 1)
        before = monitor.snapshot()
        with self.assertRaises(ValueError):
            monitor.label(Label("one", 0), at=100)
        with self.assertRaises(ValueError):
            monitor.prediction(replace(prediction, feature=0.8), at=100)
        self.assertEqual(monitor.snapshot(), before)

    def test_lifetime_alpha_spending_is_not_replenished(self):
        allocation = 0.05 / 6
        total = math.fsum(alpha_for_look(allocation, j) for j in range(1, 10001))
        self.assertLess(total, allocation)
        self.assertGreater(total, 0.999 * allocation)
        self.assertAlmostEqual(alpha_for_look(allocation, 1) / alpha_for_look(allocation, 2), 4)
        for j in (0, True, 10001):
            with self.assertRaises(ValueError):
                alpha_for_look(allocation, j)

    def test_two_sample_bound_matches_independent_enumerated_null_error(self):
        checks = 0
        for reference_count, current_count in ((2, 2), (3, 4), (4, 4)):
            for probability in (0.1, 0.5, 0.9):
                for alpha in (0.05, 0.2, 0.5):
                    for two_sided in (False, True):
                        boundary = mean_boundary(reference_count, current_count, alpha, two_sided)
                        rejection = 0.0
                        for outcomes in itertools.product((0, 1), repeat=reference_count + current_count):
                            reference = sum(outcomes[:reference_count]) / reference_count
                            current = sum(outcomes[reference_count:]) / current_count
                            effect = abs(current - reference) if two_sided else current - reference
                            if effect > boundary:
                                successes = sum(outcomes)
                                rejection += probability**successes * (1 - probability) ** (len(outcomes) - successes)
                        self.assertLessEqual(rejection, alpha + 1e-12)
                        checks += 1
        self.assertEqual(checks, 54)
        self.assertAlmostEqual(mean_boundary(100, 100, 0.05, False), math.sqrt(math.log(20) * 0.01))

    def test_hysteresis_cooldown_and_look_counter_form_real_alert_lifecycle(self):
        config = Config(family_alpha=0.2, block_size=16, open_after=2, resolve_after=2, cooldown_looks=2)
        channel = Channel("quality_loss", "global", Reference(10000, 0, 0), config)
        clock = [0]

        def feed(value, blocks):
            results = []
            for _ in range(blocks * 16):
                clock[0] += 1
                result = channel.update(value, clock[0])
                if result:
                    results.append(result)
            return results

        opened = feed(1, 2)
        self.assertEqual(opened[-1]["transition"], "opened")
        self.assertEqual(channel.phase, "firing")
        self.assertTrue(all(row["transition"] is None for row in feed(0.35, 2)))
        resolved = feed(0, 2)
        self.assertEqual(resolved[-1]["transition"], "resolved")
        self.assertEqual(channel.phase, "cooldown")
        self.assertTrue(all(row["transition"] is None for row in feed(1, 2)))
        self.assertEqual(channel.phase, "clear")
        again = feed(1, 2)
        self.assertEqual(again[-1]["transition"], "opened")
        self.assertEqual(channel.looks, 10)
        self.assertLess(again[-1]["alpha"], opened[0]["alpha"])
        self.assertLess(channel.spent_alpha, channel.stream_alpha)

    def test_pending_capacity_does_not_drop_unknown_labels_silently(self):
        config = Config(max_pending=1, max_predictions=2)
        monitor = Monitor(known_baseline(), config)
        monitor.label(Label("one", 0))
        before = monitor.snapshot()
        with self.assertRaises(RuntimeError):
            monitor.label(Label("two", 1))
        self.assertEqual(monitor.snapshot(), before)
        monitor.prediction(Prediction("one", 0, 0.5, "a"))
        monitor.label(Label("two", 1))
        monitor.prediction(Prediction("two", 1, 0.5, "b"))
        self.assertEqual(monitor.snapshot()["joined"], 2)
        with self.assertRaises(RuntimeError):
            monitor.prediction(Prediction("third", 2, 0.5, "a"))

    def test_pending_prediction_capacity_allows_progress_after_label_join(self):
        monitor = Monitor(known_baseline(), Config(max_pending=1))
        monitor.prediction(Prediction("one", 1, 0.5, "a"))
        with self.assertRaises(RuntimeError):
            monitor.prediction(Prediction("two", 2, 0.5, "a"))
        monitor.label(Label("one", 0), at=3)
        monitor.prediction(Prediction("two", 2, 0.5, "a"))
        self.assertEqual(monitor.snapshot()["pending_predictions"], 1)
        self.assertEqual(monitor.snapshot()["arrival_index"], 3)

    def test_invalid_config_reference_and_arrival_order_fail_explicitly(self):
        for options in (
            {"family_alpha": 0},
            {"family_alpha": float("nan")},
            {"segments": ("global",)},
            {"block_size": 0},
            {"hysteresis_ratio": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                Config(**options)
        with self.assertRaises(ValueError):
            Monitor(Baseline({"global": Reference(2, 0, 0)}, "0" * 64))
        with self.assertRaises(ValueError):
            Monitor(known_baseline(), Config(min_reference=6000))
        monitor = Monitor(known_baseline())
        monitor.prediction(Prediction("one", 5, 0.5, "a"))
        before = monitor.snapshot()
        with self.assertRaises(ValueError):
            monitor.label(Label("one", 0), at=4)
        self.assertEqual(monitor.snapshot(), before)
        with self.assertRaises(ValueError):
            monitor.prediction(Prediction("bad", 6, 0.5, "not_admitted"))
        for value in (-1, 1.1, True, float("nan")):
            with self.assertRaises(ValueError):
                Prediction("invalid", 0, value, "a")
        with self.assertRaises(ValueError):
            Label("invalid", True)

    def test_prometheus_export_and_snapshots_are_finite_bounded_and_independent(self):
        monitor = Monitor(known_baseline(), Config(block_size=16, trace_limit=5))
        for i in range(32):
            monitor.prediction(Prediction(f"r{i}", i, 1.0, "a"))
            monitor.label(Label(f"r{i}", 1))
        state = monitor.snapshot()
        self.assertEqual(len(state["trace"]), 5)
        self.assertGreater(state["trace_events_evicted"], 0)
        self.assertTrue(all("request_id" not in row for row in state["trace"]))
        state["channels"][0]["last"]["effect"] = 999
        state["trace"][0]["kind"] = "forged"
        if state["alert_events"]:
            state["alert_events"][0]["effect"] = 999
        self.assertNotIn("forged", json.dumps(monitor.snapshot()))
        self.assertNotIn("999", json.dumps(monitor.snapshot()))
        text = monitor.prometheus()
        self.assertIn("drift_sentinel_joined_total 32", text)
        self.assertIn('metric="quality_loss",segment="global"', text)
        for row in text.splitlines():
            if not row.startswith("#"):
                self.assertTrue(math.isfinite(float(row.rsplit(" ", 1)[1])))
        json.dumps(monitor.snapshot(), allow_nan=False)

    def test_step_changes_separate_feature_mean_from_observed_quality(self):
        for scenario, metric in (("covariate", "covariate_mean"), ("quality", "quality_loss")):
            result = simulate(scenario, seed=3, samples=1024, change_at=512)
            opens = [row for row in result["state"]["alert_events"] if row["transition"] == "opened"]
            self.assertTrue(any(row["metric"] == metric and row["arrival_index"] >= 512 for row in opens))
            self.assertFalse(any(row["metric"] != metric for row in opens))
            self.assertEqual(result["state"]["joined"], 1024)
            self.assertEqual(result["state"]["pending_labels"], 0)
            self.assertEqual(result["state"]["pending_predictions"], 0)

    def test_small_seeded_stationary_and_change_experiments_are_reproducible(self):
        first = evaluate(trials=3, samples=1024, change_at=512)
        self.assertEqual(first, evaluate(trials=3, samples=1024, change_at=512))
        self.assertEqual(first["summaries"]["stationary"]["any_false_alert"], 0)
        self.assertGreater(first["summaries"]["stationary"]["interval"]["upper"], 0.05)
        self.assertFalse(first["model_evaluation"])
        self.assertTrue(all(row["spent_alpha"] < 0.05 for row in first["trajectories"]))

    def test_metrics_cli_exports_real_text(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "metrics.prom"
            result = subprocess.run(
                [sys.executable, "-m", "drift_sentinel", "metrics", "--output", str(output)],
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(output.read_text().startswith("# HELP"))
            self.assertIn("drift_sentinel_alerts_opened_total", output.read_text())


if __name__ == "__main__":
    unittest.main()

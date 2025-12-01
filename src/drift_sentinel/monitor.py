"""Sequential mean-shift monitoring with a fixed lifetime alpha budget.

Covariate means and delayed classification loss are separate signal families.
No alert resets the spending clock, and no external notification is sent.
"""

from __future__ import annotations
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bounded(value, left=0, right=1):
    return type(value) in (int, float) and math.isfinite(value) and left <= value <= right


@dataclass(frozen=True)
class Config:
    segments: tuple[str, ...] = ("a", "b")
    family_alpha: float = 0.05
    block_size: int = 128
    open_after: int = 2
    resolve_after: int = 2
    cooldown_looks: int = 2
    hysteresis_ratio: float = 0.5
    min_reference: int = 128
    max_predictions: int = 10000
    max_pending: int = 2048
    trace_limit: int = 512

    def __post_init__(self):
        if (
            not isinstance(self.segments, tuple)
            or not 1 <= len(self.segments) <= 8
            or any(type(s) is not str or not IDENTIFIER.fullmatch(s) or s == "global" for s in self.segments)
            or len(set(self.segments)) != len(self.segments)
        ):
            raise ValueError("Declare one to eight unique bounded segments before monitoring")
        if not bounded(self.family_alpha, 1e-6, 0.2) or not bounded(self.hysteresis_ratio, 0, 0.9):
            raise ValueError("Invalid lifetime alpha or alert hysteresis ratio")
        limits = (
            (self.block_size, 4, 1024),
            (self.open_after, 1, 10),
            (self.resolve_after, 1, 10),
            (self.cooldown_looks, 0, 100),
            (self.min_reference, 1, 10000),
            (self.max_predictions, 1, 10000),
            (self.max_pending, 1, 10000),
            (self.trace_limit, 1, 10000),
        )
        if any(type(value) is not int or not lo <= value <= hi for value, lo, hi in limits):
            raise ValueError("Monitoring configuration exceeds supported integer bounds")


@dataclass(frozen=True)
class Prediction:
    request_id: str
    sequence: int
    feature: float
    segment: str
    predicted_label: int = 0

    def __post_init__(self):
        if (
            type(self.request_id) is not str
            or not IDENTIFIER.fullmatch(self.request_id)
            or type(self.segment) is not str
            or not IDENTIFIER.fullmatch(self.segment)
        ):
            raise ValueError("Request and segment IDs must be bounded identifiers")
        if type(self.sequence) is not int or not 0 <= self.sequence <= 10**9 or not bounded(self.feature):
            raise ValueError("Prediction sequence or bounded feature is invalid")
        if type(self.predicted_label) is not int or self.predicted_label not in (0, 1):
            raise ValueError("Predicted classification label must be integer zero or one")


@dataclass(frozen=True)
class Label:
    request_id: str
    observed_label: int

    def __post_init__(self):
        if (
            type(self.request_id) is not str
            or not IDENTIFIER.fullmatch(self.request_id)
            or type(self.observed_label) is not int
            or self.observed_label not in (0, 1)
        ):
            raise ValueError("Label requires a bounded request ID and integer zero/one outcome")


@dataclass(frozen=True)
class Reference:
    count: int
    feature_mean: float
    loss_mean: float

    def __post_init__(self):
        if (
            type(self.count) is not int
            or not 1 <= self.count <= 10000
            or not bounded(self.feature_mean)
            or not bounded(self.loss_mean)
        ):
            raise ValueError("Invalid independent reference sample statistics")


@dataclass(frozen=True)
class Baseline:
    references: dict[str, Reference]
    source_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.references, dict)
            or not self.references
            or not all(type(k) is str and isinstance(v, Reference) for k, v in self.references.items())
        ):
            raise ValueError("Baseline requires validated per-segment references")
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("Baseline source fingerprint must be SHA-256")


def calibrate(predictions: list[Prediction], labels: list[Label], segments=("a", "b")) -> Baseline:
    Config(segments=segments)
    if (
        not isinstance(predictions, list)
        or not isinstance(labels, list)
        or not 1 <= len(predictions) <= 10000
        or len(predictions) != len(labels)
    ):
        raise ValueError("Calibration needs aligned bounded complete prediction/label cohorts")
    if not all(isinstance(row, Prediction) for row in predictions) or not all(isinstance(row, Label) for row in labels):
        raise ValueError("Calibration rows must be validated prediction and label records")
    outcomes = {row.request_id: row.observed_label for row in labels}
    if (
        len(outcomes) != len(labels)
        or len({row.request_id for row in predictions}) != len(predictions)
        or set(outcomes) != {row.request_id for row in predictions}
    ):
        raise ValueError("Calibration cohort IDs must be unique and exactly joined")
    if any(row.segment not in segments for row in predictions):
        raise ValueError("Calibration contains an undeclared segment")
    references = {}
    for segment in ("global", *segments):
        selected = [row for row in predictions if segment == "global" or row.segment == segment]
        if not selected:
            raise ValueError("Every declared segment needs calibration observations")
        references[segment] = Reference(
            len(selected),
            math.fsum(row.feature for row in selected) / len(selected),
            sum(row.predicted_label != outcomes[row.request_id] for row in selected) / len(selected),
        )
    return Baseline(
        references,
        digest({"predictions": [asdict(row) for row in predictions], "labels": [asdict(row) for row in labels]}),
    )


def alpha_for_look(stream_alpha: float, look: int):
    if not bounded(stream_alpha, 1e-10, 0.2) or type(look) is not int or not 1 <= look <= 10000:
        raise ValueError("A look must have a positive lifetime alpha allocation and index")
    return stream_alpha * 6 / (math.pi**2 * look**2)


def mean_boundary(reference_count: int, current_count: int, alpha: float, two_sided=True):
    if any(type(n) is not int or not 1 <= n <= 10000 for n in (reference_count, current_count)) or not bounded(
        alpha, 1e-15, 0.999999
    ):
        raise ValueError("Invalid two-sample mean test inputs")
    return math.sqrt(math.log((2 if two_sided else 1) / alpha) * (1 / reference_count + 1 / current_count) / 2)


class Channel:
    def __init__(self, metric, segment, reference: Reference, config: Config):
        self.metric, self.segment, self.reference, self.config = metric, segment, reference, config
        self.stream_alpha = config.family_alpha / (2 * (len(config.segments) + 1))
        self.count = self.bucket_count = self.looks = self.bad_streak = self.good_streak = self.cooldown_remaining = 0
        self.total = 0.0
        self.sum_values = 0.0
        self.spent_alpha = 0.0
        self.phase = "clear"
        self.opened = self.resolved = 0
        self.last = None

    def update(self, value, arrival):
        if not bounded(value):
            raise ValueError("Channel values must be bounded to [0,1]")
        self.count += 1
        self.bucket_count += 1
        self.total += value
        self.sum_values += value
        if self.bucket_count < self.config.block_size:
            return None
        self.looks += 1
        allocation = alpha_for_look(self.stream_alpha, self.looks)
        self.spent_alpha += allocation
        current_mean = self.total / self.bucket_count
        baseline = self.reference.feature_mean if self.metric == "covariate_mean" else self.reference.loss_mean
        effect = current_mean - baseline
        boundary = mean_boundary(self.reference.count, self.bucket_count, allocation, self.metric == "covariate_mean")
        deviation = abs(effect) if self.metric == "covariate_mean" else effect
        signal = deviation > boundary
        inside_exit = deviation <= self.config.hysteresis_ratio * boundary
        transition = None
        if self.phase == "cooldown":
            self.cooldown_remaining -= 1
            self.bad_streak = self.good_streak = 0
            if self.cooldown_remaining <= 0:
                self.phase = "clear"
        elif self.phase == "clear":
            self.bad_streak = self.bad_streak + 1 if signal else 0
            if self.bad_streak >= self.config.open_after:
                self.phase = "firing"
                self.bad_streak = 0
                self.opened += 1
                transition = "opened"
        else:
            self.good_streak = self.good_streak + 1 if inside_exit else 0
            if self.good_streak >= self.config.resolve_after:
                self.phase = "cooldown" if self.config.cooldown_looks else "clear"
                self.cooldown_remaining = self.config.cooldown_looks
                self.good_streak = 0
                self.resolved += 1
                transition = "resolved"
        result = {
            "metric": self.metric,
            "segment": self.segment,
            "look": self.looks,
            "arrival_index": arrival,
            "sample_count": self.bucket_count,
            "reference_count": self.reference.count,
            "current_mean": current_mean,
            "reference_mean": baseline,
            "effect": effect,
            "boundary": boundary,
            "alpha": allocation,
            "spent_alpha": self.spent_alpha,
            "signal": signal,
            "phase": self.phase,
            "transition": transition,
        }
        self.total = 0.0
        self.bucket_count = 0
        self.last = result
        return result

    def snapshot(self):
        return {
            "metric": self.metric,
            "segment": self.segment,
            "observations": self.count,
            "looks": self.looks,
            "pending_block_count": self.bucket_count,
            "cumulative_mean": self.sum_values / self.count if self.count else None,
            "quality_correct": self.count - self.sum_values if self.metric == "quality_loss" else None,
            "quality_incorrect": self.sum_values if self.metric == "quality_loss" else None,
            "phase": self.phase,
            "opened": self.opened,
            "resolved": self.resolved,
            "spent_alpha": self.spent_alpha,
            "stream_alpha": self.stream_alpha,
            "last": dict(self.last) if self.last else None,
        }


class Monitor:
    def __init__(self, baseline: Baseline, config: Config | None = None):
        self.config = config or Config()
        if not isinstance(baseline, Baseline) or set(baseline.references) != {"global", *self.config.segments}:
            raise ValueError("Baseline must match all predeclared monitoring segments")
        if any(reference.count < self.config.min_reference for reference in baseline.references.values()):
            raise ValueError("Independent baseline sample is too small")
        global_ref = baseline.references["global"]
        if global_ref.count != sum(baseline.references[segment].count for segment in self.config.segments):
            raise ValueError("Global and segment calibration cohort counts disagree")
        for name in ("feature_mean", "loss_mean"):
            weighted = (
                sum(getattr(baseline.references[s], name) * baseline.references[s].count for s in self.config.segments)
                / global_ref.count
            )
            if not math.isclose(weighted, getattr(global_ref, name), rel_tol=0, abs_tol=1e-12):
                raise ValueError("Global calibration mean does not match the segment cohorts")
        self.baseline = Baseline(dict(baseline.references), baseline.source_sha256)
        self.channels = {
            (metric, segment): Channel(metric, segment, reference, self.config)
            for metric in ("covariate_mean", "quality_loss")
            for segment, reference in baseline.references.items()
        }
        self.predictions = {}
        self.labels = {}
        self.joined = set()
        self.arrival_index = 0
        self.duplicates = 0
        self.alert_events = []
        self.trace = deque(maxlen=self.config.trace_limit)
        self.trace_events_total = 0

    def _arrival(self, at):
        if type(at) is not int or not self.arrival_index <= at <= 10**9:
            raise ValueError("Arrival indices must be monotone bounded integers")
        self.arrival_index = at

    def _trace(self, kind, request_id=None, **metadata):
        row = {"kind": kind, "arrival_index": self.arrival_index, **metadata}
        if request_id is not None:
            row["request_hash"] = hashlib.sha256(request_id.encode()).hexdigest()[:16]
        self.trace.append(row)
        self.trace_events_total += 1

    def _observe(self, metric, segment, value):
        for scope in ("global", segment):
            result = self.channels[metric, scope].update(value, self.arrival_index)
            if result:
                self._trace("look", **{key: value for key, value in result.items() if key != "arrival_index"})
                if result["transition"]:
                    self.alert_events.append(dict(result))

    def _join(self, request_id):
        if request_id in self.predictions and request_id in self.labels and request_id not in self.joined:
            prediction = self.predictions[request_id]
            loss = int(prediction.predicted_label != self.labels[request_id])
            self.joined.add(request_id)
            self._observe("quality_loss", prediction.segment, loss)
            self._trace("joined", request_id, loss=loss, segment=prediction.segment)

    def prediction(self, record: Prediction, at: int | None = None):
        if not isinstance(record, Prediction) or record.segment not in self.config.segments:
            raise ValueError("Prediction belongs to an undeclared segment")
        if record.request_id in self.predictions:
            if self.predictions[record.request_id] != record:
                raise ValueError("Conflicting prediction for request ID")
            self._arrival(max(self.arrival_index, record.sequence) if at is None else at)
            self.duplicates += 1
            return "duplicate"
        if len(self.predictions) >= self.config.max_predictions:
            raise RuntimeError("Prediction identity ledger capacity exhausted")
        if record.request_id not in self.labels and len(self.predictions) - len(self.joined) >= self.config.max_pending:
            raise RuntimeError("Pending prediction capacity exhausted")
        self._arrival(max(self.arrival_index, record.sequence) if at is None else at)
        self.predictions[record.request_id] = record
        self._observe("covariate_mean", record.segment, record.feature)
        self._trace("prediction", record.request_id, segment=record.segment, sequence=record.sequence)
        self._join(record.request_id)
        return "accepted"

    def label(self, record: Label, at: int | None = None):
        if not isinstance(record, Label):
            raise ValueError("Expected a validated classification label")
        if record.request_id in self.labels:
            if self.labels[record.request_id] != record.observed_label:
                raise ValueError("Conflicting observed label for request ID")
            self._arrival(self.arrival_index if at is None else at)
            self.duplicates += 1
            return "duplicate"
        if record.request_id not in self.predictions and len(self.labels) - len(self.joined) >= self.config.max_pending:
            raise RuntimeError("Pending label capacity exhausted")
        if len(self.labels) >= self.config.max_predictions + self.config.max_pending:
            raise RuntimeError("Label identity ledger capacity exhausted")
        self._arrival(self.arrival_index if at is None else at)
        self.labels[record.request_id] = record.observed_label
        self._trace("label", record.request_id)
        self._join(record.request_id)
        return "accepted"

    def snapshot(self):
        return {
            "config": asdict(self.config),
            "baseline_sha256": self.baseline.source_sha256,
            "references": {name: asdict(reference) for name, reference in self.baseline.references.items()},
            "arrival_index": self.arrival_index,
            "predictions": len(self.predictions),
            "labels": len(self.labels),
            "joined": len(self.joined),
            "pending_predictions": len(self.predictions) - len(self.joined),
            "pending_labels": len(self.labels) - len(self.joined),
            "duplicate_inputs": self.duplicates,
            "trace_events_total": self.trace_events_total,
            "trace_events_evicted": self.trace_events_total - len(self.trace),
            "channels": [self.channels[key].snapshot() for key in sorted(self.channels)],
            "alert_events": [dict(row) for row in self.alert_events],
            "trace": [dict(row) for row in self.trace],
            "guarantee_scope": "Lifetime family false-opening probability under independent stationary bounded observations and noninformative label arrival; not universal drift detection.",
        }

    def prometheus(self):
        rows = [
            "# HELP drift_sentinel_predictions_total Unique predictions observed.",
            "# TYPE drift_sentinel_predictions_total counter",
            f"drift_sentinel_predictions_total {len(self.predictions)}",
            "# TYPE drift_sentinel_labels_total counter",
            f"drift_sentinel_labels_total {len(self.labels)}",
            "# TYPE drift_sentinel_joined_total counter",
            f"drift_sentinel_joined_total {len(self.joined)}",
            "# TYPE drift_sentinel_pending_predictions gauge",
            f"drift_sentinel_pending_predictions {len(self.predictions) - len(self.joined)}",
            "# TYPE drift_sentinel_pending_labels gauge",
            f"drift_sentinel_pending_labels {len(self.labels) - len(self.joined)}",
            "# TYPE drift_sentinel_duplicate_inputs_total counter",
            f"drift_sentinel_duplicate_inputs_total {self.duplicates}",
        ]
        names = {
            "observations_total": "counter",
            "observed_mean": "gauge",
            "looks_total": "counter",
            "alerts_opened_total": "counter",
            "alerts_resolved_total": "counter",
            "alert_firing": "gauge",
            "alpha_spent": "gauge",
            "last_effect": "gauge",
            "last_boundary": "gauge",
        }
        for name, kind in names.items():
            rows.append(f"# TYPE drift_sentinel_{name} {kind}")
            for key in sorted(self.channels):
                channel = self.channels[key]
                if name == "observed_mean" and not channel.count:
                    continue
                values = {
                    "observations_total": channel.count,
                    "observed_mean": channel.sum_values / channel.count if channel.count else 0,
                    "looks_total": channel.looks,
                    "alerts_opened_total": channel.opened,
                    "alerts_resolved_total": channel.resolved,
                    "alert_firing": int(channel.phase == "firing"),
                    "alpha_spent": channel.spent_alpha,
                    "last_effect": channel.last["effect"] if channel.last else 0,
                    "last_boundary": channel.last["boundary"] if channel.last else 0,
                }
                rows.append(
                    f'drift_sentinel_{name}{{metric="{channel.metric}",segment="{channel.segment}"}} {values[name]:.12g}'
                )
        return "\n".join(rows) + "\n"

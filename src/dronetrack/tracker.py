"""Single-target tracking in the image plane.

`BernoulliTracker` is a Bernoulli filter with probabilistic data association: a constant-velocity
Kalman filter on (x, y, vx, vy) plus a presence probability q that the drone is in view.
Notation follows the report (F, Q, S, nu, beta, q). `CounterTracker` is the conventional
alternative used as an ablation: nearest-neighbour association and hit/miss counters.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from dronetrack.config import CounterTrackerConfig, TrackerConfig
from dronetrack.structures import Detection

H = np.hstack([np.eye(2), np.zeros((2, 2))])


def transition(dt: float, process_noise: float) -> tuple[np.ndarray, np.ndarray]:
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    block = process_noise * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    return F, np.kron(block, np.eye(2))  # state order (x, y, vx, vy)


def inverse_2x2(S: np.ndarray) -> tuple[np.ndarray, float]:
    """Inverse and determinant of a 2x2 matrix in closed form"""
    (a, b), (c, d) = S
    det = a * d - b * c
    return np.array([[d, -b], [-c, a]]) / det, float(det)


class KalmanTrack:
    """State, covariance and box size shared by both trackers"""

    def __init__(self, process_noise: float, measurement_std: float, init_velocity_std: float, smoothing: float):
        self.process_noise = process_noise
        self.R = measurement_std**2 * np.eye(2)
        self.init_velocity_std = init_velocity_std
        self.smoothing = smoothing
        self.x: np.ndarray | None = None
        self.P = np.eye(4)
        self.size = (np.nan, np.nan)
        self.t: float | None = None

    @property
    def exists(self) -> bool:
        return self.x is not None

    def start(self, detection: Detection, t: float) -> None:
        self.x = np.array([detection.x, detection.y, 0.0, 0.0])
        self.P = np.diag([*np.diag(self.R), self.init_velocity_std**2, self.init_velocity_std**2])
        self.size = (detection.w, detection.h)
        self.t = t

    def reset(self) -> None:
        self.x, self.t = None, None

    def predict(self, t: float) -> None:
        if self.x is None or self.t is None:
            return
        F, Q = transition(max(t - self.t, 0.0), self.process_noise)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def innovation_cov(self) -> np.ndarray:
        return H @ self.P @ H.T + self.R

    def resize(self, detection: Detection) -> None:
        a = self.smoothing
        w, h = self.size
        self.size = (
            (detection.w, detection.h)
            if np.isnan(w)
            else ((1 - a) * w + a * detection.w, (1 - a) * h + a * detection.h)
        )


class BernoulliTracker:
    def __init__(self, config: TrackerConfig):
        self.c = config
        self.track = KalmanTrack(
            config.process_noise, config.measurement_std, config.init_velocity_std, config.size_smoothing
        )
        self.q = 0.0
        self.gamma = float(chi2.ppf(config.gate_probability, df=2))

    @property
    def declared(self) -> bool:
        return self.track.exists and self.q >= self.c.declare

    def predict(self, t: float) -> None:
        self.track.predict(t)
        self.q *= self.c.p_survival

    def update(self, detections: list[Detection], t: float) -> None:
        track = self.track
        if not track.exists:
            if detections:
                track.start(max(detections, key=lambda d: d.score), t)
                self.q = self.c.q_birth
            return

        assert track.x is not None
        S = track.innovation_cov()
        S_inv, det_S = inverse_2x2(S)
        nu = np.array([d.xy for d in detections]).reshape(-1, 2) - H @ track.x
        d2 = np.einsum("ni,ij,nj->n", nu, S_inv, nu)
        gated = d2 <= self.gamma
        nu, chosen = nu[gated], [d for d, g in zip(detections, gated, strict=True) if g]

        pd, pg = self.c.p_detect, self.c.gate_probability
        g = np.exp(-0.5 * d2[gated]) / (2 * np.pi * np.sqrt(det_S))
        ratio = pd * g / self.c.clutter_density
        L = 1 - pd * pg + ratio.sum()
        self.q = L * self.q / (1 - self.q + L * self.q)

        if len(nu):
            beta = ratio / L
            beta0 = (1 - pd * pg) / L
            K = track.P @ H.T @ S_inv
            nu_bar = beta @ nu
            spread = (beta[:, None, None] * np.einsum("ni,nj->nij", nu, nu)).sum(0) - np.outer(nu_bar, nu_bar)
            track.x = track.x + K @ nu_bar
            track.P = track.P - (1 - beta0) * K @ S @ K.T + K @ spread @ K.T
            best = int(np.argmax(beta))
            if beta[best] > 0.5:
                track.resize(chosen[best])

        if self.q < self.c.drop:
            track.reset()
            self.q = 0.0


class CounterTracker:
    """Kalman filter with nearest-neighbour association, confirmed after n hits, dropped after m misses"""

    def __init__(self, tracker: TrackerConfig, counter: CounterTrackerConfig):
        self.c = counter
        self.track = KalmanTrack(
            tracker.process_noise, tracker.measurement_std, tracker.init_velocity_std, tracker.size_smoothing
        )
        self.gamma = float(chi2.ppf(tracker.gate_probability, df=2))
        self.hits = 0
        self.misses = 0

    @property
    def q(self) -> float:
        return 1.0 if self.declared else 0.0

    @property
    def declared(self) -> bool:
        return self.track.exists and self.hits >= self.c.confirm_hits

    def predict(self, t: float) -> None:
        self.track.predict(t)

    def update(self, detections: list[Detection], t: float) -> None:
        track = self.track
        if not track.exists:
            if detections:
                track.start(max(detections, key=lambda d: d.score), t)
                self.hits, self.misses = 1, 0
            return

        assert track.x is not None
        S_inv, _ = inverse_2x2(track.innovation_cov())
        nu = np.array([d.xy for d in detections]).reshape(-1, 2) - H @ track.x
        d2 = np.einsum("ni,ij,nj->n", nu, S_inv, nu)
        nearest = int(np.argmin(d2)) if len(d2) else -1
        if nearest < 0 or d2[nearest] > self.gamma:
            self.misses += 1
            if self.misses >= self.c.max_misses:
                track.reset()
                self.hits = 0
            return

        K = track.P @ H.T @ S_inv
        track.x = track.x + K @ nu[nearest]
        track.P = (np.eye(4) - K @ H) @ track.P
        track.resize(detections[nearest])
        self.hits += 1
        self.misses = 0

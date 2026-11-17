import numpy as np

from dronetrack.config import CounterTrackerConfig, TrackerConfig
from dronetrack.structures import Detection
from dronetrack.tracker import BernoulliTracker, CounterTracker, transition


def detection(x, y, score=0.9):
    return Detection(x, y, 6, 6, score)


def test_transition_matches_constant_velocity_model():
    F, Q = transition(0.1, 2.0)
    assert np.allclose(F @ [0, 0, 10, -5], [1, -0.5, 10, -5])
    assert np.allclose(Q[0, 0], 2.0 * 0.1**3 / 3)
    assert np.allclose(Q[0, 2], 2.0 * 0.1**2 / 2)
    assert Q[0, 1] == 0 and Q[0, 3] == 0
    assert np.allclose(Q, Q.T)


def test_single_detection_update_equals_kalman_update():
    """With p_D = 1 and negligible clutter, PDA reduces to the standard Kalman update"""
    c = TrackerConfig(p_detect=1.0, clutter_density=1e-12, measurement_std=2.0, process_noise=100.0)
    tracker = BernoulliTracker(c)
    tracker.update([detection(100, 100)], t=0.0)
    tracker.predict(0.1)
    P_prior, x_prior = tracker.track.P.copy(), tracker.track.x.copy()
    tracker.update([detection(103, 99)], t=0.1)

    H = np.hstack([np.eye(2), np.zeros((2, 2))])
    S = H @ P_prior @ H.T + 4 * np.eye(2)
    K = P_prior @ H.T @ np.linalg.inv(S)
    assert np.allclose(tracker.track.x, x_prior + K @ ([103, 99] - H @ x_prior), atol=1e-6)
    assert np.allclose(tracker.track.P, (np.eye(4) - K @ H) @ P_prior, atol=1e-6)


def test_presence_rises_with_detections_and_decays_without():
    tracker = BernoulliTracker(TrackerConfig())
    for k in range(5):
        tracker.predict(k / 30)
        tracker.update([detection(200 + k, 100)], t=k / 30)
    assert tracker.declared and tracker.q > 0.95
    for k in range(5, 200):
        tracker.predict(k / 30)
        tracker.update([], t=k / 30)
        if not tracker.track.exists:
            break
    assert not tracker.declared and tracker.q == 0.0


def test_symmetric_detections_keep_the_mean_and_spread_the_covariance_along_them():
    tracker = BernoulliTracker(TrackerConfig(clutter_density=1e-3))
    tracker.update([detection(0, 0)], t=0.0)
    tracker.predict(1 / 30)
    x_prior = tracker.track.x.copy()
    tracker.update([detection(3, 0), detection(-3, 0)], t=1 / 30)
    P = tracker.track.P
    assert np.allclose(tracker.track.x[:2], x_prior[:2], atol=1e-9)
    assert np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0)
    assert P[0, 0] > P[1, 1]


def test_counter_tracker_confirms_after_n_hits():
    tracker = CounterTracker(TrackerConfig(), CounterTrackerConfig(confirm_hits=3, max_misses=2))
    for k in range(3):
        assert not tracker.declared
        tracker.predict(k / 30)
        tracker.update([detection(50, 50)], t=k / 30)
    assert tracker.declared


def test_counter_tracker_drops_after_misses():
    tracker = CounterTracker(TrackerConfig(), CounterTrackerConfig(confirm_hits=2, max_misses=2))
    for k in range(3):
        tracker.predict(k / 30)
        tracker.update([detection(50, 50)], t=k / 30)
    assert tracker.declared
    for k in range(3, 5):
        tracker.predict(k / 30)
        tracker.update([], t=k / 30)
    assert not tracker.track.exists

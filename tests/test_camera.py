import numpy as np

from dronetrack.camera import Camera


def test_pinhole_angles_and_round_numbers():
    camera = Camera.pinhole((1920, 1080), hfov_deg=90)
    center = camera.angles(np.array([[960, 540]]))[0]
    edge = camera.angles(np.array([[1920, 540]]))[0]
    top = camera.angles(np.array([[960, 0]]))[0]
    assert np.allclose(center, 0)
    assert np.isclose(edge[0], np.pi / 4) and np.isclose(edge[1], 0)
    assert top[1] > 0  # image rows grow downwards, elevation upwards


def test_direction_covariance_matches_analytic_jacobian():
    camera = Camera.pinhole((1920, 1080), hfov_deg=60)
    f = camera.K[0, 0]
    angles, cov = camera.direction(np.array([960.0, 540.0]), np.eye(2) * 4.0)
    assert np.allclose(angles, 0)
    assert np.allclose(cov, np.eye(2) * 4.0 / f**2, rtol=1e-3)


def test_resized_calibration_scales_intrinsics(tmp_path):
    path = tmp_path / "cam.json"
    path.write_text(
        '{"K-matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], '
        '"distCoeff": [0, 0, 0, 0, 0], "resolution": [1920, 1080]}'
    )
    camera = Camera.from_json(path, (3840, 2160))
    assert np.allclose(camera.K, [[2000, 0, 1920], [0, 2000, 1080], [0, 0, 1]])

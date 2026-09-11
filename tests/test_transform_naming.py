import numpy as np

from processing.transform import czt


def test_czt_exposes_explicit_depth_and_sampling_names():
    signal = np.ones(16, dtype=complex)

    _, depth_axis_m = czt(
        signal=signal,
        depth_min_m=0.0,
        depth_max_m=1e-3,
        fs_k=1e6,
        n_points=32,
    )

    assert depth_axis_m.shape == (32,)
    assert depth_axis_m[0] == 0.0
    assert depth_axis_m[-1] == 1e-3 * (31 / 32)

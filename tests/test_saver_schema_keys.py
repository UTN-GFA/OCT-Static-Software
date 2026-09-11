from datetime import datetime

import numpy as np
import pytest

from storage.saver import OCTDataSaver, SaveConfig


def _metadata():
    return {
        "software_version": "OCT_Static_Software V6.0",
        "sample_name": "schema-test",
        "dark_enabled": True,
        "nonlinearity_enabled": False,
        "position_tolerance_um": 2.0,
        "position_tolerance_mm": 0.002,
        "optics": {
            "fiber_diameter_um": 9.0,
            "wavelength_nm": 1310.0,
            "collimator_focal_length_mm": 18.0,
            "objective_focal_length_mm": 40.0,
            "spot_diffraction_um": 12.3,
            "lateral_resolution_um": 12.3,
            "rayleigh_range_um": 45.6,
        },
        "windows": [{"depth_min_m": 0.0, "depth_max_m": 0.003}],
        "k_samples": 17,
        "global_profile_samples": 19,
        "start_time": datetime(2026, 1, 1),
    }


def _save_config():
    return SaveConfig(save_spectra=True, save_peaks=False, save_profile=False)


def test_npz_persists_z_mm_and_omits_processing_sample_counts(tmp_path):
    saver = OCTDataSaver()
    saver.open_scan(
        str(tmp_path / "scan"),
        planned_points=1,
        n_pixels=2,
        metadata=_metadata(),
        save_cfg=_save_config(),
        file_format="npz",
        measurements_per_point=2,
        wavelengths_nm=np.array([800.0, 801.0]),
    )
    saver.write_point(
        idx=0,
        x_mm=1.0,
        y_mm=2.0,
        z_mechanical_mm=3.0,
        spectra=np.zeros((2, 2)),
        depth_m=None,
        amplitude=None,
    )

    output = saver.close_scan()

    with np.load(output, allow_pickle=False) as data:
        assert "z_mm" in data.files
        assert data["position_tolerance_um"] == pytest.approx(2.0)
        assert data["position_tolerance_mm"] == pytest.approx(0.002)
        assert "z_mechanical_mm" not in data.files
        assert "dark" in data.files
        assert "nonlinearity" in data.files
        assert "o_d_fib_um" in data.files
        assert "o_wl_nm" in data.files
        assert "o_f_col_mm" in data.files
        assert "o_f_obj_mm" in data.files
        assert "spot_diffraction_um" not in data.files
        assert "lateral_resolution_um" not in data.files
        assert "rayleigh_range_um" not in data.files
        assert "dark_enabled" not in data.files
        assert "nonlinearity_enabled" not in data.files
        assert "optics_fiber_diameter_um" not in data.files
        assert "k_samples" not in data.files
        assert "global_profile_samples" not in data.files
        assert "measurements_per_point" not in data.files
        assert "duration_s" not in data.files
        assert "written_points" not in data.files


def test_hdf5_persists_z_mm_and_omits_processing_sample_counts(tmp_path):
    h5py = pytest.importorskip("h5py")

    saver = OCTDataSaver()
    saver.open_scan(
        str(tmp_path / "scan"),
        planned_points=1,
        n_pixels=2,
        metadata=_metadata(),
        save_cfg=_save_config(),
        file_format="h5",
        measurements_per_point=2,
        wavelengths_nm=np.array([800.0, 801.0]),
    )
    saver.write_point(
        idx=0,
        x_mm=1.0,
        y_mm=2.0,
        z_mechanical_mm=3.0,
        spectra=np.zeros((2, 2)),
        depth_m=None,
        amplitude=None,
    )

    output = saver.close_scan()

    with h5py.File(output, "r") as data:
        assert "z_mm" in data
        assert data.attrs["position_tolerance_um"] == pytest.approx(2.0)
        assert data.attrs["position_tolerance_mm"] == pytest.approx(0.002)
        assert "z_mechanical_mm" not in data
        assert "dark" in data.attrs
        assert "nonlinearity" in data.attrs
        assert "o_d_fib_um" in data.attrs
        assert "o_wl_nm" in data.attrs
        assert "o_f_col_mm" in data.attrs
        assert "o_f_obj_mm" in data.attrs
        assert "spot_diffraction_um" not in data.attrs
        assert "lateral_resolution_um" not in data.attrs
        assert "rayleigh_range_um" not in data.attrs
        assert "dark_enabled" not in data.attrs
        assert "nonlinearity_enabled" not in data.attrs
        assert "optics_fiber_diameter_um" not in data.attrs
        assert "k_samples" not in data.attrs
        assert "global_profile_samples" not in data.attrs
        assert "measurements_per_point" not in data.attrs
        assert "duration_s" not in data.attrs
        assert "written_points" not in data


def test_npz_profile_mode_persists_modulus_and_axis(tmp_path):
    saver = OCTDataSaver()
    saver.open_scan(
        str(tmp_path / "profile"),
        planned_points=1,
        n_pixels=2,
        metadata=_metadata(),
        save_cfg=SaveConfig(
            save_spectra=False,
            save_peaks=False,
            save_profile=True,
            profile_mode="modulus",
        ),
        file_format="npz",
        measurements_per_point=1,
        wavelengths_nm=np.array([800.0, 801.0]),
    )
    saver.write_point(
        idx=0,
        x_mm=1.0,
        y_mm=2.0,
        z_mechanical_mm=3.0,
        spectra=None,
        depth_m=None,
        amplitude=None,
        profiles=[{0: np.array([3.0 + 4.0j, 5.0 + 12.0j])}],
        profile_depth_axes_m={0: np.array([0.0, 1e-6])},
    )

    output = saver.close_scan()

    with np.load(output, allow_pickle=False) as data:
        assert "profile_mod_w0" in data.files
        assert "profile_depth_m_w0" in data.files
        assert "profile_depth_axis_m_w0" not in data.files
        assert "profile_real_w0" not in data.files
        assert "profile_imag_w0" not in data.files


@pytest.mark.parametrize("file_format", ["npz", "h5"])
def test_saver_requires_same_sequential_idx_contract_for_all_formats(
    tmp_path, file_format
):
    if file_format == "h5":
        pytest.importorskip("h5py")

    saver = OCTDataSaver()
    saver.open_scan(
        str(tmp_path / f"indexed_{file_format}"),
        planned_points=2,
        n_pixels=2,
        metadata=_metadata(),
        save_cfg=_save_config(),
        file_format=file_format,
        measurements_per_point=1,
        wavelengths_nm=np.array([800.0, 801.0]),
    )

    point = dict(
        x_mm=1.0,
        y_mm=2.0,
        z_mechanical_mm=3.0,
        spectra=np.zeros((1, 2)),
        depth_m=None,
        amplitude=None,
    )
    with pytest.raises(ValueError, match="índices secuenciales"):
        saver.write_point(idx=1, **point)

    saver.write_point(idx=0, **point)
    with pytest.raises(ValueError, match="índices secuenciales"):
        saver.write_point(idx=0, **point)

    saver.close_scan(aborted=True)

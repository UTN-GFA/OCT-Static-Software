from dataclasses import fields

from processing.optics import OpticsResult, calculate_optics


def test_optics_result_uses_gaussian_beam_propagation():
    result = calculate_optics(
        fiber_diameter_um=5.0,
        wavelength_nm=850.0,
        collimator_focal_length_mm=11.0,
        objective_focal_length_mm=10.0,
    )
    field_names = {field.name for field in fields(OpticsResult)}

    assert "spot_gaussian_um" in field_names
    assert "spot_geometric_um" not in field_names
    assert "spot_diffraction_um" not in field_names
    assert "lateral_resolution_um" not in field_names
    assert "rayleigh_range_um" not in field_names
    assert not hasattr(result, "spot_diffraction_um")
    assert not hasattr(result, "rayleigh_range_um")
    assert not hasattr(result, "lateral_resolution_um")
    assert result.spot_gaussian_um > 0
    assert result.confocal_parameter_um > 0


def test_gaussian_focal_diameter_responds_to_wavelength():
    common = dict(
        fiber_diameter_um=5.0,
        collimator_focal_length_mm=6.0,
        objective_focal_length_mm=18.0,
    )
    short = calculate_optics(wavelength_nm=850.0, **common)
    long = calculate_optics(wavelength_nm=1310.0, **common)

    assert short.spot_gaussian_um != long.spot_gaussian_um
    assert short.confocal_parameter_um != long.confocal_parameter_um

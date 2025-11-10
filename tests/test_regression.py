"""Regression test for the water rocket range prediction."""

from pathlib import Path
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from water_rocket_simulation import (
    build_parameters_from_measurements,
    compute_range,
    estimate_air_density,
)


REFERENCE_RANGE = 65.52637420332036


def test_range_regression() -> None:
    """Ensure the simulated range remains within ±2% of the reference value."""

    ambient_density = estimate_air_density(29.0)

    params = build_parameters_from_measurements(
        nose_mass_g=60.0,
        water_volume_ml=385.0,
        bottle_volume_ml=3000.0,
        nozzle_diameter_mm=20.0,
        body_diameter_mm=88.0,
        launch_angle_deg=45.0,
        initial_air_pressure_psi=40.0,
        discharge_coefficient=0.92,
        drag_coefficient=0.5,
        air_temperature_c=29.0,
        air_density=ambient_density,
        polytropic_exponent=1.2,
    )

    predicted_range = compute_range(params)

    assert math.isclose(predicted_range, REFERENCE_RANGE, rel_tol=0.02)

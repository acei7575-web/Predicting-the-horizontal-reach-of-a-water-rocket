"""Water rocket flight simulation.

This module provides a time-domain simulator for a two-dimensional water rocket.
The model resolves the thrust produced while water is expelled, includes the
mass variation due to the remaining water, and accounts for aerodynamic drag and
gravity.  The primary output is the horizontal range of the rocket, but the full
state history is also returned for further analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Dict, List


R_AIR = 287.058  # Specific gas constant for dry air (J/(kg·K))
GAMMA_AIR = 1.4
RHO_WATER = 1000.0  # kg/m^3
P_ATM = 101_325.0  # Pa
PSI_TO_PA = 6_894.75729  # Exact conversion for lbf/in^2 to Pa


@dataclass
class LaunchConditions:
    """Metadata describing the environment of a particular launch."""

    location: str
    date_time: datetime
    temperature_c: float
    ambient_pressure_pa: float = P_ATM
    wind_speed_m_per_s: float = 0.0


def estimate_air_density(temperature_c: float, *, pressure_pa: float = P_ATM) -> float:
    """Estimate air density from temperature and pressure assuming dry air."""

    temperature_k = temperature_c + 273.15
    return pressure_pa / (R_AIR * temperature_k)


@dataclass
class RocketParameters:
    """Container for the physical properties of the water rocket.

    Attributes
    ----------
    dry_mass:
        Mass of the empty rocket (kg).
    water_volume:
        Initial volume of the water inside the bottle (m^3).
    bottle_volume:
        Total interior volume of the pressure vessel (m^3).
    nozzle_diameter:
        Diameter of the nozzle exit (m).
    discharge_coefficient:
        Empirical discharge coefficient (dimensionless, typically 0.8--1.0).
    drag_coefficient:
        Quadratic drag coefficient referenced to ``cross_sectional_area``.
    cross_sectional_area:
        Reference area used for the drag calculation (m^2).
    initial_air_pressure:
        Absolute pressure of the compressed air at launch (Pa).
    air_temperature:
        Absolute temperature of the compressed air (K).  Used to estimate the
        initial mass of the trapped air.
    launch_angle_deg:
        Launch rail elevation angle (degrees above the horizontal).
    time_step:
        Integration time step (s).
    max_time:
        Maximum simulation time before aborting (s).
    air_density:
        Ambient air density for aerodynamic drag (kg/m^3).
    gravity:
        Magnitude of the gravitational acceleration (m/s^2).
    """

    dry_mass: float
    water_volume: float
    bottle_volume: float
    nozzle_diameter: float
    discharge_coefficient: float
    drag_coefficient: float
    cross_sectional_area: float
    initial_air_pressure: float
    air_temperature: float = 300.0
    launch_angle_deg: float = 45.0
    time_step: float = 1e-3
    max_time: float = 30.0
    air_density: float = 1.225
    gravity: float = 9.80665


def simulate_flight(parameters: RocketParameters) -> Dict[str, List[float]]:
    """Simulate the flight of a water rocket.

    Parameters
    ----------
    parameters:
        Fully-populated :class:`RocketParameters` describing the rocket.

    Returns
    -------
    dict
        Time history of the simulation with the following keys:

        ``"time"`` (s), ``"x"`` (m), ``"y"`` (m), ``"vx"`` (m/s), ``"vy"`` (m/s),
        ``"mass"`` (kg), ``"pressure"`` (Pa) and ``"thrust"`` (N).

        The final entry of the ``"x"`` series corresponds to the interpolated
        horizontal range when the rocket returns to ground level.

    Raises
    ------
    ValueError
        If the supplied parameters are physically inconsistent.
    """

    params = parameters
    if params.bottle_volume <= 0:
        raise ValueError("Bottle volume must be positive.")
    if params.water_volume < 0 or params.water_volume >= params.bottle_volume:
        raise ValueError("Water volume must be within (0, bottle_volume).")
    if params.initial_air_pressure <= P_ATM:
        raise ValueError("Initial air pressure must exceed atmospheric pressure.")

    orientation = math.radians(params.launch_angle_deg)
    nozzle_area = math.pi * (params.nozzle_diameter * 0.5) ** 2

    water_mass = RHO_WATER * params.water_volume
    initial_air_volume = params.bottle_volume - params.water_volume
    initial_air_mass = (
        params.initial_air_pressure
        * initial_air_volume
        / (R_AIR * params.air_temperature)
    )

    total_mass = params.dry_mass + water_mass + initial_air_mass
    if total_mass <= 0:
        raise ValueError("Total mass must be positive.")

    time = 0.0
    x = 0.0
    y = 0.0
    vx = 0.0
    vy = 0.0

    time_history: List[float] = [time]
    x_history: List[float] = [x]
    y_history: List[float] = [y]
    vx_history: List[float] = [vx]
    vy_history: List[float] = [vy]
    mass_history: List[float] = [total_mass]
    pressure_history: List[float] = [params.initial_air_pressure]
    thrust_history: List[float] = [0.0]

    current_water_volume = params.water_volume
    current_air_pressure = params.initial_air_pressure

    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)

    while time < params.max_time:
        dt = params.time_step

        # Update orientation: align thrust with current velocity when moving.
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            thrust_dir_x = vx / speed
            thrust_dir_y = vy / speed
        else:
            thrust_dir_x = cos_theta
            thrust_dir_y = sin_theta

        thrust = 0.0
        current_pressure = P_ATM

        if water_mass > 1e-9:
            current_air_volume = params.bottle_volume - current_water_volume
            current_air_volume = max(current_air_volume, 1e-9)
            current_pressure = params.initial_air_pressure * (
                initial_air_volume / current_air_volume
            ) ** GAMMA_AIR

            if current_pressure > P_ATM:
                exit_velocity = math.sqrt(2.0 * (current_pressure - P_ATM) / RHO_WATER)
                mass_flow_rate = (
                    params.discharge_coefficient
                    * RHO_WATER
                    * nozzle_area
                    * exit_velocity
                )

                max_mdot = water_mass / dt
                mass_flow_rate = min(mass_flow_rate, max_mdot)

                thrust = mass_flow_rate * exit_velocity
                delta_mass = mass_flow_rate * dt

                water_mass = max(water_mass - delta_mass, 0.0)
                current_water_volume = water_mass / RHO_WATER
                total_mass = params.dry_mass + water_mass + initial_air_mass
            else:
                current_pressure = P_ATM
                thrust = 0.0
        else:
            total_mass = params.dry_mass + initial_air_mass

        drag_x = 0.0
        drag_y = 0.0
        if speed > 1e-6:
            drag_magnitude = (
                0.5
                * params.air_density
                * params.drag_coefficient
                * params.cross_sectional_area
                * speed
                * speed
            )
            drag_x = -drag_magnitude * vx / speed
            drag_y = -drag_magnitude * vy / speed

        thrust_x = thrust * thrust_dir_x
        thrust_y = thrust * thrust_dir_y

        gravity_force = -total_mass * params.gravity

        ax = (thrust_x + drag_x) / total_mass
        ay = (thrust_y + drag_y + gravity_force) / total_mass

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        time += dt

        time_history.append(time)
        x_history.append(x)
        y_history.append(y)
        vx_history.append(vx)
        vy_history.append(vy)
        mass_history.append(total_mass)
        pressure_history.append(current_pressure)
        thrust_history.append(thrust)

        if y < 0.0 and time > 0.0:
            break

    if y < 0.0 and len(y_history) >= 2:
        y_prev = y_history[-2]
        x_prev = x_history[-2]
        t_prev = time_history[-2]

        frac = y_prev / (y_prev - y)
        x_touchdown = x_prev + frac * (x - x_prev)
        t_touchdown = t_prev + frac * (time - t_prev)

        time_history[-1] = t_touchdown
        x_history[-1] = x_touchdown
        y_history[-1] = 0.0
        vx_history[-1] = vx_history[-2] + frac * (vx_history[-1] - vx_history[-2])
        vy_history[-1] = vy_history[-2] + frac * (vy_history[-1] - vy_history[-2])
    else:
        x_touchdown = x_history[-1]

    result = {
        "time": time_history,
        "x": x_history,
        "y": y_history,
        "vx": vx_history,
        "vy": vy_history,
        "mass": mass_history,
        "pressure": pressure_history,
        "thrust": thrust_history,
        "range": [x_touchdown],
    }
    return result


def compute_range(parameters: RocketParameters) -> float:
    """Return only the horizontal range of the rocket."""

    history = simulate_flight(parameters)
    return history["range"][0]


def build_parameters_from_measurements(
    *,
    nose_mass_g: float,
    structure_mass_g: float,
    water_volume_ml: float,
    bottle_volume_ml: float,
    nozzle_diameter_mm: float,
    body_diameter_mm: float,
    launch_angle_deg: float,
    initial_air_pressure_psi: float,
    discharge_coefficient: float = 0.92,
    drag_coefficient: float = 0.5,
    air_temperature_c: float = 25.0,
    time_step: float = 5e-4,
    max_time: float = 30.0,
    air_density: float = 1.2,
    gravity: float = 9.80665,
) -> RocketParameters:
    """Convert practical build measurements to :class:`RocketParameters`.

    The helper interprets the supplied values in the units commonly recorded
    during a launch preparation and returns a :class:`RocketParameters`
    instance ready for the numerical solver.

    Parameters
    ----------
    nose_mass_g, structure_mass_g:
        Mass of the detachable payload/nose and of the remaining dry structure
        (bottle shell, fins, glue, etc.) in grams.
    water_volume_ml, bottle_volume_ml:
        Filled water volume and the total internal capacity of the pressure
        vessel in millilitres.
    nozzle_diameter_mm, body_diameter_mm:
        Diameter of the nozzle throat and the main body diameter in millimetres.
    launch_angle_deg:
        Elevation angle of the launch guide relative to the horizon.
    initial_air_pressure_psi:
        Gauge pressure of the compressed air measured in pounds per square
        inch.  The conversion accounts for atmospheric pressure to yield the
        absolute pressure required by the thermodynamic model.
    discharge_coefficient:
        Empirical discharge coefficient for the nozzle.
    drag_coefficient:
        Aerodynamic drag coefficient referenced to the frontal area defined by
        ``body_diameter_mm``.
    air_temperature_c:
        Launch-day air temperature in degrees Celsius.
    time_step:
        Integration time step for :func:`simulate_flight`.
    max_time:
        Maximum simulation duration; the integrator halts early when the rocket
        returns to the ground.
    air_density:
        Free-stream air density used by the drag model.
    gravity:
        Local gravitational acceleration.
    """

    dry_mass = (nose_mass_g + structure_mass_g) / 1000.0
    water_volume = water_volume_ml / 1_000_000.0
    bottle_volume = bottle_volume_ml / 1_000_000.0
    nozzle_diameter = nozzle_diameter_mm / 1000.0
    body_diameter = body_diameter_mm / 1000.0
    cross_sectional_area = math.pi * (body_diameter * 0.5) ** 2

    air_temperature_k = air_temperature_c + 273.15
    initial_air_pressure = P_ATM + initial_air_pressure_psi * PSI_TO_PA

    return RocketParameters(
        dry_mass=dry_mass,
        water_volume=water_volume,
        bottle_volume=bottle_volume,
        nozzle_diameter=nozzle_diameter,
        discharge_coefficient=discharge_coefficient,
        drag_coefficient=drag_coefficient,
        cross_sectional_area=cross_sectional_area,
        initial_air_pressure=initial_air_pressure,
        air_temperature=air_temperature_k,
        launch_angle_deg=launch_angle_deg,
        time_step=time_step,
        max_time=max_time,
        air_density=air_density,
        gravity=gravity,
    )


def calibrate_parameters(
    base_parameters: RocketParameters,
    target_range: float,
    search_specs: List[Dict[str, float]],
) -> Dict[str, float]:
    """Sequentially tune selected parameters to reduce range error.

    Parameters
    ----------
    base_parameters:
        Initial :class:`RocketParameters` generated from measurement data.
    target_range:
        Recorded range on the launch field (m).
    search_specs:
        Sequence of dictionaries describing the parameter search bounds.  Each
        entry must include ``"name"``, ``"min"``, ``"max"`` and optionally
        ``"steps"`` to control the resolution of the grid search.

    Returns
    -------
    dict
        Mapping of parameter names to the tuned values.
    """

    tuned_parameters = base_parameters
    current_range = compute_range(tuned_parameters)

    tuned_values: Dict[str, float] = {}

    for spec in search_specs:
        name = spec["name"]
        min_value = spec["min"]
        max_value = spec["max"]
        steps = max(int(spec.get("steps", 20)), 1)

        best_value = getattr(tuned_parameters, name)
        best_range = current_range
        best_error = abs(best_range - target_range)

        for step in range(steps + 1):
            fraction = step / steps
            candidate_value = min_value + (max_value - min_value) * fraction
            candidate_parameters = replace(tuned_parameters, **{name: candidate_value})
            candidate_range = compute_range(candidate_parameters)
            candidate_error = abs(candidate_range - target_range)

            if candidate_error < best_error:
                best_error = candidate_error
                best_value = candidate_value
                best_range = candidate_range

        tuned_parameters = replace(tuned_parameters, **{name: best_value})
        current_range = best_range
        tuned_values[name] = best_value

    return tuned_values


if __name__ == "__main__":
    launch_conditions = LaunchConditions(
        location="대한민국",
        date_time=datetime(2025, 9, 5, 16, 0, 0),
        temperature_c=29.0,
        wind_speed_m_per_s=0.0,
    )

    ambient_air_density = estimate_air_density(
        launch_conditions.temperature_c, pressure_pa=launch_conditions.ambient_pressure_pa
    )

    # Launch scenario supplied by the build notes in the prompt.
    nose_mass_g = 60.0
    scenario_parameters = build_parameters_from_measurements(
        nose_mass_g=nose_mass_g,
        structure_mass_g=50.0,  # 측정값이 아닌 추정치 (보정 대상)
        water_volume_ml=385.0,
        bottle_volume_ml=550.0,
        nozzle_diameter_mm=22.0,
        body_diameter_mm=65.0,
        launch_angle_deg=45.0,
        initial_air_pressure_psi=40.0,
        discharge_coefficient=0.92,
        drag_coefficient=0.5,
        air_temperature_c=launch_conditions.temperature_c,
        air_density=ambient_air_density,
    )

    result = simulate_flight(scenario_parameters)
    range_estimate = result["range"][0]
    actual_range_m = 89.0
    range_error = range_estimate - actual_range_m
    percent_error = abs(range_error) / actual_range_m * 100.0
    print("시나리오: 탄두 60 g, 물 385 mL, 발사각 45°, 게이지 공기압 40 psi")
    print(
        "발사 조건: {loc}, {dt}, 기온 {temp:.1f}°C, 바람 {wind:.1f} m/s".format(
            loc=launch_conditions.location,
            dt=launch_conditions.date_time.strftime("%Y-%m-%d %A %H:%M"),
            temp=launch_conditions.temperature_c,
            wind=launch_conditions.wind_speed_m_per_s,
        )
    )
    print(f"추정 대기 밀도: {ambient_air_density:.3f} kg/m³")
    print(f"모의 수평 도달 거리(보정 전): {range_estimate:.2f} m")
    print(f"비행 시간(보정 전): {result['time'][-1]:.2f} s")
    print(f"실측 수평 도달 거리: {actual_range_m:.2f} m")
    print(f"오차: {range_error:+.2f} m")
    print(f"오차율: {percent_error:.2f}%")
    print("사용자 제공 입력값(변경 없음):")
    print(f" - 탄두 질량: {nose_mass_g:.0f} g")
    print(" - 주입수량: 385 mL")
    print(" - 발사각: 45°")
    print(" - 게이지 공기압: 40 psi")

    calibration_specs = [
        # 사용자가 제공하지 않은, 측정 오차가 크기 쉬운 변수만 보정 대상에 포함
        {"name": "drag_coefficient", "min": 0.35, "max": 0.75, "steps": 24},
        {"name": "discharge_coefficient", "min": 0.88, "max": 0.97, "steps": 24},
        {"name": "dry_mass", "min": 0.09, "max": 0.13, "steps": 24},
    ]

    tuned_values = calibrate_parameters(scenario_parameters, actual_range_m, calibration_specs)
    tuned_parameters = replace(scenario_parameters, **tuned_values)
    tuned_result = simulate_flight(tuned_parameters)
    tuned_range = tuned_result["range"][0]
    tuned_error = tuned_range - actual_range_m
    tuned_percent_error = abs(tuned_error) / actual_range_m * 100.0

    label_map = {
        "drag_coefficient": "항력 계수",
        "discharge_coefficient": "노즐 방출 계수",
        "dry_mass": "구조체 질량(탄두 제외)",
    }

    def describe_adjustment(name: str, before: float, after: float) -> str:
        delta = after - before
        if name == "dry_mass":
            structure_before = (before - nose_mass_g / 1000.0) * 1000.0
            structure_after = (after - nose_mass_g / 1000.0) * 1000.0
            return (
                f"{label_map[name]}: {structure_before:.1f} g → {structure_after:.1f} g "
                f"(변화량 {structure_after - structure_before:+.1f} g)"
            )
        return f"{label_map[name]}: {before:.3f} → {after:.3f} (변화량 {delta:+.3f})"

    print("\n오차율 최소화를 위한 변수 조정 결과:")
    for spec in calibration_specs:
        name = spec["name"]
        before = getattr(scenario_parameters, name)
        after = tuned_values[name]
        print(" - " + describe_adjustment(name, before, after))

    print(f"\n보정 후 모의 수평 도달 거리: {tuned_range:.2f} m")
    print(f"보정 후 비행 시간: {tuned_result['time'][-1]:.2f} s")
    print(f"보정 후 오차: {tuned_error:+.2f} m")
    print(f"보정 후 오차율: {tuned_percent_error:.2f}%")
    print("※ 보정 과정에서도 사용자 입력값(탄두 질량 60 g, 물 385 mL, 발사각 45°, 공기압 40 psi)은 그대로 유지했습니다.")
